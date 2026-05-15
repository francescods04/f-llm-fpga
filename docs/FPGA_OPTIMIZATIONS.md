# FPGA-Native Efficiency Wins for Qwen3.6-35B-A3B Decode

Author hat: senior FPGA + ML systems engineer.

This is the core research thesis: a GPU is a fixed datapath optimized for dense
FP/INT GEMM. Batch=1 MoE decode wastes most of that datapath. The FPGA wins
only if we exploit things the GPU **physically cannot do**, not by porting
CUDA kernels.

Every item below has: (a) what GPU *cannot* do, (b) math/back-of-envelope, (c)
expected win, (d) implementation cost, (e) where it lives in our kernel set.

---

## 1. Custom Sub-Byte Datapath (INT4 × INT8 native, MX4, log-domain)

**GPU limit.** Hopper/Ada commodity tensor cores do INT8 and FP8. INT4 is
"supported" in spec but rarely yields full throughput; routines dequantize to
INT8 internally on most GPUs. MX-formats only land on Blackwell (B200) and
above.

**FPGA datapath.** Each DSP48E2 on VU47P does one (27×18 signed) multiply per
cycle. Packed differently, it does:

```
1 DSP cycle = 1×(27b×18b)
            = 2×(18b×8b)  via INT8 mode
            = 3×(8b×8b)   via SIMD packing
            = 4×(4b×4b)   via LUT-assisted packing (vendor IP exists)
```

So one DSP yields **~4 INT4 MACs/cycle** at 600 MHz ≈ 2.4 GMAC/s/DSP.
VU47P has ~9k DSPs ⇒ ~21.6 INT4 TOPS from DSPs alone, plus ~8 TOPS from
LUT-based INT4 multipliers (free LUTs not used elsewhere).

**Win.** Compute is not the bottleneck at batch=1 (memory is), but compute
must keep up with HBM. At 460 GB/s × 8b/B INT4 = **920 G-INT4-loads/s**;
30 TOPS easily absorbs it. GPU at INT4 throughput often *under-runs* its own
HBM at small batches.

**Cost.** Vendor DSP SIMD pack is well-documented HLS pragma. Low risk.

**Lives in.** `matvec_int4`. Stretch: `matvec_mx4`, `matvec_log` ablations.

---

## 2. Spatial Dataflow (No Kernel Launches, No HBM Round-Trip Between Ops)

**GPU limit.** Decode of one transformer block on GPU = 8–20 kernel launches
(qkv proj, RoPE, softmax, matmul, output proj, RMSNorm, gate, up, silu, down,
norm, …). Each launch: ~5–20 µs CUDA overhead + a full HBM round-trip of the
hidden state. For batch=1, **launch overhead alone** can equal compute time.

**FPGA datapath.** Build the entire block as one spatial pipeline. Hidden
state never leaves on-chip URAM between ops within a block. Only KV and
weight reads touch HBM.

```
URAM(hidden) → RMSNorm → QKV proj → RoPE → KV write → GQA core
                                                 ↓
                              softmax(LUT) → AV → out proj
                                                       ↓
                              URAM(hidden) → RMSNorm → MoE → URAM(hidden)
```

**Win.** 4096-d hidden × FP16 = 8 KB; saving 20 reads/writes per block × 48
blocks = ~7.7 MB/token off HBM. At 460 GB/s that's **17 µs/token** saved
versus a naive port. More importantly: **zero launch jitter**, so p99 = p50.

**Cost.** High. Requires HLS dataflow region with `#pragma HLS DATAFLOW`,
stream FIFOs between stages, careful II=1 pipelining. This is the
hardest-but-highest-value win.

**Lives in.** `block_pipeline` (composite kernel).

---

## 3. Hardware Top-K Router (Bitonic Sort, O(log² N))

**GPU limit.** Top-k on 128 logits per token is a warp-level reduction with
~log N synchronizations. At batch=1 it serializes badly.

**FPGA datapath.** Bitonic sorter for N=128 = 7 stages × 6 compare-swap
banks. Combinational depth ~50 LUTs, latency ~7 cycles fully pipelined.
Throughput: 1 token's top-k per cycle.

**Win.** Routing is sub-1% of total time, so this is not the bottleneck. But
it lets the dispatch logic make **per-cycle decisions**, which enables
streaming token-batches into the expert pipe without stalls.

**Cost.** Low. Templated bitonic sort exists in Vitis HLS examples.

**Lives in.** `moe_router`.

---

## 4. On-Chip Expert Caching (Resident Hot Experts + Cold Streaming)

**GPU limit.** GPU treats MoE as: read all 8 expert weights from HBM per
token, regardless of recency. No general-purpose cache for HBM.

**FPGA datapath.** VU47P has ~64 MB URAM + ~36 MB BRAM = **~100 MB on-chip
SRAM**. Per-expert INT4 weight size ≈ `2 * 4096 * 1408 * 4b / 8 = 5.6 MB`.
So ~17 experts fit on-chip at INT4 per FPGA (or ~34 with shared up/gate).

Use offline routing trace from validation set: rank experts by activation
probability. Pin top-K residents in URAM. Cold experts stream from HBM.

If 60% of token-expert dispatches hit a resident expert:

```
effective_active_bytes/token = 8 active × (0.4 × 5.6 MB) = 17.9 MB
                              vs naive 8 × 5.6 = 44.8 MB
```

**Win.** ~2.5× reduction in per-token weight bytes ⇒ ~2.5× tok/s ceiling.
For Qwen3.6-35B-A3B this is the **single biggest FPGA-only optimization**.

**Cost.** Medium. Needs offline routing analysis + URAM partitioning + a
small directory table per FPGA. Auxiliary load-balance loss during
fine-tune helps keep hit rate predictable.

**Lives in.** `expert_dispatch` + `expert_cache_controller`.

---

## 5. Block-Floating-Point KV Cache (BFP4/BFP8)

**GPU limit.** GPU KV cache is FP16 or INT8 with per-tensor scale. Per-tensor
INT8 KV drops perplexity noticeably at long context. Per-channel adds memory
overhead. GPU lacks the bit-granularity to do real BFP.

**FPGA datapath.** BFP8: 8-bit signed mantissa + 8-bit shared exponent per
block of 32 lanes. Effective: ~8.25 bits/value, **dynamic range ≈ FP16**,
quality ≈ FP16, storage ≈ INT8.

For Qwen3-A3B (8 KV heads × 128 dim × 48 layers × 2 (K,V) × 2 (high/low ctx)):

```
FP16 KV / token = 2 × 8 × 128 × 2 × 48 = 196 KB
BFP8 KV / token ≈ 100 KB (mantissa) + 24 KB / 32 (exponent) ≈ 101 KB
```

**Win.** ~2× KV memory + ~2× KV bandwidth. At 32k context this is the
difference between fits-on-chip and doesn't. Quality drop ≈ 0.0–0.1 ppl
based on prior BFP literature (Microsoft MX, Bondarenko et al.).

**Cost.** Medium. On-read dequant is `mantissa << exp`; on-write requantize
needs block-max reduction (≤32-wide adder tree, 1 cycle).

**Lives in.** `kv_pack` and `kv_unpack` engines inside `gqa_attention`.

---

## 6. LUT-Based Transcendentals (SiLU, Softmax, RMSNorm rsqrt)

**GPU limit.** SiLU, exp, rsqrt go through SFU pipes. Throughput ~1/4 of
mul/add. Fixed datatype (FP16/FP32). Not user-replaceable.

**FPGA datapath.** Each function becomes a piecewise-linear LUT in a single
BRAM (18 Kb). For 1024 entries:

| Function | LUT entries | Max abs error | BRAM cost |
|---|---:|---:|---:|
| SiLU(x), x∈[−8,8] | 1024 | ~3e-4 | 1 BRAM |
| softmax exp shim | 256 | ~1e-3 | 1 BRAM |
| rsqrt for RMSNorm | 512 | ~2e-4 | 1 BRAM |

**Win.** Constant 1-cycle latency, zero pipeline stalls, no FP. Maybe 10%
overall throughput win, but **huge** for energy: avoid FP SFU power.

**Cost.** Low. Precompute table at bitstream gen time. Validate at Python
level (`src/fllm/lut_activations.py`).

**Lives in.** `silu_lut`, `softmax_shifted`, `rmsnorm_fxp`.

---

## 7. Hardware Token-Loop FSM (No Host Roundtrip)

**GPU limit.** Greedy decode = one CUDA stream cycle per token: host samples
next token, copies it back as input. Even with CUDA Graphs the host is in
the loop for sampling control. Minimum ~50–200 µs/token of host overhead.

**FPGA datapath.** Token loop is an FSM. State = KV pointer, last token id,
RoPE position. Argmax is one cycle. Generated tokens stream to host via
AXI-Stream; host has zero per-token responsibility until EOS or max-len.

**Win.** At 5 ms/token target, removing 200 µs host overhead = **4%**
direct. But it also kills tail latency: no GC pause, no PCIe contention.

**Cost.** Low-medium. State machine + KV ring buffer + AXI-Stream output.

**Lives in.** `token_loop_ctrl`.

---

## 8. HBM Channel-Aware Weight Placement

**GPU limit.** GPU HBM is opaque: hardware decides which channel a byte
lives in. Address-bit hashing approximates uniform spread, but you cannot
guarantee parallel access across all channels.

**FPGA datapath.** VU47P HBM2 = 32 pseudo-channels × ~14 GB/s each. The
compiler exposes channels as separate AXI master ports. We choose layout.

Strategy: stripe each expert's weights across all 32 channels so the matvec
engine pulls from 32 channels in parallel for one expert. Or shard by
expert: 4 experts × 8 channels each, run 4 dispatch lanes concurrently.

**Win.** Realized HBM efficiency 70–80% achievable vs ~50% on GPU at
batch=1. So the 460 GB/s spec is closer to delivered than 3.35 TB/s is on
H100 at batch=1.

**Cost.** Medium. Floorplan + AXI interconnect topology decisions made at
synthesis. Affects timing closure.

**Lives in.** `hbm_layout` (allocation policy file consumed at build).

---

## 9. Speculative Decoding Co-Resident (Draft Model in BRAM)

**GPU limit.** Speculative decoding works on GPU but synchronizes draft and
target on the same SM scheduler. Draft kernel launches still cost host
time. Batch=1 draft doesn't pipeline well.

**FPGA datapath.** Put a tiny draft model (e.g. 100M params INT4 = 50 MB) in
BRAM/URAM on FPGA 0. Run k speculative tokens in 1 ms while FPGA 1 verifies
the previous chunk with the target model. Two FPGAs = two physical pipes,
zero sharing contention.

**Win.** With acceptance rate ~70% on technical/general text, expect
**1.5–2.5× effective tok/s** on top of all above.

**Cost.** High. Needs draft model training, tree-attention verification,
careful KV synchronization. Phase 6+ work.

**Lives in.** Future `draft_engine` on FPGA0.

---

## 10. Quantization-Aware Sparsity (Beyond 2:4)

**GPU limit.** Ampere/Hopper sparse tensor cores only accelerate 2:4 (50%
zeros, structured). Other patterns either run dense or run slower than
dense.

**FPGA datapath.** Choose any structured sparsity: 1:4 (75% zeros), 1:8,
unstructured-within-group. Skip the zero rows in the matvec engine: each
skipped row = saved HBM read.

If we get Qwen3-A3B experts down to 1:4 with minimal quality loss (open
question — needs research), per-token expert bytes drop another 2×.

**Win.** Speculative: another **2× tok/s** stacked on expert caching. Risk:
quality. Needs careful retraining.

**Cost.** High research, low hardware. Hardware just adds a "zero-row skip"
read controller; the *retraining* is the cost.

**Lives in.** Future `sparse_expert` ablation.

---

## Stacked Win Estimate

Compose realistically (multiplicative, with diminishing returns):

| Optimization | Standalone win | Stackable factor |
|---|---:|---:|
| INT4 packed datapath (baseline) | — | 1.00× |
| On-chip expert caching (60% hit) | 2.5× | 2.5× |
| BFP8 KV at long ctx | 1.4× | 3.5× |
| Spatial dataflow (no kernel/HBM jitter) | 1.15× | 4.0× |
| Hardware token loop (host eliminated) | 1.04× | 4.2× |
| HBM channel-aware layout (50→75% eff) | 1.5× | **6.3×** |
| Speculative co-resident (Phase 6+) | 1.8× | 11.3× (stretch) |

Take the conservative 6.3× over the naive cost-model number from `cost_report.py`:

```
naive f2 VU47P tok/s ~169 → realistic FPGA-tuned ~1060
```

That puts a single 2× VU47P shard near H100 SXM tok/s **at a fraction of the
cost and likely lower J/tok**, which is the actual research result we want.

---

## 11. Operation-Folding Checklist (Talos V2 Lesson)

Every FSM idle state is a wasted cycle. Before declaring a kernel done, prove
that none of the following folds are possible:

- [ ] Per-row weight scale apply folded into matvec output beat?
- [ ] Softmax `max` tracked inside the QK^T dot pass?
- [ ] Residual-add folded into the next block's RMSNorm read?
- [ ] LM-head argmax folded into the head's projection output beat?
- [ ] RoPE rotation folded with Q/K projection write?
- [ ] KV BFP block-max tracked during V write (one fewer pass at read time)?
- [ ] Top-k router compare-swap folded into the router logits output beat?
- [ ] Expert dispatch mask computed during routing softmax (no extra pass)?

If a fold is not possible, write *why* in the kernel header. Folding is the
default; not folding is the exception.

## What This Means For The Software Reference

Each optimization above has a **Python proxy** so the architecture can be
validated before any RTL:

- `src/fllm/quant.py` — INT4/INT8 fake-quant ✓
- `src/fllm/lut_activations.py` — LUT SiLU/softmax (this PR)
- `src/fllm/bfp.py` — BFP8 KV simulation (this PR)
- `src/fllm/cost_model.py` — extended with on-chip reuse (this PR)
- future `src/fllm/sparse.py` — N:M sparsity simulation
- future `src/fllm/draft.py` — speculative decode harness

Each must answer: **what is the quality cost?** If 6.3× speed costs >+1 ppl,
the win is not real. Run `scripts/quantize_eval.py` style ablations for each.
