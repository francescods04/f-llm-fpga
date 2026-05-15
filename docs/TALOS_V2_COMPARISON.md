# Talos V2 — Why 50k tok/s, And What Of It Transfers To Qwen3-A3B

Talos V2 (Abeykoon & Chhajer, May 2026, DE1-SoC Cyclone V) achieves ~53,000
tokens/second running microGPT (char-level names model) entirely in RTL. This
is a serious accelerator-engineering result. It is also **not** directly
comparable to our Qwen3.6-35B-A3B target. This doc separates the two so we
don't lie to ourselves about the gap.

## 1. The Apples-to-Oranges Floor

microGPT-names (Talos V2 workload):

```
vocab        ~27 (chars + special)
hidden       ~32-128 (typical for makemore)
layers       1-3
ctx          ~16-32 chars
params       ~50K - 500K total
weight bytes ~100KB total (fits 1 BRAM block)
KV cache     trivially small
attention    full O(N^2), N ~= 16
```

Qwen3.6-35B-A3B (our target):

```
vocab        ~152K
hidden       ~4096
layers       48
ctx          up to 32k
total params 35 B
active params per token 3 B
INT4 weight bytes per token 1.5 GB streamed
KV at 1k ctx 200 KB streamed per step
attention    GQA over 32 heads
```

Ratio of *active compute work* per generated token:

```
Qwen3-A3B / microGPT ~= 3e9 / 5e4 ~= 60,000x
Qwen3-A3B / microGPT bytes streamed per token ~= 1.5e9 / 0 ~= infinity
```

So 50,000 tok/s on microGPT and 1,000 tok/s on Qwen3-A3B describe **completely
different physics**. Talos V2 has zero HBM traffic — all weights live in
on-chip M9K ROM. Qwen3-A3B cannot fit even one expert's weights in the entire
DE1-SoC, let alone the whole model.

## 2. What Drives The 50k Number

- 56.25 MHz fabric clock × ~1100 cycles/token ≈ 50k tok/s. So they spend
  ~1100 cycles per generated token. That is the real headline.
- All weights in on-chip ROM via `$readmemh`. Zero memory-system stalls.
- Q4.12 fixed-point throughout. No floating-point cost on Cyclone V (which
  has no FP DSP support comparable to its FPGA cousins).
- One 16-lane streamed-systolic matvec **tile**, time-multiplexed across
  Q, K, V, O, MLP1, MLP2, LM head. Single tile means low area, easy
  timing closure, no routing congestion.
- Operation folding: max-tracking lives in the dot-product pass. LM-head
  scan lives in the projection pass. No idle FSM cycles.
- Hardware categorical sampler with xorshift PRNG. Sampling never leaves
  the chip.
- Bounded multicycle math engines for softmax (exp LUT + saturated divider)
  and RMSNorm (iterative reciprocal). They are *not* combinational. They
  are *not* general-purpose either: the input range is bounded by the
  model, so the engines can be narrow.

## 3. What Does Not Transfer

- **All-weights-in-ROM.** Cannot work for 35B params. We need HBM, and HBM
  traffic dominates our cycle budget.
- **Q4.12 16-bit fixed-point.** Too wide. We are targeting INT4 weights,
  INT8 acts, BFP8 KV. Sub-byte is the whole point at 35B scale.
- **Combinational sweep across all rows.** They get away with this because
  the matrices are small. We will not.
- **A single tile.** One tile at our scale runs out of throughput well
  before HBM saturates. We need a **bank** of tiles, each fed by its own
  HBM channel.

## 4. What Absolutely Transfers (The Real Lessons)

The methodology is the value here, not the number. The principles below
should drive every design decision in `fpga/`:

### 4.1 Time-Multiplex One Known-Good MatVec Tile

Talos V2's single 16-lane tile across 7 layer functions is the right
default. For us, **one parametric tile** at the LANES/TILE_ROWS level,
**replicated K times** to match HBM channel count. Don't write a custom
datapath for each layer. Schedule it.

Action: refactor `fpga/matvec_int4.hpp` to make tile replication explicit
and to document the schedule for QKV/O/up/gate/down/lm_head reuse.

### 4.2 Operation Folding As A Default Reflex

Every FSM idle state is wasted cycles. Talos V2's wins came from folding
bookkeeping into the productive pass. For us:

- Fold per-row scale apply into the matvec output beat (already drafted).
- Fold softmax max-track into the QK^T dot-product pass.
- Fold residual-add into the next-block's RMSNorm read.
- Fold LM-head argmax into the head's matvec output beat.

Action: add a folding checklist to `docs/FPGA_OPTIMIZATIONS.md` so each
kernel must justify why it cannot be folded into the previous stage.

### 4.3 Targeted Parallelism Only At Proven Bottlenecks

Talos V2 explicitly says: more parallelism made things slower. The wins
came from **measuring** which path serialized and then duplicating *that*.
For us, this means: do not unroll an expert MoE into 128 dispatched lanes
just because you can. Profile, then duplicate.

Action: build a cycle-counting Python simulator before any RTL is written.
Add `src/fllm/cycle_sim.py` (future work) that lower-bounds cycles for
each kernel under chosen parallelism.

### 4.4 Bounded Multicycle Engines For Hard Ops

softmax/rsqrt/div should never be one giant combinational block. Talos V2
uses LUT exp + saturated divider for softmax, iterative rsqrt for RMSNorm.
We already plan this — `src/fllm/lut_activations.py` is the Python
reference. The lesson: do not let HLS infer general-purpose math; write
the bounded engine yourself.

Action: add an HLS skeleton for `softmax_engine` matching the Python
shifted-softmax module.

### 4.5 Hardware Sampler With On-Chip PRNG

Talos V2 ships logits to a hardware xorshift PRNG and never returns to
host for sampling. Our `token_loop_ctrl` must do the same.

Action: add `src/fllm/sampler.py` Python reference matching the planned
xorshift PRNG + cumulative-sum sampler.

### 4.6 Close Timing First, Raise Clock Second

DE1-SoC base is 50 MHz; Talos V2 added a PLL to push to 56.25 MHz **only
after** critical paths shrank. For VU47P we target 600 MHz. We should
keep an explicit "fmax slack" budget per kernel and refuse to ship until
all targets close at 500 MHz with positive slack.

### 4.7 Weights In On-Chip Storage Whenever You Can

For us, "on-chip" means URAM/BRAM (~100 MB). 35B weights don't fit, but:
- shared (non-expert) weights ≈ 6-8 GB INT4 — does fit if sharded over 2
  FPGAs but not in URAM. Streams from HBM.
- top-32 hot experts at INT4 ≈ 180 MB — fits across 2 FPGAs' URAM.

So the Talos lesson maps as: **what is "ROM" for Talos = "URAM cache" for
us, and the question is which subset is hot enough to deserve URAM.** This
is exactly our expert-cache-hit-rate parameter in `cost_model.py`.

### 4.8 Determinism Is A Feature, Not An Accident

Talos V2's RTL is deterministic — same input, same cycle count, same
output bits. We should treat the same as a *first-class result*. Report
p99 == p50 latency and use it as a paper contribution. GPUs cannot.

## 5. Concrete Code Changes Triggered By This Doc

- `src/fllm/sampler.py` (new) — xorshift PRNG + cumulative-sum categorical
  sampler. Bit-exact reference for the FPGA hardware sampler.
- `fpga/matvec_int4.hpp` — comment header explaining tile reuse schedule
  (Q, K, V, O, up, gate, down, lm_head all dispatch the same instance).
- `docs/FPGA_OPTIMIZATIONS.md` — add §11 Operation-Folding Checklist.
- future `src/fllm/cycle_sim.py` — cycle-accurate Python estimator per
  kernel under chosen parallelism. Replaces the current HBM-only roofline.

## 6. Honest Summary

Talos V2 is the right *engineering style* for FPGA-native LLMs. The 50k
tok/s number is a small-model artifact, not a scaling claim. Our project
takes Talos V2's discipline (one good tile, fold ops, bounded engines,
hardware sampler, close timing first) and stretches it across HBM-attached
35B MoE inference. The question we are answering is the one Talos V2
doesn't have to: **what happens when the weights don't fit on-chip and
HBM bandwidth becomes the floor instead of the cycle count.**
