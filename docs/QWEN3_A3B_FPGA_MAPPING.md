# Qwen3.6-35B-A3B → FPGA Mapping

Target: run [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
end-to-end on AWS F2 (Virtex UltraScale+ VU47P, 16GB HBM2, ~460 GB/s) and beat
g6/g6e GPU on `$/1M tokens` and `joule/token` for batch=1 decode.

This doc is a *mapping plan*, not committed kernels. Numbers marked `*` are
public-spec upper bounds; substitute measured values once silicon work starts.

## 1. Model Shape (assumed)

| Item                    | Value (assumed, verify on release) |
|-------------------------|--------------------------------------|
| Total params            | ~35 B                                |
| Active params per token | ~3 B                                 |
| Architecture            | Decoder-only MoE                     |
| Layers                  | 48                                   |
| Hidden                  | 4096                                 |
| KV heads (GQA)          | 8                                    |
| Q heads                 | 32                                   |
| Head dim                | 128                                  |
| Experts / layer         | 128                                  |
| Active experts          | 8 (top-k)                            |
| Expert hidden           | 1408                                 |
| Vocab                   | ~152k                                |
| RoPE                    | yes                                  |
| Norm                    | RMSNorm                              |
| Activation              | SwiGLU                               |

All assumed shapes feed `src/fllm/cost_model.py`. Update once the weights are
on HuggingFace and `config.json` is readable.

## 2. Why Batch=1 Decode Is Mem-Bound

For batch=1 decode each generated token streams every *active* weight exactly
once. At INT4:

```text
active_weight_bytes = 3.0e9 * 4 / 8 = 1.5 GB
```

Per-token KV traffic for a 1k-context decode:

```text
kv_bytes_per_step = 2 (k,v) * num_kv_heads * head_dim * dtype_bytes * num_layers
                  ~= 2 * 8 * 128 * 2 * 48 = 196 KB read+write
kv_traffic_at_1k_ctx ~= 196 KB * 1024 ~= 200 MB
```

So **~1.7 GB** must move through HBM per generated token. Ceilings:

| Device                 | HBM BW (GB/s)* | Peak tok/s | Real tok/s @ eff |
|------------------------|---------------:|-----------:|------------------:|
| AWS f2 VU47P (FPGA)    |   460          | ~270       | ~190 @ 0.70       |
| AWS g6 L4              |   300          | ~176       | ~97  @ 0.55       |
| AWS g6e L40S           |   864          | ~508       | ~280 @ 0.55       |
| AWS p5 H100 (per GPU)  |  3350          | ~1970      | ~1080 @ 0.55      |

(Run `scripts/cost_report.py` for current numbers and `$/1M tokens`.)

The FPGA does **not** win on raw tokens/s versus H100. The thesis is:

1. **Cost**: AWS f2 is far cheaper than p5 per device-hour and tracks g6/g6e.
2. **Energy**: VU47P TDP ~225 W vs L40S 350 W and H100 SXM 700 W; INT4 matvec
   uses LUT/DSP packed multipliers (no FP overhead).
3. **Latency floor**: deterministic token loop, no kernel launch jitter.

## 3. Memory Plan (VU47P, 16 GB HBM2)

35 B INT4 weights = **17.5 GB**. Single VU47P **does not fit** all weights. Two
viable paths:

### Path A: 2-FPGA shard (preferred; matches `f2.12xlarge`)

- Shard experts across 2 FPGAs (64 experts / FPGA).
- Shared (non-expert) weights replicated.
- Router runs on FPGA 0; dispatch over PCIe peer link or chip-to-chip via host.
- Each active token uses top-k experts, ~half land on each FPGA on average →
  load-balance auxiliary loss during fine-tune.

### Path B: 1-FPGA + DDR-staged experts

- Hot expert cache in HBM (~32 experts), cold experts streamed from DDR.
- Workable only if expert reuse locality is high. Needs trace analysis.
- Risk: DDR BW (~30 GB/s) kills the win. Use only as fallback.

## 4. Kernel Inventory

Build order (each kernel ships with cycle-accurate testbench vs Python ref):

1. **`matvec_int4`** — packed INT4 × INT8, INT32 accum, INT8 requant out.
   Per-row scales loaded from BRAM. This is the bottleneck primitive.
2. **`rmsnorm_fxp`** — fixed-point RMSNorm. Reciprocal-sqrt via LUT.
3. **`rope_apply`** — RoPE rotation on Q/K. Precomputed sin/cos table in BRAM.
4. **`gqa_attention`** — local + compressed-global GQA. KV in HBM, queries on-chip.
5. **`moe_router`** — top-k routing, INT8 logits, hardware top-k tree.
6. **`expert_dispatch`** — token-to-expert grouping; uses sparse mask buffer.
7. **`lm_head`** — chunked argmax over 152k vocab. Streaming `max` reduce.
8. **`token_loop_ctrl`** — autoregressive FSM, holds KV pointer, never returns
   control to host between tokens.

## 5. Quantization Recipe

- Weights: INT4 symmetric, per-row scales (group_size=-1 for first pass; try
  group_size=128 if quality drops > 0.5 ppl).
- Activations: INT8 symmetric per-tensor.
- KV cache: INT8 per-head dynamic scale.
- Router logits: FP16 → INT16 inside FPGA; gates softmax in FP16 LUT.
- Embeddings + LM head: kept INT8 to preserve token-level fidelity.

Validation: `src/fllm/quant.py` simulates this end-to-end. Measure perplexity
on Wikitext-2 + a code subset; FPGA target is ≤ +0.5 ppl vs BF16 reference.

## 6. Fair Benchmark Protocol

Same model, same INT4 weights, same prompts, same greedy decode, on:

- f2.6xlarge (1 VU47P) — fallback path only if Path B works.
- f2.12xlarge (2 VU47P) — primary FPGA target.
- g6.xlarge (L4) — cost-matched GPU.
- g6e.xlarge (L40S) — performance-matched GPU.
- p5.48xlarge (H100) — high-end reference, expected to lead on tok/s.

Report per `docs/BENCHMARK_PLAN.md`: tok/s, ms/tok p50/p99, J/tok, $/1M tok,
quality. The win condition is **$/1M tokens** and **J/tok**, not raw tok/s.

## 7. Open Risks

- VU47P HBM is HBM2 (slower than H100 HBM3); the cost win requires the price
  gap to outpace the BW gap.
- Expert routing imbalance kills shard utilization → need aux load-balance loss.
- KV cache at 32k context grows past on-chip capacity; need INT8 KV + paged HBM.
- AWS F2 hourly pricing changes; rerun `cost_report.py` quarterly.
