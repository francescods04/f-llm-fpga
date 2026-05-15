# Plan: Beat H100 on Qwen3.6-35B-A3B Decode

PhD-grade research execution plan. Target hardware: AWS f2.12xlarge (2× VU47P).
Goal: end-to-end measurable win on tok/s **and** J/token vs p5 H100 SXM, with
quality cost ≤ +1.0 ppl on Wikitext-2 + MMLU.

This is the binding document for project execution. Every claim points to a
file path, a script, or a measurement gate. No hand-wavy stages.

---

## 0. Target Definition (frozen)

| Quantity                | Value                  | Source                  |
|-------------------------|------------------------|-------------------------|
| Model                   | Qwen3.6-35B-A3B INT4   | `docs/QWEN3_A3B_FPGA_MAPPING.md` |
| Target tok/s (2× VU47P) | **≥ 3500** (moderate) / **9000** (aggressive) | §6 math |
| Target J/token          | **≤ 0.10** (moderate)                          | §6 math |
| Target $/1M tok         | **≤ $0.50**                                    | §6 math |
| Max Δppl vs BF16 ref    | **+1.0 ppl** Wikitext-2; **−2 pts** MMLU max  | §5 budget |
| Decode batch            | 1                       | by design |
| Context length          | 1k primary, 4k stretch  | §11 |
| H100 baseline (real)    | **measured**, not spec  | §1 mandatory |

**Win condition**: 2× VU47P beats H100 SXM on tok/s **and** J/token at the
moderate quality budget, on identical model weights, prompts, and decode
algorithm (greedy or speculative with deterministic seed).

---

## 1. Phase A — Baseline And Quality Floor (weeks 1–4)

No FPGA work yet. Without baselines and quality floor, every later number is
unfalsifiable.

### A1. Real GPU baseline (week 1–2)

Mandatory. Cost report's GPU numbers are spec-derived; reviewers reject.

- [ ] **A1.1** Provision p5.48xlarge (or single-H100 alternative if cheaper).
      Run vLLM with Qwen3-A3B (proxy if Qwen3.6 not yet released — likely
      Qwen2-MoE-A14B/A3B or Mixtral-8x7B as fallback). Capture:
  - tok/s (greedy, batch=1, 1k ctx)
  - ms/token p50 / p95 / p99
  - GPU power via `nvidia-smi dmon` at 10 Hz
  - SM utilization, memory utilization
- [ ] **A1.2** Same on g6.xlarge (L4) and g6e.xlarge (L40S). Same script.
- [ ] **A1.3** Write `benchmarks/measured_gpu_baseline.json` with raw + summary.
      Lock this — never recompute later.
- [ ] **A1.4** Update `cost_model.py::HARDWARES` with measured efficiency
      factors (not 0.55 spec).

**Gate G1**: H100 real tok/s falls within ±15% of cost-model prediction.
If not, root-cause before continuing (vLLM config, KV layout, etc.).

### A2. Quality reference (week 2)

- [ ] **A2.1** Wikitext-2 ppl on Qwen3-A3B BF16 reference (HF transformers).
      Lock value in `benchmarks/quality_reference.json`.
- [ ] **A2.2** MMLU 5-shot accuracy on same model. Same lock.
- [ ] **A2.3** HumanEval pass@1 (code subset). Same lock.
- [ ] **A2.4** GSM8K pass@1 (reasoning robustness check).

These four numbers are the **Quality North Star** every downstream lever is
measured against.

### A3. Software stack validation (week 3–4)

Validate the FPGA-equivalent forward already in `src/fllm/fpga_sim.py`
reproduces a real Qwen3-A3B forward (or proxy) to within numerical tolerance.

- [ ] **A3.1** Extend `scripts/prepare_qwen_weights.py` to load real HF
      weights into FLLM config (tied embeddings, GQA, MoE).
- [ ] **A3.2** Run `scripts/fpga_sim_eval.py` on loaded weights, compare
      against `transformers` forward on same prompts. Tolerance: relative
      L2 error < 1e-3 on logits in BF16; < 5e-2 after INT4+INT8+BFP8.
- [ ] **A3.3** Run quality eval (`scripts/quantize_eval.py`) at each quant
      stage. Produce `experiments/quant_ablation.csv` with ppl per stage.

**Gate G2**: BF16 FLLM reproduces HF forward; INT4+INT8+BFP8 fake-quant ppl
sits within +0.5 of BF16 on Wikitext-2.

---

## 2. Phase B — Lever Validation In Software (weeks 4–10)

Each lever (L1–L6 from `docs/FPGA_OPTIMIZATIONS.md`) must clear a quality
gate **before** RTL work starts.

### B1. Lever L1 — Cache-aware MoE routing (week 4–7) — RESEARCH NOVELTY

This is contribution N1. Highest novelty, highest quality risk.

- [ ] **B1.1** Build `src/fllm/cache_aware_router.py`:
  - Maintain a `resident_mask: BoolTensor[num_experts]` per layer.
  - Modify router loss: `L_route = L_lb + λ · KL(p_router || softmax(logits + α·resident_mask))`
  - Sweep λ ∈ {0, 0.05, 0.1, 0.3, 1.0}; α determines resident bias.
- [ ] **B1.2** Offline routing trace: run validation set on BF16 target,
      record per-layer expert usage histogram. Pick top-k resident.
- [ ] **B1.3** Fine-tune (LoRA on router only? or full router) on a small
      corpus (TinyStories + Wikipedia-en sample, 100M tokens). Measure:
  - Resident-hit rate per layer (target: 80%)
  - Δppl Wikitext-2
  - Δacc MMLU
- [ ] **B1.4** Decision gate **G3**: Δppl ≤ +0.5 at hit rate ≥ 75% for some λ.
      If not → cache-aware lever drops to "60% hit, no FT" mode (less win).

Files: `src/fllm/cache_aware_router.py`, `experiments/cache_aware_sweep.csv`,
`scripts/train_router_finetune.py`.

### B2. Lever L2 — INT3 weight quant (week 6–7, parallel to B1)

- [ ] **B2.1** Add `src/fllm/quant.py::quantize_int3_gptq` with group_size=64.
      Use Hessian-free or simple round-to-nearest as v1; GPTQ as v2 if v1
      drops > +0.5 ppl.
- [ ] **B2.2** Sweep group_size ∈ {32, 64, 128}; measure ppl.
- [ ] **B2.3** Decision gate **G4**: best INT3 setting holds Δppl ≤ +0.4
      vs INT4 baseline. Else drop INT3 globally; keep INT4 globally.

### B3. Lever L2b — INT2/ternary on cold experts only (week 7–8)

- [ ] **B3.1** Identify "cold" experts from B1 trace (bottom 50% usage).
- [ ] **B3.2** Re-quant only those to ternary using `src/fllm/quant.py`
      ternary path (already exists from `1ce9301`).
- [ ] **B3.3** Sweep "cold cutoff": top-10%, 25%, 50% kept INT4, rest ternary.
- [ ] **B3.4** Gate **G5**: cumulative Δppl with B1+B2+B3 ≤ +0.8 ppl.

### B4. Lever L3 — N:M sparsity on cold path (week 8–9)

`src/fllm/sparsity.py` already exists. Extend:

- [ ] **B4.1** Add retraining pass: post-mask LoRA fine-tune to recover ppl.
- [ ] **B4.2** Apply 1:4 only to cold experts (stacked with B3 ternary or
      alternative). Compare 1:4-on-INT4 vs ternary; pick better Pareto.
- [ ] **B4.3** Gate **G6**: cumulative budget after stack ≤ +0.9 ppl.

### B5. Lever L4 — BFP4 KV cache (week 9)

`src/fllm/bfp.py` already has BFP8. Add BFP4.

- [ ] **B5.1** Implement `bfp.py::quantize_bfp4(block=32)`.
- [ ] **B5.2** Sweep block size ∈ {16, 32, 64}.
- [ ] **B5.3** Test at 1k, 4k, 16k context. Long-context degradation expected.
- [ ] **B5.4** Gate **G7**: BFP4 KV adds ≤ +0.15 ppl at 4k context.
      Else fall back to BFP8 (still 2× saving).

### B6. Lever L6 — Speculative decoding cascade (week 8–10) — RESEARCH NOVELTY

Contribution N3. Independent of B1–B5 (lossless).

- [ ] **B6.1** Train tiny draft: 4-layer dense 100M params, distilled from
      target Qwen3.6-A3B outputs. Use `scripts/train_tiny.py` extended.
- [ ] **B6.2** Implement tree speculation (K=4 paths). Extend
      `src/fllm/speculative.py`.
- [ ] **B6.3** Measure acceptance rate on diverse prompts (code, prose,
      reasoning). Stratify by domain.
- [ ] **B6.4** Gate **G8**: tree-speculation effective tok/forward ≥ 2.5
      averaged across domains. Else drop tree, keep single-draft.
- [ ] **B6.5** Stretch: 3-level cascade (50M → 500M → 35B). Measure if 2-level
      is enough.

### B7. Combined Pareto sweep (week 10)

- [ ] **B7.1** Run all valid combinations of B1–B6 on Wikitext-2 + MMLU.
- [ ] **B7.2** Plot Pareto frontier: x = Δppl, y = projected tok/s (from
      `cost_model.py` with measured bytes-per-token).
- [ ] **B7.3** Pick 3 operating points: **conservative**, **moderate**,
      **aggressive**. Freeze each as a named config in
      `src/fllm/presets.py`.

**Phase B exit criteria**: at least one config achieves Δppl ≤ +1.0 ppl
**and** projected tok/s ≥ 3500 on 2× VU47P (cost-model with measured byte
factors). If no config clears both → research thesis is wrong; pivot to
energy-only argument or smaller model.

---

## 3. Phase C — Cycle-Accurate Validation (weeks 8–12, overlapping B)

Before silicon, validate cycle counts and bandwidth at HLS-sim level.

### C1. Extend cycle simulator (week 8–9)

`src/fllm/cycle_sim.py` exists. Add:

- [ ] **C1.1** Cache hit/miss accounting (L1 lever) — bytes_streamed
      becomes a function of resident set + access trace.
- [ ] **C1.2** Sparse skip controller — skip cycles for zero rows.
- [ ] **C1.3** Tree-speculation overhead model (K parallel matvec, shared
      weights → multi-port read of single weight stream).
- [ ] **C1.4** Cross-FPGA dispatch latency model. Initially constant
      `pcie_peer_us`; refined in C4.

### C2. Kernel-level HLS sim (week 9–10)

Each `fpga/*.hpp` kernel:

- [ ] **C2.1** `matvec_int4` — verify II=1, latency ≤ N_in/8 cycles, BRAM
      usage ≤ budget. Compare cycle count vs `cycle_sim.py` prediction.
      Tolerance: ±10%.
- [ ] **C2.2** Add `fpga/matvec_int3.hpp` (new). Same testbench template.
- [ ] **C2.3** Add `fpga/sparse_matvec.hpp` (new) with row-skip controller.
- [ ] **C2.4** Add `fpga/kv_bfp4.hpp` (new) pack/unpack engines.
- [ ] **C2.5** Update `fpga/block_pipeline_dense_tb.cpp` to integrate the
      new kernels behind feature flags.

### C3. Dataflow timing closure dry-run (week 10–11)

Vitis HLS report (csynth only, no place&route yet) on f2 target part:

- [ ] **C3.1** Report II, latency, DSP/LUT/BRAM/URAM per kernel.
- [ ] **C3.2** Identify highest-resource kernel. Reduce if over budget.
- [ ] **C3.3** Target clock: **400 MHz minimum, 600 MHz stretch**. Document
      achieved Fmax per kernel in `fpga/resource_report.md`.

### C4. Cross-FPGA dispatch protocol design (week 11–12)

This is the single largest implementation risk for 2-FPGA shard.

- [ ] **C4.1** Design AXI-Stream protocol over QSFP/PCIe peer for
      router → expert dispatch. Spec the per-token payload:
  - 8 × (expert_id, gate_weight, hidden_chunk) → ~ 4 KB / dispatch.
- [ ] **C4.2** Model async double-buffer dispatch overlapping with previous
      layer's matvec. Goal: hide dispatch latency entirely.
- [ ] **C4.3** Gate **G9**: simulated dispatch overhead ≤ 50 µs/token at
      48 layers. Else 2-FPGA shard is unworkable → fall back to single-FPGA
      Path B (DDR-staged experts, worse numbers but valid result).

---

## 4. Phase D — Silicon On AWS F2 (weeks 12–20)

### D1. F2 sandbox (week 12–14)

- [ ] **D1.1** Provision f2.6xlarge developer instance.
- [ ] **D1.2** Build `matvec_int4` to bitstream. Generate AFI.
      First AFI build ~24h; budget accordingly.
- [ ] **D1.3** Load weights subset to HBM. Measure realized BW vs predicted.
- [ ] **D1.4** Gate **G10**: measured BW efficiency ≥ 60% (out of 460 GB/s).
      Else investigate HBM controller config, channel layout.

### D2. Block-pipeline single layer on FPGA (week 14–16)

- [ ] **D2.1** Build composite `block_pipeline_dense` to bitstream.
      Validate one full transformer block end-to-end vs Python ref.
- [ ] **D2.2** Numerical match: bit-exact vs `fpga_sim.py` fixed-point ref.
- [ ] **D2.3** Cycle measurement at real silicon. Update `cycle_sim.py`
      with corrected constants.

### D3. Token loop FSM on FPGA (week 16–17)

- [ ] **D3.1** Implement `token_loop_ctrl` (referenced in mapping doc but
      not yet in `fpga/`).
- [ ] **D3.2** Argmax sampler runs entirely on FPGA. Greedy decode only.
- [ ] **D3.3** Generate text end-to-end on FPGA. Compare vs CPU greedy.
- [ ] **D3.4** Gate **G11**: bit-exact output vs Python fixed-point ref for
      first 64 tokens on 10 prompts. Allow speculative-decode rejection
      randomness only if seed-controlled.

### D4. Two-FPGA shard (week 17–19)

- [ ] **D4.1** Provision f2.12xlarge. Bring up cross-FPGA dispatch protocol
      designed in C4.
- [ ] **D4.2** Shard experts: even-indexed to FPGA0, odd to FPGA1. Shared
      weights replicated.
- [ ] **D4.3** Run end-to-end Qwen3.6-A3B token loop on 2 FPGAs.
- [ ] **D4.4** Measure realized tok/s, ms/tok p50/p99, power.
- [ ] **D4.5** Gate **G12**: measured tok/s ≥ 50% of `cost_model` prediction
      for chosen preset. Else iterate on bottleneck (dispatch / layout /
      pipeline depth).

### D5. Speculative decoding on silicon (week 19–20) — stretch

- [ ] **D5.1** Place 100M draft model on FPGA0 BRAM.
- [ ] **D5.2** Run draft + target verification in tree mode.
- [ ] **D5.3** Measure effective tok/s vs single-stream silicon.

---

## 5. Phase E — Measurement And Paper (weeks 20–26)

### E1. Final benchmarks (week 20–22)

Per `docs/BENCHMARK_PLAN.md`. Same model weights, prompts, decode policy on:

- f2.12xlarge (FPGA target)
- g6.xlarge, g6e.xlarge (cost-matched GPU)
- p5.48xlarge / 1×H100 (performance reference)

Capture: tok/s, ms/tok p50/p95/p99, J/tok (wallplug + chip), $/1M tok,
Wikitext-2 ppl, MMLU acc, HumanEval pass@1, GSM8K.

Report each preset (conservative / moderate / aggressive) separately.

### E2. Ablations (week 22–23)

For the paper. Each lever isolated and stacked:

- [ ] tok/s and Δppl with L1 only, L1+L2, L1+L2+L3, ..., full stack.
- [ ] BFP8 vs BFP4 KV at 1k, 4k, 16k context.
- [ ] Cache hit rate vs Δppl Pareto.
- [ ] Speculation tree depth K ∈ {1, 2, 4} vs acceptance.

### E3. Writeup (week 23–26)

- [ ] Extended abstract (Phase 0 todo).
- [ ] Full paper draft. Target: MLSys / ASPLOS / ISCA.
- [ ] Submit `paper/` artifacts: code, weights, eval scripts, AFI hashes.
- [ ] Reproducibility checklist.

---

## 6. Concrete Projection Targets (with math)

From the `B_token` derivation in chat analysis:

```
Preset           B_token   2×VU47P real tok/s   tok/s after spec K=4 (×3.2)
─────────────────────────────────────────────────────────────────────────
naive port       1.700 GB         381                    n/a
conservative     0.700 GB         986                   3155
moderate         0.350 GB        1971                   6308
aggressive       0.090 GB        7667                  24536
─────────────────────────────────────────────────────────────────────────
H100 real:       ≈ 1080 tok/s (cost report; verify in A1)
```

Apply **0.4× reality discount** to FPGA (dispatch overhead, p99 stalls,
HBM eff slip, sync) → realistic targets:

| Preset       | Realistic tok/s | vs H100 | J/tok | $/1M tok | Δppl |
|--------------|----------------:|--------:|------:|---------:|-----:|
| conservative |          ~1260  | 1.17×   | 0.36  | $1.16    | +0.3 |
| moderate     |          ~2500  | 2.31×   | 0.18  | $0.59    | +0.7 |
| aggressive   |          ~9800  | 9.07×   | 0.046 | $0.37    | +1.2 |

Paper claim sits at **moderate** preset. Aggressive is a stretch chapter.

---

## 7. Quality Budget Tracking

Single source of truth: `experiments/quality_budget.csv` updated after every
gate.

| Lever                       | Budget | Measured | Status |
|-----------------------------|-------:|---------:|--------|
| INT4 W + INT8 A             | +0.20  | TBD      | pending |
| BFP8 KV @ 4k                | +0.10  | TBD      | pending |
| BFP4 KV @ 4k                | +0.20  | TBD      | pending |
| INT3 global                 | +0.30  | TBD      | pending |
| Ternary cold experts        | +0.20  | TBD      | pending |
| 1:4 sparse cold (retrained) | +0.20  | TBD      | pending |
| Cache-aware routing (λ tuned)| +0.30 | TBD      | pending |
| Spec decode (lossless)      |  0.00  | TBD      | pending |

Each row blocked by its gate (G2–G8). If running budget exceeds +1.0 ppl,
the next lever is **dropped**, not added.

---

## 8. Decision Gates (full register)

| ID  | Phase | Test                                                  | Pass         | Fail action                       |
|-----|-------|-------------------------------------------------------|--------------|-----------------------------------|
| G1  | A1    | H100 real tok/s within ±15% of cost-model spec        | continue     | fix vLLM config / KV; rerun       |
| G2  | A3    | BF16 FLLM == HF forward; quant stack ≤ +0.5 ppl       | continue     | debug quant; tighten BFP block    |
| G3  | B1    | Cache-aware FT: hit≥75% with Δppl ≤ +0.5              | continue full L1 | downgrade to no-FT 60% hit    |
| G4  | B2    | INT3 best setting Δppl ≤ +0.4 vs INT4                  | adopt INT3   | stay at INT4                      |
| G5  | B3    | Cumulative L1+L2+L3 Δppl ≤ +0.8                        | continue     | drop ternary, restore INT4 cold   |
| G6  | B4    | After 1:4 sparse + retrain, cumulative ≤ +0.9          | continue     | drop sparsity                     |
| G7  | B5    | BFP4 KV adds ≤ +0.15 @ 4k                              | adopt BFP4   | stay BFP8                         |
| G8  | B6    | Tree speculation accept ≥ 2.5 tok/forward              | adopt tree   | single-draft fallback             |
| G9  | C4    | Simulated dispatch ≤ 50 µs/tok at 48 layers            | continue 2-FPGA | fall to single-FPGA Path B     |
| G10 | D1    | F2 HBM efficiency ≥ 60% measured                       | continue     | redesign channel layout           |
| G11 | D3    | Token-loop FSM bit-exact for 64 tok × 10 prompts       | continue     | debug fixed-point; widen accum    |
| G12 | D4    | 2-FPGA realized tok/s ≥ 50% of model prediction        | claim valid  | iterate; if persistent → publish negative result honestly |

A failed gate is **information**, not failure. Document each in
`experiments/gate_log.md`.

---

## 9. Risk Register (top 8, ranked by impact)

| # | Risk                                            | Impact            | Mitigation                                                 |
|---|-------------------------------------------------|-------------------|------------------------------------------------------------|
| 1 | Cache-aware routing tanks quality (Δppl > 1)    | Aggressive preset dies | Pareto curve at multiple λ; fall back to no-FT cache  |
| 2 | Spec decode acceptance < 50% on MoE             | Lose 3× spec multiplier | Train domain-specific draft; tree-depth sweep         |
| 3 | Cross-FPGA dispatch > 100 µs/tok                | Kills 2-FPGA shard      | Async double-buffer; if fails → single-FPGA Path B    |
| 4 | Timing closure at < 400 MHz                     | BW efficiency drops 1.5–2× | Manual placement; pipeline-deepen worst paths      |
| 5 | Qwen3.6-A3B not released on time                | No real target weights  | Use Qwen2-MoE-A14B as proxy; rename in paper           |
| 6 | HBM channel-aware layout under-delivers         | Lose 1.5× BW factor     | Trace-aware allocation; measure on real F2             |
| 7 | AFI build queue stalls (AWS-side)               | Schedule slip 1–2 weeks | Budget 2× iteration; pre-stage bitstream variants      |
| 8 | f2 hourly pricing change                        | Breaks $/1M tok claim   | Lock pricing snapshot; report sensitivity ± 30%        |

---

## 10. File / Code Touch List

New files to create:

- `src/fllm/cache_aware_router.py` — N1 contribution.
- `src/fllm/draft_cascade.py` — N3 contribution.
- `src/fllm/bfp.py::quantize_bfp4` — add to existing.
- `src/fllm/quant.py::quantize_int3_gptq` — add to existing.
- `fpga/matvec_int3.hpp` + testbench.
- `fpga/sparse_matvec.hpp` + testbench.
- `fpga/kv_bfp4.hpp` + testbench.
- `fpga/token_loop_ctrl.hpp` + testbench.
- `fpga/cross_fpga_dispatch.hpp` (2-FPGA only).
- `scripts/train_router_finetune.py`.
- `scripts/train_draft_cascade.py`.
- `scripts/measure_h100_baseline.py` (vLLM wrapper).
- `scripts/measure_f2_realized.py` (silicon measurement).
- `benchmarks/measured_gpu_baseline.json`.
- `benchmarks/quality_reference.json`.
- `experiments/quant_ablation.csv`.
- `experiments/cache_aware_sweep.csv`.
- `experiments/quality_budget.csv`.
- `experiments/gate_log.md`.
- `fpga/resource_report.md`.

Existing files to extend:

- `src/fllm/cost_model.py` — add cross-FPGA dispatch term, tree-spec term.
- `src/fllm/cycle_sim.py` — see §3 C1.
- `src/fllm/sparsity.py` — add retrain hook.
- `src/fllm/speculative.py` — tree variant.
- `src/fllm/presets.py` — add three named presets from §6.
- `fpga/block_pipeline_dense.hpp` — integrate new kernels behind flags.

---

## 11. Stretch Goals (post-paper)

- 4k → 16k context with sliding-window + attention sinks.
- f2.48xlarge (8× VU47P) aggregate scaling, tensor + pipeline parallel.
- 3-level cascade speculation (50M → 500M → 35B).
- Log-domain matvec ablation chapter.
- Joint compile-time HBM channel allocation from offline trace.
- Open-source AFI for reproducibility.

---

## 12. Weekly Cadence

- **Mon**: gate review of previous week. Update `gate_log.md`.
- **Wed**: pair review of new code / RTL.
- **Fri**: re-run `scripts/cost_report.py` + `scripts/quantize_eval.py` with
  current numbers. Commit to `experiments/weekly_status_YYYY-WW.md`.

Schedule slips: documented in `gate_log.md` with reason and replan. No
silent slips. Six months of weekly cadence = 26 status snapshots → strong
artifact for paper appendix.

---

## 13. TL;DR

- 6 phases, 26 weeks, 12 decision gates.
- 3 PhD-grade novel contributions: cache-aware MoE routing, joint
  quant×sparse×locality, on-chip cascade speculation.
- Win path: ridurre `B_token` di ~10× sotto floor GPU; 2× VU47P beats H100
  by 2-9× tok/s and 5-14× J/tok at moderate-to-aggressive presets.
- Quality budget +1.0 ppl is binding constraint. Pareto sweep mandatory.
- Real measurement at every gate, no spec-only claims.
- Negative result still publishable: documented gates, falsifiable claims.
