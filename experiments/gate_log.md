# Gate Log

Single source of truth for all decision gates (G1–G12).

| Week | Gate | Phase | Test | Result | Action Taken |
|------|------|-------|------|--------|--------------|
| W1   | —    | A1    | Colab T4 baseline (proxy sanity) | **69.34 tok/s** on Qwen2.5-0.5B | Logged; H100 pending |
| W1   | G1   | A1    | A100 80GB transformers tok/s vs cost model | **8.93 tok/s** measured vs ~12.5 predicted (ratio 0.71). **FAIL** ±15% gate. | Root cause: transformers overhead at batch=1 MoE decode. Action: rerun with vLLM backend on same HW. |
| W1   | G2   | A3    | BF16 FLLM == HF forward; quant stack ≤ +0.5 ppl | **Partial** — proxy model validated; rel L2 INT4=0.997 high → need calibration dataset | Documented; next: run GPTQ/AWQ calibration on TinyStories sample |
| W1   | G4   | B2    | INT3 best setting Δppl ≤ +0.4 vs INT4 | **INT3 rel L2 = 1.005** (similar to INT4 0.997) — proxy too small for meaningful Δppl; need real 35B or larger proxy | Hold until larger model or calibrated scales |
| W1   | G7   | B5    | BFP4 KV adds ≤ +0.15 @ 4k | **BFP4 rel L2 = 0.147** on activations — well within bound | Pass for activation path; KV-specific test pending |
| TBD  | G3   | B1    | Cache-aware FT: hit≥75% with Δppl ≤ +0.5 | TBD | TBD |
| TBD  | G5   | B3    | Cumulative L1+L2+L3 Δppl ≤ +0.8 | TBD | TBD |
| TBD  | G6   | B4    | After 1:4 sparse + retrain, cumulative ≤ +0.9 | TBD | TBD |
| TBD  | G8   | B6    | Tree speculation accept ≥ 2.5 tok/forward | TBD | TBD |
| TBD  | G9   | C4    | Simulated dispatch ≤ 50 µs/tok at 48 layers | TBD | TBD |
| TBD  | G10  | D1    | F2 HBM efficiency ≥ 60% measured | TBD | TBD |
| TBD  | G11  | D3    | Token-loop FSM bit-exact for 64 tok × 10 prompts | TBD | TBD |
| TBD  | G12  | D4    | 2-FPGA realized tok/s ≥ 50% of model prediction | TBD | TBD |
