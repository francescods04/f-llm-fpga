# Gate Log

Single source of truth for all decision gates (G1–G12).

| Week | Gate | Phase | Test | Result | Action Taken |
|------|------|-------|------|--------|--------------|
| TBD  | G1   | A1    | H100 real tok/s within ±15% of cost-model spec | TBD | TBD |
| TBD  | G2   | A3    | BF16 FLLM == HF forward; quant stack ≤ +0.5 ppl | TBD | TBD |
| TBD  | G3   | B1    | Cache-aware FT: hit≥75% with Δppl ≤ +0.5 | TBD | TBD |
| TBD  | G4   | B2    | INT3 best setting Δppl ≤ +0.4 vs INT4 | TBD | TBD |
| TBD  | G5   | B3    | Cumulative L1+L2+L3 Δppl ≤ +0.8 | TBD | TBD |
| TBD  | G6   | B4    | After 1:4 sparse + retrain, cumulative ≤ +0.9 | TBD | TBD |
| TBD  | G7   | B5    | BFP4 KV adds ≤ +0.15 @ 4k | TBD | TBD |
| TBD  | G8   | B6    | Tree speculation accept ≥ 2.5 tok/forward | TBD | TBD |
| TBD  | G9   | C4    | Simulated dispatch ≤ 50 µs/tok at 48 layers | TBD | TBD |
| TBD  | G10  | D1    | F2 HBM efficiency ≥ 60% measured | TBD | TBD |
| TBD  | G11  | D3    | Token-loop FSM bit-exact for 64 tok × 10 prompts | TBD | TBD |
| TBD  | G12  | D4    | 2-FPGA realized tok/s ≥ 50% of model prediction | TBD | TBD |
