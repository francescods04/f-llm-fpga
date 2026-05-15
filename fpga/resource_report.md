# FPGA Resource Report

Placeholder for Vitis HLS csynth results. Updated after each kernel synthesis run.

## Kernel Inventory

| Kernel | Target II | Latency (cycles) | DSP | LUT | BRAM | URAM | Fmax (MHz) | Status |
|--------|-----------|------------------|-----|-----|------|------|------------|--------|
| matvec_int4_tile_dense | 1 | ≤ N_in/8 | TBD | TBD | TBD | TBD | TBD | C++ testbench pass |
| matvec_int2_tile_dense | 1 | ≤ N_in/16 | TBD | TBD | TBD | TBD | TBD | C++ testbench pass |
| matvec_int3_tile_dense | 1 | ≤ N_in/10 | TBD | TBD | TBD | TBD | TBD | C++ testbench pass |
| sparse_matvec | 1 | variable | TBD | TBD | TBD | TBD | TBD | C++ testbench pass |
| kv_bfp4_pack | 1 | TBD | TBD | TBD | TBD | TBD | TBD | C++ testbench pass |
| kv_bfp4_unpack | 1 | TBD | TBD | TBD | TBD | TBD | TBD | C++ testbench pass |
| rmsnorm_engine | 1 | TBD | TBD | TBD | TBD | TBD | TBD | C++ testbench pass |
| silu_lut | 1 | TBD | TBD | TBD | TBD | TBD | TBD | C++ testbench pass |
| softmax_engine | 1 | TBD | TBD | TBD | TBD | TBD | TBD | C++ testbench pass |
| rope_engine | 1 | TBD | TBD | TBD | TBD | TBD | TBD | C++ testbench pass |
| sampler_engine | 1 | TBD | TBD | TBD | TBD | TBD | TBD | C++ testbench pass |
| token_loop_ctrl | TBD | TBD | TBD | TBD | TBD | TBD | TBD | C++ structural test pass |

## Composite Build Status

| Build | Device | Clock Target | Achieved Fmax | Utilization | Bitstream | AFI |
|-------|--------|--------------|---------------|-------------|-----------|-----|
| block_pipeline_dense | VU47P | 400 MHz | TBD | TBD | TBD | TBD |

## Notes

- Target clock: **400 MHz minimum, 600 MHz stretch**.
- HBM efficiency target: ≥ 60% measured (Gate G10).
- If any kernel exceeds BRAM/URAM budget, reduce tile_rows or lanes.
