# F-LLM FPGA

Full-FPGA inference research project for FPGA-native language models.

## Thesis

Autoregressive LLM decode at batch size 1 is constrained by memory movement,
low utilization on general-purpose GPUs, and the overhead of launching thousands
of small kernels per token. This project investigates whether a model co-designed
for FPGA execution can beat a GPU baseline in energy efficiency (`joule/token`)
for inference-only text generation.

The approach is not to port an existing dense Transformer unchanged, but to
co-design hardware and software:

- **Low-bit weights**: INT4 for MoE experts, INT2 (ternary) for dense paths
- **Structured sparsity**: 2:4 pruning with skip-mask in the matvec engine
- **Compressed attention**: GQA + RoPE + local window, KV cache in BFP8
- **On-chip residency**: vocab cache (top-K tokens in URAM), tiny draft model
  for speculative decode
- **Spatial dataflow**: one Transformer block = one DATAFLOW kernel with
  streaming FIFOs between stages
- **Hardware-owned token loop**: the FPGA autonomously runs decode without
  host intervention

## Run on Google Colab (Free GPU)

You can validate the quant ablation and measure GPU baseline tok/s on a **free T4** without installing anything locally.

**Option A — One-click notebook:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/francescods04/f-llm-fpga/blob/main/notebooks/colab_baseline.ipynb)

**Option B — One-cell script (copy-paste):**
Open Colab, create a new notebook, paste the contents of `scripts/colab_quickstart.py` into a single code cell, and run it. It installs deps, clones the repo, loads a proxy model, runs INT3/INT4/BFP4 tests, and measures tok/s automatically. Results are saved to the Files panel for download.

## Run Real Target Model (Qwen3.6-35B-A3B) on Colab Pro

If you have **Colab Pro/Pro+ with an A100 (40 GB or 80 GB)** and a HuggingFace token with access to the gated model, you can benchmark the actual target model:

**Prerequisite:**
1. Get a HuggingFace token: https://huggingface.co/settings/tokens (scope: read)
2. In Colab, click 🔑 **Secrets** (left panel)
3. Add secret: Name = `HF_TOKEN`, Value = your token
4. Toggle **Notebook access** ON
5. Runtime → **Restart session**

**Option A — One-click notebook:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/francescods04/f-llm-fpga/blob/main/notebooks/colab_35b_baseline.ipynb)

**Option B — One-cell script:**
Paste the contents of `scripts/colab_35b_baseline.py` into a single code cell. It reads `HF_TOKEN` from Colab secrets, auto-detects VRAM, picks FP16/8-bit/4-bit quantization, downloads the 35 B checkpoint, runs greedy decode, and exports a JSON with tok/s and memory usage. This is the measurement used for **Gate G1**.

*If you do not have HF access to the gated model, switch to a public proxy (e.g. `Qwen/Qwen2.5-14B`) by editing the `MODEL_NAME` variable at the top of the script.*

## Primary Metric

```text
joule/token
```

Secondary metrics:

```text
tokens/s/W
ms/token
tokens/s
model perplexity
FPGA resource utilization
```

## Target Model

[Qwen3-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) on AWS F2
(2× AMD VU47P).  See [docs/QWEN3_A3B_FPGA_MAPPING.md](docs/QWEN3_A3B_FPGA_MAPPING.md).

Hardware specs:
- 2× VU47P, 460 GB/s HBM per device
- ~40 MB URAM per SLR, 9 MB BRAM
- 600 MHz target fmax

## Repository Layout

```text
fpga/           HLS kernels (host-compilable with g++ via hls_stubs.hpp)
  matvec_int4.hpp      INT4×INT8 matvec tile (dense + 2:4 sparse)
  matvec_int2.hpp      INT2×INT8 ternary matvec tile
  rmsnorm_engine.hpp   Streaming RMSNorm with rsqrt LUT
  silu_lut.hpp         Piecewise-linear SiLU via BRAM LUT
  softmax_engine.hpp   Shifted softmax with exp LUT
  rope_engine.hpp      Rotary Position Embedding (decode)
  sampler_engine.hpp   Greedy + categorical PRNG sampler
  block_pipeline_dense.hpp  Full 10-stage dense block DATAFLOW
  *_tb.cpp             Host testbenches (no Vitis HLS required)
src/fllm/       Python reference implementation
  model.py             Transformer + generate loop
  quant.py             Fake-quant INT4 / INT2 / per-channel scales
  sparsity.py          N:M structured pruning
  vocab_cache.py       Two-path LM head (URAM cache + HBM fallback)
  speculative.py       Draft model + speculative decode
  export.py            Packed INT4 binary exporter (FLLM v1 format)
scripts/        Research & utility scripts
  token_loop_sim.py    Cycle-accurate decode simulator
  cost_report.py       FPGA-vs-GPU roofline comparison
  prepare_qwen_weights.py   HF download → INT4 → FLLM export
  prepare_dummy_weights.py  Random Qwen-shaped → FLLM export
  cycle_report.py      Per-kernel cycle budget
tests/          Smoke & integration tests
```

## FPGA Kernel Status

All kernels compile with standard `g++` (no Xilinx tools installed) using
`fpga/hls_stubs.hpp`, which provides `ap_uint<N>`, `ap_int<N>`, and `hls::stream`
up to 65536 bits.

| Kernel | Testbench | Max Error vs Python |
|--------|-----------|---------------------|
| `matvec_int4` | `matvec_int4_kernel_tb.cpp` | **0** (dense + sparse) |
| `matvec_int2` | `matvec_int2_tb.cpp` | **0** (ternary) |
| `rmsnorm_engine` | `rmsnorm_engine_tb.cpp` | 1e-6 |
| `silu_lut` | `silu_lut_tb.cpp` | 1.5e-5 |
| `softmax_engine` | `softmax_engine_tb.cpp` | < 1e-2 |
| `rope_engine` | `rope_engine_tb.cpp` | **0** |
| `sampler_engine` | `sampler_engine_tb.cpp` | exact (deterministic) |
| `block_pipeline_dense` | `block_pipeline_dense_tb.cpp` | structural (zero-weight sanity) |

## Quickstart

### Run all host kernel tests

```bash
g++ -std=c++17 -I. -I./fpga -DFLLM_LANES=32 -DFLLM_TILE_ROWS=16 \
    -o /tmp/matvec_int4_tb fpga/matvec_int4_kernel_tb.cpp && /tmp/matvec_int4_tb
g++ -std=c++17 -I. -I./fpga -DFLLM_LANES=32 -DFLLM_TILE_ROWS=16 \
    -o /tmp/matvec_int2_tb fpga/matvec_int2_tb.cpp && /tmp/matvec_int2_tb
g++ -std=c++17 -I. -I./fpga -o /tmp/rmsnorm_tb fpga/rmsnorm_engine_tb.cpp && /tmp/rmsnorm_tb
g++ -std=c++17 -I. -I./fpga -o /tmp/silu_tb fpga/silu_lut_tb.cpp && /tmp/silu_tb
g++ -std=c++17 -I. -I./fpga -o /tmp/softmax_tb fpga/softmax_engine_tb.cpp && /tmp/softmax_tb
g++ -std=c++17 -I. -I./fpga -o /tmp/rope_tb fpga/rope_engine_tb.cpp && /tmp/rope_tb
g++ -std=c++17 -I. -I./fpga -o /tmp/sampler_tb fpga/sampler_engine_tb.cpp && /tmp/sampler_tb
g++ -std=c++17 -I. -I./fpga -DFLLM_LANES=32 -DFLLM_TILE_ROWS=16 \
    -o fpga/block_pipeline_dense_tb fpga/block_pipeline_dense_tb.cpp && ./fpga/block_pipeline_dense_tb
```

### Run Python smoke tests

```bash
PYTHONPATH=src python3 tests/test_qwen_fpga_sim.py
PYTHONPATH=src python3 tests/test_ternary_linear.py
PYTHONPATH=src python3 tests/test_speculative.py
PYTHONPATH=src python3 tests/test_end_to_end.py
```

### FPGA-vs-GPU cost projection

```bash
PYTHONPATH=src python3 scripts/cost_report.py
```

### Token loop simulator (with speculative decode)

```bash
PYTHONPATH=src python3 scripts/token_loop_sim.py \
    --decode-steps 128 --spec-draft 4 --spec-accept 0.70
```

### Export dummy Qwen-shaped weights to FLLM v1

```bash
PYTHONPATH=src python3 scripts/prepare_dummy_weights.py --out-dir checkpoints/dummy-fpga
# → 121 MB, manifest.json, round-trip MSE < 0.02
```

### Export real HuggingFace weights (requires `transformers`, ~22 GB disk for 35B INT4)

```bash
PYTHONPATH=src python3 scripts/prepare_qwen_weights.py \
    --model-id Qwen/Qwen3.6-35B-A3B \
    --out-dir checkpoints/qwen3-fpga \
    --nm-n 2 --nm-m 4
```

For smaller variants (0.5B–7B) to test the pipeline on a laptop:

```bash
PYTHONPATH=src python3 scripts/prepare_qwen_weights.py \
    --model-id Qwen/Qwen2.5-0.5B-Instruct \
    --out-dir checkpoints/qwen05b-fpga
```

## Current Status

- ✅ **Python reference**: Transformer with GQA, RoPE, MoE, RMSNorm, SwiGLU,
  INT4/INT2 fake-quant, 2:4 sparsity, vocab cache, speculative decode.
- ✅ **FPGA kernels**: matvec (INT4 + INT2), RMSNorm, SiLU, softmax, RoPE,
  sampler, and a full 10-stage dense block pipeline.
- ✅ **Cost model**: HBM-bound decode identified; speculative decode + INT2
  dense-path URAM residency are the primary speed-up levers.
- ✅ **Export format**: FLLM v1 binary with packed INT4 weights, per-channel
  scales, deterministic layout, manifest JSON.
- 🔄 **Next**: C++ loader for FLLM binaries → FPGA kernel integration test,
  then SystemC token-loop controller simulation.

## End Goal

Run Qwen3-35B-A3B decode on AWS F2 at >300 tok/s with lower `joule/token`
than a g6/g6e GPU instance.
