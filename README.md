# F-LLM FPGA

Full-FPGA inference research project for FPGA-native language models.

## Thesis

Autoregressive LLM decode at batch size 1 is often constrained by memory movement,
small sequential kernels, and low utilization on general-purpose GPUs. This project
investigates whether a model co-designed for FPGA execution can beat a GPU baseline
in energy efficiency for inference-only text generation.

The target is not to port an existing dense Transformer unchanged. The target is to
design a small but functional FPGA-native LLM with:

- low-bit weights and activations;
- compressed local/global attention;
- optional low-bit MoE experts;
- hardware-owned token loop;
- on-FPGA sampling;
- reproducible GPU and FPGA efficiency benchmarks.

## Primary Metric

The main metric is:

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

## Initial Target

The first publishable target is a functional model in the 50M-350M parameter range.
The stretch target is a 1B parameter class model, assuming the FPGA memory and
tooling constraints prove workable.

## Repository Layout

```text
paper/        Paper draft, outline, figures, references
docs/         Architecture, benchmark plan, research plan
src/fllm/     Python reference implementation
fpga/         HLS/RTL implementation notes and kernels
benchmarks/   Reproducible benchmark harnesses
experiments/  Training, quantization, and ablation notes
scripts/      Utility scripts
```

## Current Status

Software reference phase. The repository includes a tiny PyTorch decoder model
(with optional top-k MoE), byte/BPE tokenizer, local corpus, training script,
decode benchmark, INT4/INT8 fake-quant, packed INT4 weight exporter, and a
roofline FPGA-vs-GPU cost model. The next milestone is compressed global
context and HLS kernels.

End target: run [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
on AWS F2 (2x VU47P) and beat g6/g6e on `$/1M tokens` and `joule/token`. See
[docs/QWEN3_A3B_FPGA_MAPPING.md](docs/QWEN3_A3B_FPGA_MAPPING.md).

Print the current FPGA-vs-GPU projection:

```bash
PYTHONPATH=src python3 scripts/cost_report.py
```

Inspect the Qwen3-A3B target shape and instantiate the matching FLLM config:

```bash
PYTHONPATH=src python3 scripts/inspect_hf_config.py \
  --emit-template datasets/qwen3-a3b/config.json
PYTHONPATH=src python3 scripts/inspect_hf_config.py \
  --config datasets/qwen3-a3b/config.json
```

Per-kernel cycle budget at chosen FPGA tile parallelism:

```bash
PYTHONPATH=src python3 scripts/cycle_report.py --num-tiles 16 --tile-rows 32
```

End-to-end FPGA-equivalent forward (INT4 weights + INT8 acts + BFP8 KV +
LUT softmax/SiLU/rsqrt) on a trained checkpoint:

```bash
PYTHONPATH=src python3 scripts/fpga_sim_eval.py \
  --checkpoint checkpoints/bpe-smoke/model.pt
```

## Quickstart

Run a model smoke test:

```bash
PYTHONPATH=src python3 scripts/smoke_model.py
```

Train the tiny reference model on the local sample corpus:

```bash
PYTHONPATH=src python3 scripts/train_tiny.py --steps 50
```

Benchmark decode:

```bash
PYTHONPATH=src python3 benchmarks/benchmark_decode.py \
  --checkpoint checkpoints/tiny/model.pt \
  --new-tokens 64
```

For the first minimally useful model, use the BPE/TinyStories path in
[docs/TRAINING.md](docs/TRAINING.md).

Generate from a checkpoint:

```bash
PYTHONPATH=src python3 scripts/generate.py \
  --checkpoint checkpoints/tiny/model.pt \
  --prompt "Once upon a time" \
  --new-tokens 120
```
