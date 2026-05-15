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

Software reference phase. The repository includes a tiny PyTorch decoder model,
byte tokenizer, local corpus, training script, and decode benchmark. The next
milestone is to add compressed global context and quantization simulation.

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
