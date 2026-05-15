# Benchmark Plan

## Objective

Measure whether a full-FPGA inference implementation beats a GPU baseline in
energy efficiency for batch-1 autoregressive decode.

## Primary Metric

```text
joule/token
```

If direct power measurement is unavailable in early stages, report:

```text
tokens/s/W estimated
active power proxy
tokens/s
ms/token
```

Power methodology must be stated explicitly.

## Workloads

Start with deterministic greedy decoding.

Prompt sets:

- short prompts: 32-128 tokens;
- medium prompts: 512-2048 tokens;
- long prompts: 4096+ tokens once compressed attention works.

Generation length:

```text
128 new tokens
256 new tokens
512 new tokens
```

## Baselines

Minimum:

- CPU baseline;
- Mac M1 Metal/MLX baseline;
- NVIDIA GPU baseline;
- FPGA implementation.

Ideal:

- same model on all systems;
- same tokenizer;
- same quantization where possible;
- same decoding.

## Reported Numbers

For each run:

```text
model size
parameter count
precision
context length
generated tokens
tokens/s
ms/token
power
joule/token
quality metric
```

For FPGA:

```text
clock frequency
LUT utilization
FF utilization
BRAM utilization
URAM utilization
DSP utilization
HBM bandwidth
```

## Fairness Rules

- Do not compare FPGA INT4 to GPU FP16 as the main result unless clearly labeled.
- Main comparison should use the same quantized model where possible.
- Separate prefill and decode.
- Report host overhead separately.
- Report failed or unfavorable configurations.

