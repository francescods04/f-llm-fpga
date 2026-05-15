# F-LLM: FPGA-Native Low-Bit Language Models for Energy-Efficient Autoregressive Inference

## Abstract Draft

Autoregressive language model inference at batch size one is dominated by sequential
token generation, repeated weight movement, and low arithmetic intensity. GPUs are
highly effective for dense batched computation, but can be inefficient for small,
latency-sensitive decode workloads. We investigate a full-FPGA inference stack for a
language model co-designed with the hardware. Instead of porting an existing dense
Transformer unchanged, we design an FPGA-native decoder architecture using low-bit
matrix-vector execution, compressed attention, optional low-bit MoE experts, and a
hardware-owned token loop. We evaluate whether this design can improve joule/token
relative to a GPU baseline under identical model, quantization, prompt, and decoding
conditions.

## 1. Introduction

Key claims to make precise:

- Batch-1 decode is structurally different from batched prefill.
- GPU peak FLOPs do not directly translate into decode efficiency.
- FPGA can expose custom low-bit datapaths and deterministic token loops.
- Model architecture must be co-designed with FPGA constraints.

Research question:

```text
Can a full-FPGA, FPGA-native LLM beat GPU energy efficiency for batch-1 decode
without relying on a GPU verifier?
```

## 2. Background And Motivation

Discuss:

- autoregressive decode bottlenecks;
- GPU memory movement and utilization at batch size 1;
- FPGA strengths and weaknesses;
- TALOS-V2 as a readable full-hardware Transformer demonstration;
- FlightLLM, LUT-LLM, and TeLLMe as related FPGA LLM inference systems;
- Qwen3.6 and DeepSeek-V4 as architecture inspiration, not direct first targets.

## 3. Model Architecture

Working name:

```text
F-LLM
```

Candidate block:

```text
RMSNorm
Local attention over recent tokens
Compressed global context attention
Residual add
RMSNorm
Low-bit MLP or low-bit MoE
Residual add
```

Design constraints:

- fixed tensor shapes;
- decode-first optimization;
- limited vocabulary for the first model;
- low-bit weight storage;
- simple activation formats;
- hardware-friendly approximations.

## 4. Hardware Architecture

Core engines:

- packed INT4/INT8 matrix-vector engine;
- compressed-context update engine;
- local attention engine;
- low-bit expert engine;
- hierarchical or streaming LM head;
- token loop controller;
- host interface.

Main hardware hypothesis:

```text
If the token loop and low-bit matvecs remain on FPGA, the design can reduce
joule/token by avoiding GPU-style general-purpose overheads and using native
sub-byte compute.
```

## 5. Training And Quantization

Training path:

1. Train small floating-point model.
2. Add architectural constraints.
3. Quantize weights and activations.
4. Optionally fine-tune with quantization-aware training.
5. Export packed weights.

Quantization targets:

- INT8 activations;
- INT4 weights;
- INT2 or ternary ablations;
- FP4-style expert weights if useful.

## 6. Evaluation Methodology

Primary metric:

```text
joule/token
```

Secondary metrics:

- tokens/s/W;
- tokens/s;
- p50/p99 ms/token;
- prefill latency;
- decode latency;
- perplexity;
- FPGA resource utilization;
- HBM bandwidth utilization.

Baselines:

- CPU baseline;
- Mac M1 Metal/MLX baseline;
- NVIDIA GPU baseline;
- FPGA implementation.

Rules:

- same model;
- same tokenizer;
- same prompt set;
- same quantization where supported;
- same decoding policy.

## 7. Results

Planned tables:

- model quality;
- speed;
- energy;
- FPGA utilization;
- ablation study;
- scaling projection.

## 8. Limitations

Expected limitations:

- first model is smaller than production LLMs;
- FPGA implementation effort is high;
- HBM capacity limits model scale;
- GPU baseline quality depends on available hardware and runtime;
- power measurement may require careful methodology.

## 9. Conclusion

The conclusion should state whether the co-designed full-FPGA model actually
improves inference efficiency and under what workload assumptions.

