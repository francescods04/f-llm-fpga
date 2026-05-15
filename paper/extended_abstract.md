# Extended Abstract Draft

## Working Title

F-LLM: FPGA-Native Low-Bit Language Models for Energy-Efficient Batch-1 Inference

## Problem

Modern GPUs deliver high throughput for dense batched computation, but interactive
LLM inference often runs in a different regime: autoregressive decoding at small
batch sizes. In this setting, each generated token depends on the previous token,
and the serving stack repeatedly executes small memory-intensive operations. The
result is that peak GPU FLOPs are not the right predictor of energy efficiency.

FPGA accelerators offer a different design point. They can implement custom low-bit
datapaths, deterministic memory schedules, streaming control logic, and token loops
that do not require a general-purpose runtime per step. However, simply mapping a
standard dense Transformer to FPGA is unlikely to outperform a modern GPU. The
model architecture must be co-designed with the target hardware.

## Proposed Approach

We propose F-LLM, a decoder-only language model architecture designed for full-FPGA
inference. F-LLM uses low-bit matrix-vector execution, local attention for recent
tokens, compressed global context for long-range information, and optional low-bit
MoE experts. The autoregressive token loop, output projection, and sampling are
kept on FPGA, so the host does not participate in per-token scheduling.

## Research Hypothesis

For batch-1 autoregressive decoding, a full-FPGA implementation of an FPGA-native
low-bit language model can reduce joule/token relative to a GPU baseline running
the same model under the same decoding policy.

## Evaluation Plan

We will compare the same model across CPU, Mac M1 Metal/MLX, NVIDIA GPU, and FPGA.
The primary metric is joule/token. We will separately report prefill and decode,
because the hardware bottlenecks differ. We will also report perplexity, generation
examples, resource utilization, memory bandwidth, and ablations for compression,
quantization, MoE routing, and LM head design.

## Expected Contribution

This work aims to contribute:

1. An FPGA-native LLM architecture optimized for decode efficiency.
2. A full-FPGA token generation loop without a GPU verifier.
3. A reproducible benchmark methodology for joule/token comparisons.
4. An empirical answer to when FPGA inference can beat GPU inference in efficiency.

## Initial Scope

The first implementation targets a functional 50M-350M parameter model. The stretch
target is a 1B parameter class model if memory and bandwidth constraints allow.

