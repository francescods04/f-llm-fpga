# References And Reading List

This file tracks the references that shape the project. Convert to BibTeX when the
paper stabilizes.

## Direct Inspirations

- TALOS-V2: hardware implementation of a small Transformer in RTL, optimized past
  50k tokens/s. Useful as a readable example of mapping a model into explicit
  hardware blocks.
  <https://v2.talos.wtf/>

- FlightLLM: FPGA LLM inference mapping flow. Useful baseline for energy efficiency
  claims and FPGA-specific LLM mapping.
  <https://arxiv.org/abs/2401.03868>

- LUT-LLM: memory/table-lookup based LLM inference on FPGA. Important because it
  argues that FPGA advantage should come from memory-centric computation rather
  than trying to out-FLOP GPUs.
  <https://arxiv.org/abs/2511.06174>

- TeLLMe v2: ternary/table-lookup LLM accelerator for edge FPGAs. Relevant for
  low-bit matrix operations, prefill/decode separation, and resource-aware design.
  <https://arxiv.org/abs/2510.15926>

## Architecture Inspirations

- Qwen3.6-35B-A3B: MoE, Gated DeltaNet, Gated Attention, MTP, long context. Too
  large for first full-FPGA implementation, but useful for architectural ideas.
  <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>

- DeepSeek-V4 technical report: CSA/HCA compressed attention, MoE FP4/FP8 mixed
  precision, million-token context. Too large as a direct target, but valuable for
  compressed attention and low-bit MoE ideas.
  <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf>

## Hardware References

- AMD Virtex UltraScale+ HBM FPGAs: HBM capacity/bandwidth and resource constraints.
  <https://www.amd.com/en/products/adaptive-socs-and-fpgas/fpga/virtex-ultrascale-plus-hbm.html>

- AWS F2 instances: cloud FPGA option based on AMD Virtex UltraScale+ HBM VU47P.
  <https://aws.amazon.com/ec2/instance-types/f2/>

