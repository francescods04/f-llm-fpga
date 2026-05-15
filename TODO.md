# TODO

This is the working checklist for turning the project into a defensible paper.

## Phase 0 - Paper Frame And Problem Definition

- [x] Define the full-FPGA objective.
- [x] Define the primary metric as joule/token.
- [x] Create repository skeleton.
- [ ] Lock the first baseline GPU target.
- [ ] Lock the first FPGA target board/cloud platform.
- [ ] Write the first 2-page extended abstract.
- [ ] Build the benchmark protocol before implementing kernels.

## Phase 1 - Software Reference Model

- [x] Implement a minimal decoder-only reference model in `src/fllm`.
- [x] Add local attention over a fixed window.
- [ ] Add compressed global context blocks.
- [x] Add optional low-bit MoE block in software.
- [ ] Add a hierarchical or streaming LM head.
- [x] Add greedy decoding.
- [x] Add deterministic benchmark prompts.
- [x] Add byte tokenizer for early FPGA-friendly output vocabulary.
- [x] Add BPE tokenizer training path for usable models.
- [x] Add local corpus preparation and optional TinyStories download scripts.
- [x] Add tiny training loop.
- [x] Add decode benchmark script.
- [ ] Train `usable-25m` on a real corpus.
- [x] Add repetition-aware generation metrics.
- [x] Add quality evaluator with perplexity and diversity metrics.

Exit criteria:

- The model trains or overfits on a small text corpus.
- The model generates coherent text at small scale.
- The decode loop is deterministic and benchmarkable.

## Phase 2 - Quantization And FPGA-Native Constraints

- [x] Add simulated INT8 activation quantization.
- [x] Add INT4 weight quantization.
- [ ] Test INT2 or ternary weights for selected blocks.
- [ ] Replace hardware-hostile ops with approximations where needed.
- [x] Measure perplexity and generation degradation per quantization mode (scripts/quantize_eval.py).
- [x] Export weights in FPGA-friendly packed format (src/fllm/export.py).

Exit criteria:

- INT4 model keeps usable quality.
- Weight export format is stable.
- Hardware-visible tensor shapes are frozen for the first FPGA kernel.

## Phase 3 - Baselines

- [ ] Benchmark model on Mac M1 CPU.
- [ ] Benchmark model on Mac M1 Metal/MPS or MLX.
- [ ] Benchmark model on an NVIDIA GPU baseline.
- [ ] Record wall-clock tokens/s and estimated/available power metrics.
- [ ] Document GPU kernel/runtime assumptions.

Exit criteria:

- GPU-only baseline is reproducible.
- The same model, tokenizer, prompts, and decoding are used across baselines.

## Phase 4 - FPGA Kernels

- [ ] Implement low-bit matrix-vector kernel.
- [ ] Validate kernel numerics against Python reference.
- [ ] Implement compressed-context update kernel.
- [ ] Implement local attention or recurrent compressed attention kernel.
- [ ] Implement low-bit MoE expert kernel.
- [ ] Implement LM head and argmax/sampler path.

Exit criteria:

- Each kernel has a standalone testbench.
- Each kernel reports cycles, bandwidth, resource usage, and clock.

## Phase 5 - Full FPGA Token Loop

- [ ] Load packed weights to FPGA memory.
- [ ] Keep autoregressive token loop inside FPGA.
- [ ] Avoid host roundtrip per generated token.
- [ ] Stream generated tokens back to host.
- [ ] Validate output against fixed-point software model.

Exit criteria:

- FPGA generates text end-to-end without GPU.
- Output matches fixed-point software reference under greedy decoding.

## Phase 6 - Paper Evaluation

- [ ] Compare FPGA against GPU on same model.
- [ ] Compare prefill and decode separately.
- [ ] Report joule/token, tokens/s/W, ms/token, and tokens/s.
- [ ] Add ablations for compression, quantization, and MoE.
- [ ] Add scaling projection to larger parameter counts.

Exit criteria:

- Results answer whether full-FPGA beats GPU efficiency in the target regime.
- Limitations are explicit and measured.
