# Architecture

## Design Principle

The model is not a standard Transformer copied onto FPGA. It is a decoder-only
language model shaped around FPGA constraints:

- streaming token generation;
- matrix-vector rather than matrix-matrix decode;
- low-bit packed weights;
- compressed context;
- deterministic memory layout;
- hardware-controlled autoregressive loop.

## Candidate F-LLM Block

```text
x
 |
 v
RMSNorm
 |
 v
Local Attention, window W
 |
 v
Compressed Global Context Attention
 |
 v
Residual Add
 |
 v
RMSNorm
 |
 v
Low-bit MLP or MoE
 |
 v
Residual Add
```

## Attention Plan

The first model should avoid full quadratic attention over long contexts.

Proposed split:

- local exact attention over the most recent `W` tokens;
- compressed global context built from blocks of `B` tokens;
- top-k compressed block selection for global attention;
- fixed-size context state for predictable FPGA memory use.

Initial parameters:

```text
local window W: 256 or 512
compression block B: 16 or 32
selected global blocks K: 8 or 16
```

## MoE Plan

MoE is optional for the first model but important for the paper narrative.

Initial MoE shape:

```text
experts: 16-64
active experts: top-2
expert hidden dim: 2x or 4x model hidden
expert weights: INT4 or FP4-like
```

## LM Head Plan

The LM head can become the bottleneck. Start with a smaller vocabulary, then scale.

Options:

1. Streaming full-vocab argmax.
2. Hierarchical head: cluster logits, then token logits inside cluster.
3. Shortlist head for early experiments.

The first paper-grade version should prefer hierarchical or streaming full-vocab
head over a dynamic shortlist, because it is easier to compare fairly.

