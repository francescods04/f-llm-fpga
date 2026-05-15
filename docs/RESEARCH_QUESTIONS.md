# Research Questions

## Main Question

Can a full-FPGA, FPGA-native language model beat GPU energy efficiency for
batch-1 autoregressive decode?

## Subquestions

1. How much model quality is lost when moving from a GPU-friendly Transformer to an
   FPGA-friendly compressed architecture?
2. Which component dominates joule/token on FPGA: matvec, attention, LM head, MoE,
   memory movement, or host overhead?
3. Does compressed attention make the FPGA advantage larger as context length grows?
4. Does low-bit MoE improve efficiency enough to justify routing complexity?
5. Can a hierarchical LM head reduce output projection cost without unacceptable
   quality loss?
6. At what parameter count does HBM bandwidth become the limiting factor?

## Falsifiable Hypotheses

H1:

```text
For the same quantized model and greedy decode at batch=1, full-FPGA inference
achieves lower joule/token than a GPU baseline.
```

H2:

```text
Compressed attention improves joule/token more at longer contexts than at short
contexts.
```

H3:

```text
INT4 weights preserve enough language quality for a useful small LLM while
improving hardware efficiency.
```

H4:

```text
MoE improves parameter efficiency, but may hurt FPGA efficiency unless routing and
expert memory layout are hardware-aware.
```

