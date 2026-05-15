# Roadmap

## Milestone 1 - Software Model

Goal: prove the architecture can generate usable text.

Tasks:

- build PyTorch reference model;
- train a 10M-50M toy version;
- implement deterministic greedy generation;
- measure baseline speed on Mac.

Deliverable:

```text
F-LLM-small generates coherent text and has a reproducible benchmark.
```

## Milestone 2 - Quantized Reference

Goal: freeze hardware-visible math.

Tasks:

- add simulated INT8 activations;
- add INT4 weights;
- replace unsupported ops;
- export packed weights.

Deliverable:

```text
Fixed-point software model matches floating-point model within acceptable loss.
```

## Milestone 3 - First FPGA Kernel

Goal: validate the bottleneck primitive.

Tasks:

- implement packed INT4 matrix-vector engine;
- test against Python reference;
- measure cycles and resource use.

Deliverable:

```text
Matvec kernel is correct and benchmarked.
```

## Milestone 4 - Full Token Loop

Goal: generate tokens entirely on FPGA.

Tasks:

- integrate layers;
- implement LM head;
- keep decode loop on FPGA;
- stream tokens to host.

Deliverable:

```text
FPGA generates text end-to-end.
```

## Milestone 5 - Paper Results

Goal: answer the research question.

Tasks:

- benchmark against GPU;
- compute joule/token;
- write ablations;
- finalize paper.

Deliverable:

```text
Paper-ready evidence for or against the full-FPGA efficiency hypothesis.
```

