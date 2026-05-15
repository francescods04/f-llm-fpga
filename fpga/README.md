# FPGA Implementation

This directory will hold HLS/RTL kernels and hardware integration notes.

## Kernel Order

1. Packed INT4 matrix-vector engine.
2. Fixed-point RMSNorm approximation.
3. Local attention kernel.
4. Compressed global context kernel.
5. Low-bit MLP or MoE expert kernel.
6. LM head and argmax/sampler.
7. Full token loop controller.

## First Kernel Contract

The first kernel should compute:

```text
y = W x
```

with:

```text
W: packed INT4
x: INT8
accumulator: INT32
output: INT8 or INT16
```

The benchmark must report:

- cycles;
- effective bandwidth;
- effective operations/s;
- resource utilization;
- numerical error vs Python reference.

