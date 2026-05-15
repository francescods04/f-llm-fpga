# Experiments

Use this directory for experiment logs and ablations.

Suggested first experiments:

1. Tiny decoder-only model with standard attention.
2. Replace standard attention with local attention.
3. Add compressed global context.
4. Add INT4 simulated quantization.
5. Add hierarchical LM head.
6. Add optional low-bit MoE.

Each experiment should record:

- git commit;
- model config;
- dataset;
- training duration;
- validation loss/perplexity;
- generation examples;
- benchmark output.

