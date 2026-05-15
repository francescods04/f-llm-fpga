# Benchmarks

Benchmark scripts will compare the same model and decoding configuration across:

- CPU;
- Mac M1 Metal/MLX or PyTorch MPS;
- NVIDIA GPU;
- FPGA.

## Required Outputs

Every benchmark should emit machine-readable JSON:

```json
{
  "model": "f-llm-small",
  "backend": "gpu",
  "precision": "int4",
  "prompt_tokens": 512,
  "generated_tokens": 128,
  "tokens_per_second": 0.0,
  "ms_per_token": 0.0,
  "watts": null,
  "joule_per_token": null
}
```

Current decode benchmark:

```bash
PYTHONPATH=src python3 benchmarks/benchmark_decode.py \
  --checkpoint checkpoints/tiny/model.pt \
  --new-tokens 64
```
