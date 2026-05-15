# BPE Quality Path - 2026-05-15

Purpose: move from byte-level plumbing to a minimally usable training path.

Changes validated:

- BPE tokenizer training through `scripts/train_tokenizer.py`.
- Checkpoint saves tokenizer metadata and copies `tokenizer.json`.
- Generation can reload checkpoint and tokenizer.
- Quality evaluator reports loss, perplexity, and simple diversity metrics.
- GPT-style initialization fixed the high initial loss caused by tied embeddings.

Smoke commands:

```bash
PYTHONPATH=src python3 scripts/train_tokenizer.py \
  --corpus docs/sample_corpus.txt \
  --out datasets/tokenizer-smoke.json \
  --vocab-size 512 \
  --min-frequency 1

PYTHONPATH=src python3 scripts/train_tiny.py \
  --corpus docs/sample_corpus.txt \
  --tokenizer datasets/tokenizer-smoke.json \
  --preset tiny \
  --seq-len 32 \
  --batch-size 2 \
  --steps 3 \
  --eval-every 3 \
  --eval-iters 1 \
  --out-dir checkpoints/bpe-smoke \
  --generate-tokens 16

PYTHONPATH=src python3 benchmarks/evaluate_quality.py \
  --checkpoint checkpoints/bpe-smoke/model.pt \
  --corpus docs/sample_corpus.txt \
  --iters 2 \
  --seq-len 32 \
  --new-tokens 12
```

Key result:

```text
BPE vocab size: 512
tiny model parameters: 100,544
initial train loss after initialization fix: 6.2425
quality eval loss: 6.2125
quality eval perplexity: 498.93
```

Interpretation:

- This is not a quality result; the corpus is far too small.
- The important result is that the usable path now exists.
- A minimally useful model now requires a real corpus such as TinyStories and the
  `usable-25m` preset.

