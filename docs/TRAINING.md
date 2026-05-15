# Training Path

The smoke corpus is only for plumbing. A minimally usable model needs a real text
corpus and a subword tokenizer.

## Recommended First Useful Run

Download TinyStories, train a BPE tokenizer, then train the 25M preset.

```bash
PYTHONPATH=src python3 scripts/download_tinystories.py \
  --max-examples 100000 \
  --out datasets/tinystories.txt

PYTHONPATH=src python3 scripts/train_tokenizer.py \
  --corpus datasets/tinystories.txt \
  --out datasets/tokenizer-8k.json \
  --vocab-size 8192

PYTHONPATH=src python3 scripts/train_tiny.py \
  --corpus datasets/tinystories.txt \
  --tokenizer datasets/tokenizer-8k.json \
  --preset usable-25m \
  --seq-len 256 \
  --batch-size 16 \
  --steps 20000 \
  --eval-every 500 \
  --dropout 0.05 \
  --out-dir checkpoints/usable-25m
```

On a MacBook Pro M1 this may be slow in plain PyTorch if MPS is unavailable. The
purpose of this run is to establish a smart-enough reference model before FPGA
work, not to set final speed numbers.

## Model Size Presets

Approximate intent:

```text
tiny         plumbing only
small        local sanity checks
usable-25m   first minimally useful model
usable-60m   stronger local/GPU training target
paper-150m   paper-grade stretch target before FPGA constraints
```

## Quality Criteria

Do not move to FPGA until the software model passes these gates:

```text
generates coherent paragraph-level text
validation loss decreases cleanly
generation is not mostly repeated tokens
decode benchmark is reproducible
checkpoint includes tokenizer and config
```

Run:

```bash
PYTHONPATH=src python3 benchmarks/evaluate_quality.py \
  --checkpoint checkpoints/usable-25m/model.pt \
  --corpus datasets/tinystories.txt \
  --prompt "Once upon a time" \
  --prompt "A small robot learned"
```

## Why BPE Matters

The byte tokenizer is small and FPGA-friendly, but it makes learning harder and
generation uglier at small model sizes. BPE keeps the vocabulary manageable while
giving the model meaningful subword units. The FPGA head can later use a streaming
or hierarchical implementation to handle the larger vocabulary.
