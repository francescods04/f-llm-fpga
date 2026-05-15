# First Week Plan

The first week should answer one question:

```text
Can we define a small FPGA-native language model that is worth implementing?
```

## Day 1 - Repository And Paper Frame

- [x] Create repository skeleton.
- [x] Create initial paper outline.
- [x] Create TODO and roadmap.
- [ ] Choose the first model size.
- [ ] Choose first tokenizer strategy.

Decision to make:

```text
Start with vocab 8k or 16k?
```

Recommendation: start with 8k for speed, then move to 16k/32k.

## Day 2 - Software Model Skeleton

- [x] Implement config object.
- [x] Implement embedding.
- [x] Implement RMSNorm.
- [x] Implement simple causal local attention.
- [x] Implement MLP.
- [x] Implement LM head.
- [x] Add greedy generation test.

## Day 3 - Data And Tokenizer

- [ ] Pick first dataset.
- [x] Add byte tokenizer for first experiments.
- [x] Add local sample corpus.
- [ ] Train or load small BPE tokenizer.
- [ ] Create tokenized dataset cache.
- [ ] Add overfit test on a tiny batch.

## Day 4 - Baseline Training

- [ ] Train tiny model.
- [ ] Save loss curve.
- [ ] Generate sample text.
- [ ] Benchmark CPU and Mac backend if available.

## Day 5 - Compression Prototype

- [ ] Add compressed global context placeholder.
- [ ] Compare standard local attention vs compressed-local hybrid.
- [ ] Record quality/speed tradeoff.

## Weekend - Research Memo

- [ ] Write 1-2 page memo:
  - model size;
  - tensor shapes;
  - memory footprint;
  - expected FPGA bottlenecks;
  - first kernel to implement.
