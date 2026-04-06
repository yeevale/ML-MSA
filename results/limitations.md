# Known Limitations and Speed Tradeoffs

## Why our method is slower than MAFFT for large N

1. **Guide tree construction is O(N²)** — our k-mer Jaccard distance matrix requires all pairwise comparisons, while MAFFT uses O(N log N) via FFT-based distance estimation and PartTree for large N.

2. **Profile-profile DP at each tree node** — each merge step runs a full banded NW alignment on profiles, adding overhead compared to MAFFT's FFT-based profile comparison.

3. **Neural inference adds ~1-2ms per node** — the CNN+MLP prediction dominates runtime for short sequences. For N=100, this means ~99 inference calls (batched by tree level, but still non-trivial). MAFFT has no per-node ML overhead.

## Design rationale

These are known tradeoffs: our method prioritizes **alignment quality** and **interpretability** (the neural band prediction provides insight into expected structural deviation) over raw speed for large N.

The neural band prediction provides the greatest benefit for:
- Divergent sequence pairs where fixed bands are too narrow or wasteful
- Medium-length sequences (500-5000 bp) where band width selection significantly affects runtime
- Accuracy-critical applications where optimal band placement reduces alignment errors
