# MSA Band Prediction — Code & Data Structure Report

**Date:** April 5, 2026  
**Status:** Code complete, tested, ready for GPU training on vast.ai  
**Python:** 3.13.11 | **PyTorch:** 2.11.0 | **C++17** with pybind11 + AVX2

---

## Project Overview

Neural-guided banded Needleman-Wunsch aligner for multiple sequence alignment (MSA). A CNN+MLP neural network predicts optimal band parameters (centre diagonal, half-width) for each pairwise alignment, replacing fixed-width bands. Progressive MSA follows a guide tree, with iterative MUSCLE-style refinement.

---

## Directory Structure

```
DIPLOM/
├── aligner/                    # C++ alignment kernels (pybind11)
│   ├── full_nw.cpp             # Full O(n²) Needleman-Wunsch
│   ├── banded_nw.cpp           # Banded NW with affine gaps
│   ├── simd_banded_nw.cpp      # AVX2 SIMD-accelerated banded NW
│   ├── four_russians.cpp       # Four Russians speedup (block lookup table)
│   ├── hirschberg.cpp          # Linear-memory Hirschberg variant
│   ├── band_doubling.cpp       # Asymmetric band doubling + pybind11 bindings
│   ├── profile_dp.cpp          # Profile-profile alignment DP
│   └── anchored_align.cpp      # Anchor-based long sequence alignment
│
├── data/
│   ├── simulate.py             # Synthetic training data generation (O(n) per sample)
│   ├── loaders.py              # FASTA file loaders
│   ├── processed/
│   │   ├── train.parquet       # 49,998 DNA pairs (13.6 MB) ✅ READY
│   │   └── val.parquet         # 9,999 DNA pairs (2.7 MB) ✅ READY
│   ├── cache/
│   │   └── train/              # ~50k precomputed .npz feature files (partial)
│   └── __init__.py
│
├── features/
│   ├── profile_features.py     # MAIN: make_input() → (1,64,64) matrix + (70,) scalars
│   ├── kmer.py                 # K-mer frequency + minimizer features (SCALAR_DIM=70)
│   ├── dotplot.py              # Compressed dot-plot tensor (1×64×64)
│   ├── anchors.py              # Anchor detection for long sequences (>5000bp)
│   └── __init__.py
│
├── model/
│   ├── band_predictor.py       # BandPredictor CNN+MLP+Embedding neural network
│   ├── train.py                # Training on synthetic data (DNA + protein)
│   ├── evaluate.py             # Batched GPU inference with torch.compile
│   └── __init__.py
│
├── msa/
│   ├── guide_tree.py           # UPGMA/NJ guide tree from k-mer Jaccard distances
│   ├── progressive_msa.py      # Progressive MSA with neural band prediction
│   ├── iterative_refine.py     # MUSCLE-style iterative refinement (3 passes)
│   └── __init__.py
│
├── scoring/
│   ├── metrics.py              # SP-score, TC-score, internal SP, benchmarking
│   ├── band_metrics.py         # Band recall, width efficiency, doubling count
│   └── __init__.py
│
├── baselines/
│   ├── classical.py            # Wrappers: MAFFT, MUSCLE, ClustalW2
│   └── __init__.py
│
├── experiments/
│   ├── ablation.py             # Neural band vs fixed-width ablation study
│   ├── compare.py              # Full comparison: our method vs MAFFT/MUSCLE/ClustalW
│   └── __init__.py
│
├── tests/
│   ├── test_correctness.py     # Banded=Full NW, Hirschberg, doubling convergence
│   ├── test_four_russians.py   # FR accumulation, speed, correctness
│   ├── test_speed_pairwise.py  # Pairwise speedup benchmarks (300-5000bp)
│   ├── test_neural_vs_fixed.py # Ablation + doubling reduction (needs checkpoint)
│   ├── test_msa_quality.py     # MSA quality on synthetic DNA (needs checkpoint)
│   └── __init__.py
│
├── aligner.cp313-win_amd64.pyd # Compiled C++ module ✅ BUILT
├── CMakeLists.txt              # CMake build for C++ module
├── setup.py                    # setuptools build alternative
├── requirements.txt            # Python dependencies
├── run_data_gen.py             # Data generation runner script
├── test_training.py            # Training smoke test script
├── test_report.md              # Pre-deployment test report
└── msa_band_prediction_plan.md # Original project plan
```

---

## C++ Alignment Module (`aligner`)

Compiled to `aligner.cp313-win_amd64.pyd`. Single compilation unit via `#include` chain.

### Exported Python API

```python
import aligner

# Pairwise sequence alignment
result = aligner.align_banded(seq1, seq2, centre_diag, half_width,
                              gap_open=-10.0, gap_extend=-0.5, is_protein=False)
# result.score, result.aligned1, result.aligned2, result.escaped, result.max_deviation

result = aligner.align_hirschberg(seq1, seq2, centre_diag, half_width, ...)
# Linear-memory variant (auto-selected for >200MB matrices)

result = aligner.align_with_doubling(seq1, seq2, pred_centre, pred_hw, ...)
# result.alignment (BandedResult), result.iterations, result.final_half_width

# Profile-profile alignment (for MSA internal nodes)
result = aligner.align_profiles(profile1, profile2, subst_matrix,
                                centre_diag, half_width, ...)

result = aligner.align_profiles_with_doubling(profile1, profile2, subst, ...)

# Full NW (for ground truth / testing only)
score, aligned1, aligned2 = aligner.full_nw_align(seq1, seq2, ...)
traceback = aligner.full_nw_traceback(seq1, seq2, ...)
```

### Optimization Stack
1. **AVX2 SIMD**: Vectorized inner loop of banded NW (8×int32 lanes)
2. **Four Russians**: Block lookup table for repeated subproblems (block_size=4)
3. **Hirschberg**: Linear-memory divide-and-conquer for large matrices (>200MB)
4. **Band Doubling**: Asymmetric doubling (×2) when alignment escapes predicted band
5. **Affine Gap Penalties**: Full gap-open/gap-extend model throughout

---

## Neural Network Architecture

### Model: `BandPredictor`

```
Input 1: Similarity matrix (batch, 1, 64, 64)  ─→  DotPlotCNN  ─→ (batch, 256)
Input 2: Scalar features   (batch, 70)          ─→  ScalarMLP   ─→ (batch, 64)   ──→ Concat ─→ MLP Head ─→ (batch, 2)
Input 3: Sequence type      (batch,)             ─→  Embedding   ─→ (batch, 8)         ↑            ↓
                                                                                    (batch, 328)  [centre_diag, log_half_width]
```

**DotPlotCNN**: 4 blocks of Conv2d→BN→ReLU→Pool, then Linear(2048→256)→ReLU→Dropout(0.3)  
**ScalarMLP**: Linear(70→128)→BN→ReLU→Dropout→Linear(128→64)→ReLU  
**Head**: Linear(328→128)→ReLU→Dropout(0.2)→Linear(128→2)

### Loss Function

```python
loss = lam * Huber(pred_centre, true_centre)
     + asymmetric_huber(pred_log_hw, true_hw, penalty=5.0)
```

Asymmetric: underestimating half_width is penalised **5×** more than overestimating (underestimate → costly band doubling; overestimate → just extra compute).

### Input Features

**Matrix (1×64×64):**
- Mode 1 (sequences): k-mer dot plot, resized via `scipy.ndimage.zoom`
- Mode 2 (profiles): profile similarity matrix via `einsum('ia,ab,jb->ij', p1, subst, p2)`

**Scalars (70-dim vector):**
- K-mer frequencies (k=4 for DNA / k=3 for protein) for both sequences
- Minimizer Jaccard similarity
- Cosine similarity of k-mer vectors
- L1 distance of k-mer vectors
- Shannon entropy of each sequence
- Length ratio, GC content (DNA) / charged fraction (protein)

---

## Data Pipeline

### Synthetic Data Generation (`data/simulate.py`)

```
For each sample:
  1. Generate random seq1 (50-500bp DNA)
  2. Mutate seq1 → seq2 via mutate_with_alignment() [O(n)]
     - Tracks (i, j) alignment path during mutation
     - Substitutions, insertions, deletions at controlled rates
  3. Compute band params from known path:
     - centre_diag = median(i - j for all aligned pairs)
     - true_half_width = max|deviation from centre| + MARGIN(3)
  4. Compute divergence = substitution_count / aligned_positions
```

**Three divergence groups** (equal stratification):

| Group | p_sub | p_ins | p_del |
|-------|-------|-------|-------|
| Low | 0.01–0.08 | 0.00–0.02 | 0.00–0.02 |
| Medium | 0.08–0.20 | 0.01–0.05 | 0.01–0.05 |
| High | 0.20–0.40 | 0.03–0.10 | 0.03–0.10 |

### Generated Parquet Schema

| Column | Type | Description |
|--------|------|-------------|
| seq1 | string | Original DNA sequence |
| seq2 | string | Mutated DNA sequence |
| centre_diag | int | Median diagonal of alignment path |
| true_half_width | int | Max deviation from centre + 3 |
| divergence | float | Fraction of substituted positions |
| seq_type | string | "dna" or "protein" |

### Current Data on Disk

```
data/processed/train.parquet  — 49,998 samples (13.6 MB)
data/processed/val.parquet    — 9,999 samples (2.7 MB)
```

### Feature Cache

During training, `BandDataset.__getitem__()` computes features on first access and caches as `.npz`:
```
data/cache/train/sample_0.npz    # matrix: (1,64,64), scalars: (70,)
data/cache/train/sample_1.npz
...
data/cache/val/sample_0.npz
```

~6.64ms per `make_input()` call. First epoch builds cache (~5-7 min for 50k), subsequent epochs read from cache (~40s/epoch).

---

## Training Pipeline (`model/train.py`)

### Training

**Stage 1 — Pretrain** (synthetic data):
- WeightedRandomSampler: equal weight to low/medium/high divergence groups
- AdamW optimizer, CosineAnnealingLR scheduler
- Early stopping with patience=5 on band_recall@1x
- Checkpoints: `best_model.pt` + every 5 epochs

### CLI Usage

```bash
python -m model.train \
  --data_dir data/processed \
  --cache_dir data/cache \
  --checkpoint_dir checkpoints \
  --epochs_pretrain 20 \
  --batch_size 128 \
  --device cuda \
  --patience 5
```

### Metrics Logged Per Epoch

| Metric | Description |
|--------|-------------|
| loss | Combined centre + width loss |
| mae_centre | Mean absolute error of centre diagonal |
| band_recall@1.0x | Fraction where true_hw ≤ pred_hw × 1.0 |
| band_recall@1.5x | Fraction where true_hw ≤ pred_hw × 1.5 |
| band_recall@2.0x | Fraction where true_hw ≤ pred_hw × 2.0 |
| width_ratio | Mean pred_hw / true_hw |

---

## MSA Pipeline

### Flow

```
Input sequences
       │
       ▼
  K-mer Jaccard distance matrix (parallel)
       │
       ▼
  Guide tree (NJ or UPGMA)
       │
       ▼
  Bottom-up progressive alignment:
    For each internal node:
      1. Neural band prediction → (centre, half_width)
      2. If len > 5000bp: anchor-based block splitting
      3. Banded NW (or profile-profile DP) with doubling fallback
      4. Build profile for merged node
       │
       ▼
  Iterative refinement (3 passes):
    For each sequence:
      1. Remove from MSA
      2. Build profile of remaining
      3. Re-align sequence to profile
      4. Keep if SP-score improves
       │
       ▼
  Final MSA
```

### Key Functions

```python
from msa.progressive_msa import progressive_msa
from msa.iterative_refine import iterative_refine
from model.evaluate import BandPredictorInference

predictor = BandPredictorInference("checkpoints/best_model.pt", device="cuda")
msa = progressive_msa(sequences, seq_ids, predictor, seq_type="dna")
msa = iterative_refine(msa, sequences, predictor, seq_type="dna")
```

---

## Test Suite Summary

**20 tests total: 16 pass, 4 skip (expected)**

| File | Tests | Status | Notes |
|------|-------|--------|-------|
| `test_correctness.py` | 7 | All PASS | Banded=Full, Hirschberg, doubling |
| `test_four_russians.py` | 3 | All PASS | Table growth, speed, correctness |
| `test_speed_pairwise.py` | 6 | All PASS | Speedup verified 300–5000bp |
| `test_neural_vs_fixed.py` | 2 | SKIP | Need trained checkpoint |
| `test_msa_quality.py` | 2 | SKIP | Need trained checkpoint |

---

## Checkpoint Format

```python
{
    "epoch": int,
    "model_state": OrderedDict,    # model.state_dict()
    "optimizer_state": dict,       # only in periodic checkpoints
    "config": dict,                # training hyperparameters
    "val_metrics": {               # only in best_model.pt
        "loss": float,
        "mae_centre": float,
        "width_ratio": float,
        "band_recall@1.0x": float,
        "band_recall@1.5x": float,
        "band_recall@2.0x": float,
    }
}
```

---

## vast.ai Deployment Notes

### What to Upload
- All Python modules (`data/`, `features/`, `model/`, `msa/`, `scoring/`, `baselines/`, `experiments/`, `tests/`)
- `data/processed/train.parquet` + `val.parquet` (16.3 MB total)
- `requirements.txt`
- Do NOT upload: `data/cache/` (regenerated on GPU), `build/`, `.pytest_cache/`, `__pycache__/`
- Do NOT upload: `aligner.cp313-win_amd64.pyd` (needs recompile on Linux)

### Setup on vast.ai (Linux + CUDA)

```bash
# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pandas pyarrow scipy numpy pybind11 tqdm wandb biopython scikit-learn

# Build C++ module
pip install pybind11
python setup.py build_ext --inplace
# Produces: aligner.cpython-3XX-x86_64-linux-gnu.so

# Train
python -m model.train \
  --data_dir data/processed \
  --cache_dir data/cache \
  --checkpoint_dir checkpoints \
  --epochs_pretrain 20 \
  --batch_size 256 \
  --device cuda \
  --patience 5

# Download result: checkpoints/best_model.pt
```

### Expected GPU Training Time
- **A100/H100**: ~15-20 min total (20 epochs)
- **RTX 3090/4090**: ~25-35 min total
- **T4**: ~45-60 min total

---

## Known Issues / TODOs

1. **Protein training data** — To add protein pairs, run: `python -m data.simulate --n_samples 50000 --seq_type protein --output data/processed/train_protein.parquet`
3. **`→` character in print** — Fixed to `->` for Windows cp1251 compatibility.
4. **Cache I/O bottleneck** — 50k individual `.npz` files cause slow first epoch on HDD. On SSD/GPU instance this is negligible.
5. **C++ module needs recompile on Linux** — `setup.py build_ext --inplace` handles this, but gcc/g++ must be installed.
