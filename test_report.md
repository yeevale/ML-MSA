# Pre-Deployment Test Report

**Date:** April 5, 2026  
**Platform:** Windows, Python 3.13.11, PyTorch 2.11.0+cpu  
**Purpose:** Verify pipeline correctness before deploying to vast.ai GPU server

---

## 1. Import & Dependency Check

All core modules imported successfully without errors:

| Module | Version/Status |
|--------|---------------|
| torch | 2.11.0+cpu |
| pandas | 3.0.2 |
| pyarrow | OK |
| numpy | 2.4.4 |
| scipy | OK |
| features.profile_features (make_input) | OK, SCALAR_DIM=70 |
| features.kmer | OK |
| features.dotplot | OK |
| model.band_predictor (BandPredictor, band_loss) | OK |
| data.simulate (simulate_one, generate_dataset) | OK |

---

## 2. Data Generation Verification

Synthetic training data was generated and validated:

| File | Samples | Size | Status |
|------|---------|------|--------|
| `data/processed/train.parquet` | 49,998 | 13.6 MB | Valid |
| `data/processed/val.parquet` | 9,999 | 2.7 MB | Valid |

- **Sequence type:** DNA (ACGT)
- **Length range:** 50–500 bp
- **Divergence groups:** low / medium / high (equal stratification)
- **Labels:** `centre_diag`, `true_half_width`, `divergence`
- **Generation method:** O(n) mutation tracking via `mutate_with_alignment()` — no NW alignment needed

---

## 3. Training Pipeline Smoke Test

A mini training run was executed to verify the full pipeline end-to-end.

### Configuration

| Parameter | Value |
|-----------|-------|
| Train samples | 150 |
| Val samples | 30 |
| Epochs | 3 (pretrain only) |
| Batch size | 32 |
| Learning rate | 1e-3 |
| Device | CPU |
| Loss | Asymmetric band loss (λ=2.0, penalty=5.0) |

### Results

| Epoch | Train Loss | Val Recall@1x | Val MAE (centre) |
|-------|-----------|---------------|-------------------|
| 0 | 22.6365 | 0.0333 | 2.27 |
| 1 | 18.6035 | 1.0000 | 2.27 |
| 2 | 12.1049 | 1.0000 | 2.14 |

### Best Checkpoint

- **Saved to:** `checkpoints/test_run/best_model.pt`
- **Best epoch:** 1
- **Checkpoint verification:** Loaded successfully, all fields present

**Val metrics from checkpoint:**

| Metric | Value |
|--------|-------|
| loss | 10.464 |
| mae_centre | 2.265 |
| width_ratio | 2.881 |
| band_recall@1.0x | 1.000 |
| band_recall@1.5x | 1.000 |
| band_recall@2.0x | 1.000 |

**Conclusion:** Loss decreases monotonically across epochs, recall@1x reaches 100% by epoch 1 on the small validation set. The training loop, checkpointing, and evaluation all function correctly.

---

## 4. Full Test Suite

**Command:** `python -m pytest tests/ -v --tb=short`  
**Duration:** 37.92 seconds  
**Result:** **16 passed, 4 skipped, 0 failed**

### Detailed Results

#### Correctness Tests (`test_correctness.py`) — 7/7 PASSED

| Test | Description | Result |
|------|------------|--------|
| `test_banded_equals_full[100-0.05-1]` | Banded NW = Full NW, L=100, div=5% | PASSED |
| `test_banded_equals_full[200-0.15-2]` | Banded NW = Full NW, L=200, div=15% | PASSED |
| `test_banded_equals_full[500-0.25-3]` | Banded NW = Full NW, L=500, div=25% | PASSED |
| `test_banded_equals_full[1000-0.1-4]` | Banded NW = Full NW, L=1000, div=10% | PASSED |
| `test_banded_equals_full[300-0.3-5]` | Banded NW = Full NW, L=300, div=30% | PASSED |
| `test_hirschberg_equals_banded` | Hirschberg ≈ Banded (statistical match ≥ 70%) | PASSED |
| `test_doubling_convergence` | Band doubling converges to correct score | PASSED |

#### Four Russians Tests (`test_four_russians.py`) — 3/3 PASSED

| Test | Description | Result |
|------|------------|--------|
| `test_fr_accumulation` | Lookup table grows with usage | PASSED |
| `test_fr_vs_scalar_speed` | FR timing within expected bounds | PASSED |
| `test_fr_correctness` | FR produces correct alignment scores | PASSED |

#### MSA Quality Tests (`test_msa_quality.py`) — 0/2 PASSED, 2 SKIPPED

| Test | Description | Result | Reason |
|------|------------|--------|--------|
| `test_full_comparison` | Compare MSA methods on BAliBASE | SKIPPED | BAliBASE not downloaded |
| `test_our_method_competitive` | Our MSA vs baselines | SKIPPED | BAliBASE not downloaded |

#### Neural vs Fixed Tests (`test_neural_vs_fixed.py`) — 0/2 PASSED, 2 SKIPPED

| Test | Description | Result | Reason |
|------|------------|--------|--------|
| `test_ablation_study` | Feature ablation analysis | SKIPPED | No trained checkpoint at default path |
| `test_neural_reduces_doublings` | Neural band reduces doubling count | SKIPPED | No trained checkpoint at default path |

#### Speed / Pairwise Tests (`test_speed_pairwise.py`) — 6/6 PASSED

| Test | Description | Result |
|------|------------|--------|
| `test_pairwise_speedup[300-0.05-8]` | Banded speedup, L=300 | PASSED |
| `test_pairwise_speedup[500-0.1-20]` | Banded speedup, L=500 | PASSED |
| `test_pairwise_speedup[1000-0.15-50]` | Banded speedup, L=1000 | PASSED |
| `test_pairwise_speedup[2000-0.2-120]` | Banded speedup, L=2000 | PASSED |
| `test_pairwise_speedup[5000-0.1-80]` | Banded speedup, L=5000 | PASSED |
| `test_save_results` | Results saved to JSON | PASSED |

---

## 5. Skipped Tests — Expected Resolution

The 4 skipped tests will become runnable after vast.ai training:

| Blocker | Tests Affected | Resolution |
|---------|---------------|------------|
| BAliBASE dataset not downloaded | `test_full_comparison`, `test_our_method_competitive` | Download BAliBASE v3.1, convert to parquet |
| No trained checkpoint at default path | `test_ablation_study`, `test_neural_reduces_doublings` | Train on vast.ai → copy `checkpoints/best_model.pt` |

---

## 6. Summary

| Category | Result |
|----------|--------|
| Imports & dependencies | All OK |
| Data generation | 49,998 train + 9,999 val samples valid |
| Training loop | 3 epochs completed, loss decreasing, checkpoint saved |
| Checkpoint loading | Verified, all fields present |
| Test suite | **16/16 runnable tests PASSED** |
| Skipped tests | 4 (expected — need BAliBASE + trained model) |

**Verdict: Pipeline is ready for vast.ai deployment.**
