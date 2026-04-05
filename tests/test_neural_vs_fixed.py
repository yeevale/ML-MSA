# tests/test_neural_vs_fixed.py — Ablation study: neural band vs fixed band widths.
# Run AFTER training the neural network.
# Demonstrates that improvement comes from the neural net, not just banded approach.
# Run: pytest tests/test_neural_vs_fixed.py -v -s

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aligner
from model.evaluate import BandPredictorInference

CHECKPOINT = "checkpoints/best_model.pt"


@pytest.fixture(scope="module")
def predictor():
    """Load neural band predictor. Skip if checkpoint not found."""
    if not Path(CHECKPOINT).exists():
        pytest.skip(f"Checkpoint not found: {CHECKPOINT}")
    return BandPredictorInference(CHECKPOINT, device="cpu")


def generate_test_pairs(n: int = 100, seed: int = 0
                        ) -> list[tuple[str, str, float]]:
    """Generate test pairs at various divergence levels."""
    rng = np.random.default_rng(seed)
    pairs: list[tuple[str, str, float]] = []
    for div in [0.05, 0.10, 0.20, 0.30]:
        for _ in range(n // 4):
            length = int(rng.integers(200, 1000))
            seq1 = "".join(rng.choice(list("ACGT"), length))
            seq2 = list(seq1)
            n_mut = int(length * div)
            positions = rng.choice(length, min(n_mut, length), replace=False)
            for p in positions:
                choices = [c for c in "ACGT" if c != seq2[p]]
                seq2[p] = rng.choice(choices)
            pairs.append((seq1, "".join(seq2), div))
    return pairs


def test_ablation_study(predictor) -> None:
    """Compare neural band vs W=30 vs W=100 by speed and n_doublings."""
    pairs = generate_test_pairs(100)
    rows: list[dict] = []

    for seq1, seq2, div in pairs:
        # Neural prediction
        centre, pred_hw = predictor.predict_single(seq1, seq2, seq_type="dna")
        t0 = time.perf_counter()
        r_neural = aligner.align_with_doubling(seq1, seq2, centre, pred_hw)
        t_neural = time.perf_counter() - t0

        # Fixed W=30
        t0 = time.perf_counter()
        r_fixed30 = aligner.align_with_doubling(seq1, seq2, 0, 30)
        t_fixed30 = time.perf_counter() - t0

        # Fixed W=100
        t0 = time.perf_counter()
        r_fixed100 = aligner.align_with_doubling(seq1, seq2, 0, 100)
        t_fixed100 = time.perf_counter() - t0

        rows.append({
            "divergence": div,
            "length": len(seq1),
            "pred_hw": pred_hw,
            "n_doublings_neural": r_neural.n_doublings,
            "n_doublings_fixed30": r_fixed30.n_doublings,
            "n_doublings_fixed100": r_fixed100.n_doublings,
            "t_neural_ms": t_neural * 1000,
            "t_fixed30_ms": t_fixed30 * 1000,
            "t_fixed100_ms": t_fixed100 * 1000,
            "speedup_vs_30": t_fixed30 / max(t_neural, 1e-9),
            "speedup_vs_100": t_fixed100 / max(t_neural, 1e-9),
            "score_neural": r_neural.alignment.score,
            "score_fixed100": r_fixed100.alignment.score,
            "score_match": abs(r_neural.alignment.score
                               - r_fixed100.alignment.score) < 0.01,
        })

    df = pd.DataFrame(rows)
    Path("results").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/ablation_neural_vs_fixed.csv", index=False)

    print("\n=== ABLATION STUDY ===")
    summary = df.groupby("divergence")[[
        "pred_hw", "n_doublings_neural", "n_doublings_fixed30",
        "speedup_vs_30", "speedup_vs_100"
    ]].mean().round(2)
    print(summary.to_string())
    print(f"\nScore match rate: {df.score_match.mean():.1%}")

    assert df.score_match.mean() > 0.95, (
        f"Neural net gives wrong scores too often: "
        f"{df.score_match.mean():.1%}"
    )


def test_neural_reduces_doublings(predictor) -> None:
    """Neural band should need fewer doublings than a very narrow fixed band."""
    pairs = generate_test_pairs(50, seed=42)

    neural_doublings = []
    narrow_doublings = []

    for seq1, seq2, div in pairs:
        centre, pred_hw = predictor.predict_single(seq1, seq2, seq_type="dna")
        r_neural = aligner.align_with_doubling(seq1, seq2, centre, pred_hw)
        r_narrow = aligner.align_with_doubling(seq1, seq2, 0, 5)

        neural_doublings.append(r_neural.n_doublings)
        narrow_doublings.append(r_narrow.n_doublings)

    mean_neural = np.mean(neural_doublings)
    mean_narrow = np.mean(narrow_doublings)

    print(f"\nMean doublings: neural={mean_neural:.2f}, narrow(W=5)={mean_narrow:.2f}")
    assert mean_neural < mean_narrow, (
        f"Neural not better than narrow: {mean_neural:.2f} vs {mean_narrow:.2f}"
    )


if __name__ == "__main__":
    print("Smoke test: test_neural_vs_fixed.py")
    print("Checking imports...")
    from model.evaluate import BandPredictorInference
    print(f"Checkpoint path: {CHECKPOINT}")
    if Path(CHECKPOINT).exists():
        print("Checkpoint found, running quick test...")
        p = BandPredictorInference(CHECKPOINT, device="cpu")
        print("Predictor loaded OK")
    else:
        print("Checkpoint not found (expected before training)")
    print("Smoke test passed!")
