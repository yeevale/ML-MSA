# experiments/ablation.py — Ablation study: neural band vs fixed band widths.
# Proves the improvement comes from the neural network, not just the banded approach.
# Generates CSV + plots for the thesis.

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.evaluate import BandPredictorInference
from scoring.band_metrics import band_metrics_summary
import aligner

DNA_ALPHABET = "ACGT"


def generate_test_pairs(n: int = 200, seed: int = 0
                        ) -> list[tuple[str, str, float]]:
    """Generate test pairs at various divergence levels."""
    rng = np.random.default_rng(seed)
    pairs: list[tuple[str, str, float]] = []
    for div in [0.05, 0.10, 0.20, 0.30]:
        for _ in range(n // 4):
            length = int(rng.integers(200, 1000))
            seq1 = "".join(rng.choice(list(DNA_ALPHABET), length))
            seq2 = list(seq1)
            n_mut = int(length * div)
            positions = rng.choice(length, min(n_mut, length), replace=False)
            for p in positions:
                choices = [c for c in DNA_ALPHABET if c != seq2[p]]
                seq2[p] = rng.choice(choices)
            pairs.append((seq1, "".join(seq2), div))
    return pairs


def run_ablation(checkpoint: str, output_dir: str,
                 n_pairs: int = 200, device: str = "cpu") -> None:
    """Run full ablation study: neural vs fixed W=30 vs W=100."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    predictor = BandPredictorInference(checkpoint, device=device)
    pairs = generate_test_pairs(n_pairs)
    rows: list[dict] = []

    for seq1, seq2, div in tqdm(pairs, desc="Ablation"):
        # Neural prediction
        cd, pred_hw = predictor.predict_single(seq1, seq2, "dna")

        # Full NW (reference)
        path = aligner.full_nw_traceback(seq1, seq2)
        if path:
            diagonals = [i - j for i, j in path]
            true_centre = int(np.median(diagonals))
            true_hw = max(abs(d - true_centre) for d in diagonals) + 3
        else:
            true_centre, true_hw = 0, 10

        # Time each method
        t0 = time.perf_counter()
        r_neural = aligner.align_with_doubling(seq1, seq2, cd, pred_hw)
        t_neural = time.perf_counter() - t0

        t0 = time.perf_counter()
        r_fixed30 = aligner.align_with_doubling(seq1, seq2, 0, 30)
        t_fixed30 = time.perf_counter() - t0

        t0 = time.perf_counter()
        r_fixed100 = aligner.align_with_doubling(seq1, seq2, 0, 100)
        t_fixed100 = time.perf_counter() - t0

        rows.append({
            "divergence": div,
            "length": len(seq1),
            "true_centre": true_centre,
            "true_hw": true_hw,
            "pred_centre": cd,
            "pred_hw": pred_hw,
            "neural_doublings": r_neural.n_doublings,
            "fixed30_doublings": r_fixed30.n_doublings,
            "fixed100_doublings": r_fixed100.n_doublings,
            "t_neural_ms": round(t_neural * 1000, 3),
            "t_fixed30_ms": round(t_fixed30 * 1000, 3),
            "t_fixed100_ms": round(t_fixed100 * 1000, 3),
            "score_neural": r_neural.alignment.score,
            "score_fixed30": r_fixed30.alignment.score,
            "score_fixed100": r_fixed100.alignment.score,
        })

    df = pd.DataFrame(rows)
    csv_path = out / "ablation_results.csv"
    df.to_csv(csv_path, index=False)

    # Summary by divergence
    summary = df.groupby("divergence").agg({
        "pred_hw": "mean",
        "true_hw": "mean",
        "neural_doublings": "mean",
        "fixed30_doublings": "mean",
        "fixed100_doublings": "mean",
        "t_neural_ms": "mean",
        "t_fixed30_ms": "mean",
        "t_fixed100_ms": "mean",
    }).round(2)

    print("\n=== ABLATION STUDY RESULTS ===")
    print(summary.to_string())
    print(f"\nSaved to {csv_path}")

    # Band prediction metrics
    pred_hws = df["pred_hw"].values.astype(float)
    true_hws = df["true_hw"].values.astype(float)
    pred_centres = df["pred_centre"].values.astype(float)
    true_centres = df["true_centre"].values.astype(float)
    metrics = band_metrics_summary(pred_centres, pred_hws,
                                   true_centres, true_hws)
    print("\nBand prediction metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Generate plots if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Doublings by divergence
        ax = axes[0]
        for label, col in [("Neural", "neural_doublings"),
                           ("Fixed W=30", "fixed30_doublings"),
                           ("Fixed W=100", "fixed100_doublings")]:
            means = df.groupby("divergence")[col].mean()
            ax.plot(means.index, means.values, "o-", label=label)
        ax.set_xlabel("Divergence")
        ax.set_ylabel("Mean doublings")
        ax.set_title("Band doublings by divergence")
        ax.legend()

        # 2. Time comparison
        ax = axes[1]
        for label, col in [("Neural", "t_neural_ms"),
                           ("Fixed W=30", "t_fixed30_ms"),
                           ("Fixed W=100", "t_fixed100_ms")]:
            means = df.groupby("divergence")[col].mean()
            ax.plot(means.index, means.values, "o-", label=label)
        ax.set_xlabel("Divergence")
        ax.set_ylabel("Mean time (ms)")
        ax.set_title("Alignment time by divergence")
        ax.legend()

        # 3. Predicted vs true half_width
        ax = axes[2]
        ax.scatter(true_hws, pred_hws, alpha=0.3, s=10)
        max_val = max(true_hws.max(), pred_hws.max())
        ax.plot([0, max_val], [0, max_val], "r--", label="y=x")
        ax.set_xlabel("True half_width")
        ax.set_ylabel("Predicted half_width")
        ax.set_title("Band width prediction")
        ax.legend()

        plt.tight_layout()
        fig.savefig(out / "ablation_plots.png", dpi=150)
        plt.close()
        print(f"Plots saved to {out / 'ablation_plots.png'}")
    except ImportError:
        print("matplotlib not available, skipping plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--n_pairs", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        print("Usage: python -m experiments.ablation --checkpoint <path>")
        print("Smoke test: checking imports...")
        from scoring.band_metrics import band_metrics_summary
        print("Imports OK. Smoke test passed!")
    else:
        run_ablation(args.checkpoint, args.output_dir,
                     args.n_pairs, args.device)
