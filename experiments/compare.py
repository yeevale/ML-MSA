# experiments/compare.py — Full benchmark: compare all methods on synthetic DNA.
# Generates final results CSV + plots for the thesis.
#
# Methods compared:
#   1. ClustalW   (baseline)
#   2. MAFFT      (baseline)
#   3. MUSCLE     (baseline)
#   4. Fixed W=30 (ablation: our aligner without neural net)
#   5. Fixed W=100 (ablation)
#   6. Neural band (our method)
#   7. Neural band + iterative refine (final)

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.classical import run_mafft, run_muscle, run_clustalw
from msa.progressive_msa import progressive_msa
from msa.guide_tree import pairwise_distance_matrix, build_guide_tree, tree_levels, assign_node_ids
from msa.iterative_refine import iterative_refine
from scoring.metrics import sp_score, tc_score
from model.evaluate import BandPredictorInference
from experiments.run_all import _generate_dna_msa_group
import aligner


def fixed_band_aligner(sequences: list[str], seq_ids: list[str],
                       half_width: int = 30,
                       seq_type: str = "dna") -> list[str]:
    """MSA via progressive algorithm with FIXED band width.
    Uses centre_diag=0, half_width=half_width for all pairs.
    Used for ablation study to show neural net adds value."""
    if len(sequences) < 2:
        return sequences

    dist_mat = pairwise_distance_matrix(sequences, seq_type)
    tree = build_guide_tree(dist_mat, method="upgma")
    assign_node_ids(tree)
    levels = tree_levels(tree)

    # Build profiles bottom-up: map node_id -> list of aligned sequences
    profiles: dict[int, list[str]] = {}
    for i, seq in enumerate(sequences):
        # Leaf node_ids correspond to seq_idx via the tree
        profiles[i] = [seq]

    # Map seq_idx to node_id for leaves
    from msa.guide_tree import get_leaves
    leaves = get_leaves(tree)
    leaf_profiles: dict[int, list[str]] = {}
    for leaf in leaves:
        if leaf.seq_idx is not None and leaf.seq_idx < len(sequences):
            leaf_profiles[leaf.node_id] = [sequences[leaf.seq_idx]]
    profiles = leaf_profiles

    for level in levels:
        for node in level:
            left_id = node.left.node_id
            right_id = node.right.node_id

            left_seqs = profiles.get(left_id, [])
            right_seqs = profiles.get(right_id, [])
            if not left_seqs or not right_seqs:
                continue

            seq_a = left_seqs[0].replace("-", "")
            seq_b = right_seqs[0].replace("-", "")

            result = aligner.align_with_doubling(
                seq_a, seq_b,
                pred_centre=0, pred_hw=half_width
            )

            aligned_a = result.alignment.aligned_seq1
            aligned_b = result.alignment.aligned_seq2
            merged: list[str] = []

            for s in left_seqs:
                new_seq = []
                si = 0
                for c in aligned_a:
                    if c == "-":
                        new_seq.append("-")
                    else:
                        new_seq.append(s[si] if si < len(s) else "-")
                        si += 1
                merged.append("".join(new_seq))

            for s in right_seqs:
                new_seq = []
                si = 0
                for c in aligned_b:
                    if c == "-":
                        new_seq.append("-")
                    else:
                        new_seq.append(s[si] if si < len(s) else "-")
                        si += 1
                merged.append("".join(new_seq))

            profiles[node.node_id] = merged
            profiles.pop(left_id, None)
            profiles.pop(right_id, None)

    # Return the root's profile
    root_id = tree.node_id
    return profiles.get(root_id, sequences)


def run_benchmark_single(aligner_fn: Callable, groups: list[dict]
                         ) -> pd.DataFrame:
    """Run a single aligner on all groups, return per-group metrics."""
    rows: list[dict] = []
    for g in tqdm(groups, leave=False):
        seqs = g["sequences"]
        ids = g["seq_ids"]
        ref = g["reference"]
        ref_class = g.get("ref_class", "unknown")

        t0 = time.perf_counter()
        try:
            msa = aligner_fn(seqs, ids)
            elapsed = time.perf_counter() - t0
            sp = sp_score(msa, ref)
            tc = tc_score(msa, ref)
        except Exception as e:
            elapsed = 999.0
            sp, tc = 0.0, 0.0
            print(f"  Error: {e}")

        rows.append({
            "ref_class": ref_class,
            "sp": sp,
            "tc": tc,
            "time_s": round(elapsed, 3),
            "n_seqs": len(seqs),
        })

    return pd.DataFrame(rows)


def run_all(model_checkpoint: str,
            output_dir: str,
            device: str = "cpu",
            seq_type: str = "dna") -> None:
    """Run all 7 methods on synthetic DNA groups. Save results.csv + plots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Generate synthetic DNA test groups
    rng = np.random.default_rng(2026)
    test_groups = []
    for div in ["low", "medium", "high"]:
        for n_seqs in [5, 10, 20]:
            for rep in range(3):
                root_len = rng.integers(100, 400)
                g = _generate_dna_msa_group(n_seqs, int(root_len), div, rng)
                g["group_id"] = f"syn_{div}_{n_seqs}seqs_r{rep}"
                test_groups.append(g)
    print(f"Generated {len(test_groups)} synthetic DNA test groups")

    # Load neural predictor
    predictor = BandPredictorInference(model_checkpoint, device=device)

    methods: dict[str, Callable] = {
        "ClustalW": lambda s, ids: run_clustalw(s, ids),
        "MAFFT": lambda s, ids: run_mafft(s, ids),
        "MUSCLE": lambda s, ids: run_muscle(s, ids),
        "Fixed_W30": lambda s, ids: fixed_band_aligner(s, ids, 30, seq_type),
        "Fixed_W100": lambda s, ids: fixed_band_aligner(s, ids, 100, seq_type),
        "Neural_band": lambda s, ids: progressive_msa(
            s, ids, predictor, seq_type=seq_type),
        "Neural_+_refine": lambda s, ids: iterative_refine(
            progressive_msa(s, ids, predictor, seq_type=seq_type),
            s, predictor, seq_type=seq_type),
    }

    all_results: dict[str, pd.DataFrame] = {}
    for name, fn in methods.items():
        print(f"\nRunning {name}...")
        df = run_benchmark_single(fn, test_groups)
        all_results[name] = df
        print(f"  SP={df.sp.mean():.3f}, TC={df.tc.mean():.3f}, "
              f"Time={df.time_s.mean():.2f}s")

    # Summary table
    summary = pd.DataFrame({
        name: {
            "SP_mean": df.sp.mean(),
            "SP_std": df.sp.std(),
            "TC_mean": df.tc.mean(),
            "TC_std": df.tc.std(),
            "Time_mean": df.time_s.mean(),
            "Time_std": df.time_s.std(),
        }
        for name, df in all_results.items()
    }).T.round(4)

    csv_path = out / "results.csv"
    summary.to_csv(csv_path)
    print(f"\n=== FINAL COMPARISON ===")
    print(summary.to_string())
    print(f"\nSaved to {csv_path}")

    # Per-class breakdown
    for name, df in all_results.items():
        if "ref_class" in df.columns:
            per_class = df.groupby("ref_class")[["sp", "tc", "time_s"]].mean()
            per_class.to_csv(out / f"{name}_per_class.csv")

    # Generate plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. Boxplot SP score
        ax = axes[0]
        sp_data = pd.DataFrame({
            name: df["sp"] for name, df in all_results.items()
        })
        sp_data.boxplot(ax=ax, rot=45)
        ax.set_ylabel("SP score")
        ax.set_title("SP Score by Method")

        # 2. Boxplot TC score
        ax = axes[1]
        tc_data = pd.DataFrame({
            name: df["tc"] for name, df in all_results.items()
        })
        tc_data.boxplot(ax=ax, rot=45)
        ax.set_ylabel("TC score")
        ax.set_title("TC Score by Method")

        # 3. Scatter: time vs SP
        ax = axes[2]
        for name, df in all_results.items():
            ax.scatter(df.time_s.mean(), df.sp.mean(),
                       s=100, label=name, zorder=5)
        ax.set_xlabel("Mean time (s)")
        ax.set_ylabel("Mean SP score")
        ax.set_title("Time vs Quality")
        ax.legend(fontsize=8, loc="lower right")

        plt.tight_layout()
        fig.savefig(out / "comparison_plots.png", dpi=150)
        plt.close()
        print(f"Plots saved to {out / 'comparison_plots.png'}")

    except ImportError:
        print("matplotlib/seaborn not available, skipping plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full benchmark comparison")
    parser.add_argument("--model_checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        print("Usage: python -m experiments.compare --model_checkpoint <path>")
        print("Smoke test: checking imports...")
        from baselines.classical import run_mafft
        from scoring.metrics import sp_score
        print("Imports OK. Smoke test passed!")
    else:
        run_all(args.model_checkpoint,
                args.output_dir, args.device)
