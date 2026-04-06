# tests/test_msa_quality.py — Final BAliBASE comparison: SP-score and time.
# Run after full neural network training.
# Generates the final comparison table for the thesis.
# Run: pytest tests/test_msa_quality.py -v -s

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from pathlib import Path

import pandas as pd
import pytest

from data.loaders import BAliBASELoader
from baselines.classical import run_mafft, run_muscle, run_clustalw
from msa.progressive_msa import progressive_msa
from msa.guide_tree import pairwise_distance_matrix, build_guide_tree, tree_levels, assign_node_ids
from msa.iterative_refine import iterative_refine
from scoring.metrics import sp_score, tc_score
from model.evaluate import BandPredictorInference
import aligner

BALIBASE_DIR = "data/raw/balibase/DATASET-BALiBASE"
CHECKPOINT = "checkpoints/best_model.pt"


@pytest.fixture(scope="module")
def balibase_test():
    """Load BAliBASE test set. Skip if not found."""
    if not Path(BALIBASE_DIR).exists():
        pytest.skip(f"BAliBASE not found: {BALIBASE_DIR}")
    loader = BAliBASELoader(BALIBASE_DIR)
    _, _, test = loader.train_val_test_split()
    return test[:30]  # first 30 groups for speed


@pytest.fixture(scope="module")
def predictor():
    """Load neural band predictor. Skip if checkpoint not found."""
    if not Path(CHECKPOINT).exists():
        pytest.skip(f"Checkpoint not found: {CHECKPOINT}")
    return BandPredictorInference(CHECKPOINT, device="cpu")


def fixed_band_msa(sequences: list[str], seq_ids: list[str],
                   half_width: int = 30) -> list[str]:
    """MSA with fixed band (ablation)."""
    if len(sequences) < 2:
        return sequences

    dist_mat = pairwise_distance_matrix(sequences, "protein")
    tree = build_guide_tree(dist_mat, method="upgma")
    assign_node_ids(tree)
    levels = tree_levels(tree)

    from msa.guide_tree import get_leaves
    leaves = get_leaves(tree)
    profiles: dict[int, list[str]] = {}
    for leaf in leaves:
        if leaf.seq_idx is not None and leaf.seq_idx < len(sequences):
            profiles[leaf.node_id] = [sequences[leaf.seq_idx]]

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
                seq_a, seq_b, pred_centre=0, pred_hw=half_width
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

    root_id = tree.node_id
    return profiles.get(root_id, sequences)


def run_benchmark(aligner_fn, groups: list[dict]) -> pd.DataFrame:
    """Run an aligner function on all groups, return per-group metrics."""
    rows: list[dict] = []
    for g in groups:
        seqs = g["sequences"]
        ids = g["seq_ids"]
        ref = g["reference"]
        ref_class = g.get("ref_class", "unknown")

        if ref is None:
            print(f"  Skipping {ref_class}: no valid reference alignment")
            continue

        t0 = time.perf_counter()
        try:
            msa = aligner_fn(seqs, ids)
            elapsed = time.perf_counter() - t0
            sp = sp_score(msa, ref)
            tc = tc_score(msa, ref)
        except Exception as e:
            elapsed, sp, tc = 999.0, 0.0, 0.0
            print(f"  Error on {ref_class}: {e}")

        rows.append({
            "ref_class": ref_class,
            "sp": sp,
            "tc": tc,
            "time_s": round(elapsed, 3),
        })

    return pd.DataFrame(rows)


def test_full_comparison(balibase_test, predictor) -> None:
    """Final comparison table of all methods on BAliBASE (5 groups for speed)."""
    groups = balibase_test[:5]
    methods = {
        "ClustalW": lambda s, ids: run_clustalw(s, ids),
        "MAFFT": lambda s, ids: run_mafft(s, ids),
        "MUSCLE": lambda s, ids: run_muscle(s, ids),
        "Fixed_W30": lambda s, ids: fixed_band_msa(s, ids, 30),
        "Fixed_W100": lambda s, ids: fixed_band_msa(s, ids, 100),
        "Neural_band": lambda s, ids: progressive_msa(
            s, ids, predictor, seq_type="protein"),
        "Neural_+_refine": lambda s, ids: iterative_refine(
            progressive_msa(s, ids, predictor, seq_type="protein"),
            s, predictor, seq_type="protein"),
    }

    all_results: dict[str, pd.DataFrame] = {}
    for name, fn in methods.items():
        print(f"\nRunning {name}...")
        df = run_benchmark(fn, groups)
        all_results[name] = df
        if df.empty:
            print(f"  (no groups with valid reference)")
        else:
            print(f"  SP={df.sp.mean():.3f}, TC={df.tc.mean():.3f}, "
                  f"Time={df.time_s.mean():.2f}s")

    # Skip if no valid results
    if all(df.empty for df in all_results.values()):
        pytest.skip("No BAliBASE groups with valid reference alignments")

    # Summary table
    summary = pd.DataFrame({
        name: {
            "SP_mean": round(df.sp.mean(), 3) if not df.empty else 0.0,
            "TC_mean": round(df.tc.mean(), 3) if not df.empty else 0.0,
            "Time_mean": round(df.time_s.mean(), 2) if not df.empty else 0.0,
        }
        for name, df in all_results.items()
    }).T

    Path("results").mkdir(parents=True, exist_ok=True)
    summary.to_csv("results/final_comparison.csv")
    print("\n=== FINAL TABLE ===")
    print(summary.to_string())


def test_our_method_competitive(balibase_test, predictor) -> None:
    """Our neural band method should have reasonable SP-score."""
    df = run_benchmark(
        lambda s, ids: progressive_msa(
            s, ids, predictor, seq_type="protein"),
        balibase_test[:5]
    )
    if df.empty:
        pytest.skip("No BAliBASE groups with valid reference alignments found")
    mean_sp = df.sp.mean()
    print(f"\nOur method mean SP: {mean_sp:.3f}")
    assert mean_sp > 0.3, f"SP-score too low: {mean_sp:.3f}"


if __name__ == "__main__":
    print("Smoke test: test_msa_quality.py")
    print("Checking imports...")
    from data.loaders import BAliBASELoader
    from baselines.classical import run_mafft
    from scoring.metrics import sp_score, tc_score
    from model.evaluate import BandPredictorInference
    print("All imports OK.")
    if not Path(BALIBASE_DIR).exists():
        print(f"BAliBASE not found at {BALIBASE_DIR} (expected before download)")
    if not Path(CHECKPOINT).exists():
        print(f"Checkpoint not found at {CHECKPOINT} (expected before training)")
    print("Smoke test passed!")
