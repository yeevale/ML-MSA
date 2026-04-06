# scoring/metrics.py — Standard MSA quality metrics (BAliBASE benchmark).
# SP-score and TC-score computed against reference alignment.
# sp_score_internal — for iterative refinement (no reference needed).
# benchmark() — run aligner on dataset, measure SP, TC, time, memory.

import numpy as np
import time
import tracemalloc
from collections import defaultdict


def _build_residue_map(msa: list[str]) -> list[dict[int, int]]:
    """For each sequence in MSA, build mapping:
    column_index → residue_index (skipping gaps).
    Returns list of dicts, one per sequence."""
    maps = []
    for seq in msa:
        m = {}
        res_idx = 0
        for col, ch in enumerate(seq):
            if ch != '-':
                m[col] = res_idx
                res_idx += 1
        maps.append(m)
    return maps


def _match_sequences(predicted_msa: list[str],
                     reference_msa: list[str]) -> list[str] | None:
    """Reorder predicted_msa to match reference_msa sequence order.
    Matches by ungapped sequence content (case-insensitive).
    Returns reordered predicted_msa or None if matching fails."""
    if len(predicted_msa) != len(reference_msa):
        return None

    pred_ungapped = [s.replace('-', '').upper() for s in predicted_msa]
    ref_ungapped = [s.replace('-', '').upper() for s in reference_msa]

    reordered: list[str | None] = [None] * len(reference_msa)
    used = [False] * len(predicted_msa)

    for i, ref_seq in enumerate(ref_ungapped):
        for j, pred_seq in enumerate(pred_ungapped):
            if not used[j] and pred_seq == ref_seq:
                reordered[i] = predicted_msa[j]
                used[j] = True
                break

    if any(r is None for r in reordered):
        return None
    return reordered


def debug_sp(predicted_msa: list[str], reference_msa: list[str],
             label: str = "") -> None:
    """Print diagnostic info for SP-score debugging."""
    print(f"[SP DEBUG {label}]")
    print(f"  pred seqs: {len(predicted_msa)}, "
          f"cols: {len(predicted_msa[0]) if predicted_msa else 0}")
    print(f"  ref  seqs: {len(reference_msa)}, "
          f"cols: {len(reference_msa[0]) if reference_msa else 0}")
    print(f"  pred[0][:80]: {predicted_msa[0][:80] if predicted_msa else 'EMPTY'}")
    print(f"  ref[0][:80]:  {reference_msa[0][:80] if reference_msa else 'EMPTY'}")
    gaps_pred = [s.count('-') for s in predicted_msa]
    gaps_ref = [s.count('-') for s in reference_msa]
    print(f"  gaps in pred: {gaps_pred[:5]}")
    print(f"  gaps in ref:  {gaps_ref[:5]}")
    # Check sequence matching
    pred_ug = [s.replace('-', '').upper() for s in predicted_msa]
    ref_ug = [s.replace('-', '').upper() for s in reference_msa]
    order_match = all(p == r for p, r in zip(pred_ug, ref_ug))
    print(f"  sequence order match: {order_match}")
    if not order_match:
        matched = _match_sequences(predicted_msa, reference_msa)
        print(f"  reorder possible: {matched is not None}")


def sp_score(predicted_msa: list[str],
             reference_msa: list[str]) -> float:
    """Sum-of-Pairs score (BAliBASE standard).
    Handles sequence order mismatch by matching ungapped content."""
    n = len(reference_msa)
    if n < 2:
        return 1.0
    if len(predicted_msa) != n:
        return 0.0

    # Match sequence order by ungapped content
    pred_ungapped = [s.replace('-', '').upper() for s in predicted_msa]
    ref_ungapped = [s.replace('-', '').upper() for s in reference_msa]
    if any(p != r for p, r in zip(pred_ungapped, ref_ungapped)):
        matched = _match_sequences(predicted_msa, reference_msa)
        if matched is None:
            return 0.0
        predicted_msa = matched

    # Validate internal consistency (all seqs same length within each MSA)
    if any(len(s) != len(reference_msa[0]) for s in reference_msa):
        return 0.0
    if any(len(s) != len(predicted_msa[0]) for s in predicted_msa):
        return 0.0

    ref_maps = _build_residue_map(reference_msa)
    pred_maps = _build_residue_map(predicted_msa)

    # Build reverse maps for predicted: residue_idx → column
    pred_reverse: list[dict[int, int]] = []
    for seq_idx in range(n):
        rev = {}
        for col, res in pred_maps[seq_idx].items():
            rev[res] = col
        pred_reverse.append(rev)

    correct = 0
    total = 0
    L_ref = len(reference_msa[0])

    for k in range(L_ref):
        for i in range(n):
            if reference_msa[i][k] == '-':
                continue
            ri_i = ref_maps[i].get(k)
            if ri_i is None:
                continue
            for j in range(i + 1, n):
                if reference_msa[j][k] == '-':
                    continue
                ri_j = ref_maps[j].get(k)
                if ri_j is None:
                    continue

                total += 1
                pred_col_i = pred_reverse[i].get(ri_i)
                pred_col_j = pred_reverse[j].get(ri_j)
                if pred_col_i is not None and pred_col_j is not None:
                    if pred_col_i == pred_col_j:
                        correct += 1

    return correct / max(total, 1)


def tc_score(predicted_msa: list[str],
             reference_msa: list[str]) -> float:
    """Total Column score.
    Handles sequence order mismatch by matching ungapped content."""
    n = len(reference_msa)
    if n < 2:
        return 1.0
    if len(predicted_msa) != n:
        return 0.0

    # Match sequence order by ungapped content
    pred_ungapped = [s.replace('-', '').upper() for s in predicted_msa]
    ref_ungapped = [s.replace('-', '').upper() for s in reference_msa]
    if any(p != r for p, r in zip(pred_ungapped, ref_ungapped)):
        matched = _match_sequences(predicted_msa, reference_msa)
        if matched is None:
            return 0.0
        predicted_msa = matched

    # Validate internal consistency
    if any(len(s) != len(reference_msa[0]) for s in reference_msa):
        return 0.0
    if any(len(s) != len(predicted_msa[0]) for s in predicted_msa):
        return 0.0

    ref_maps = _build_residue_map(reference_msa)
    pred_maps = _build_residue_map(predicted_msa)

    # For predicted: build (seq_idx, residue_idx) → column
    pred_reverse: list[dict[int, int]] = []
    for seq_idx in range(n):
        rev = {}
        for col, res in pred_maps[seq_idx].items():
            rev[res] = col
        pred_reverse.append(rev)

    L_ref = len(reference_msa[0])
    matching = 0
    total = 0

    for k in range(L_ref):
        # Get all non-gap residue indices for this column
        residues: list[tuple[int, int]] = []  # (seq_idx, residue_idx)
        for i in range(n):
            if reference_msa[i][k] != '-':
                ri = ref_maps[i].get(k)
                if ri is not None:
                    residues.append((i, ri))

        if len(residues) < 2:
            continue

        total += 1

        # Check: are all these residues in the same predicted column?
        pred_cols = set()
        all_found = True
        for seq_idx, res_idx in residues:
            pc = pred_reverse[seq_idx].get(res_idx)
            if pc is None:
                all_found = False
                break
            pred_cols.add(pc)

        if all_found and len(pred_cols) == 1:
            matching += 1

    return matching / max(total, 1)


def sp_score_internal(msa: list[str], seq_type: str = "dna") -> float:
    """SP-score without reference — for iterative refinement."""
    n = len(msa)
    if n < 2:
        return 0.0
    L = len(msa[0])
    if L == 0 or any(len(s) != L for s in msa):
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(L):
                ci = msa[i][k]
                cj = msa[j][k]
                if ci != '-' and cj != '-':
                    total += (1.0 if ci.upper() == cj.upper() else -1.0)
                    count += 1
    return total / max(count, 1)


def profile_consensus(profile: np.ndarray, seq_type: str = "dna") -> str:
    """Consensus string from profile: argmax symbol at each column.
    If argmax is gap symbol → '-'."""
    DNA_ALPHA = "ACGT-"
    PROTEIN_ALPHA = "ACDEFGHIKLMNPQRSTVWY-"
    alpha = DNA_ALPHA if seq_type == "dna" else PROTEIN_ALPHA
    result: list[str] = []
    for i in range(profile.shape[0]):
        idx = int(np.argmax(profile[i]))
        result.append(alpha[idx])
    return "".join(result)


def benchmark(aligner_func,
              dataset: list[dict],
              measure_memory: bool = True) -> dict:
    """Run aligner_func on all examples and measure metrics.
    aligner_func(sequences: list[str], seq_ids: list[str]) → list[str]
    Returns dict with mean±std of each metric, grouped by ref_class.

    dataset entries must have keys:
      sequences, seq_ids, reference, ref_class, group_id
    """
    results: list[dict] = []

    for group in dataset:
        seqs = group["sequences"]
        ids = group["seq_ids"]
        ref = group["reference"]
        ref_class = group["ref_class"]

        if measure_memory:
            tracemalloc.start()

        t0 = time.perf_counter()
        try:
            predicted = aligner_func(seqs, ids)
        except Exception as e:
            results.append({
                "group_id": group["group_id"],
                "ref_class": ref_class,
                "sp_score": 0.0,
                "tc_score": 0.0,
                "time_s": 0.0,
                "peak_mb": 0.0,
                "error": str(e),
            })
            if measure_memory:
                tracemalloc.stop()
            continue

        elapsed = time.perf_counter() - t0

        peak_mb = 0.0
        if measure_memory:
            _, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / (1024 * 1024)
            tracemalloc.stop()

        sp = sp_score(predicted, ref)
        tc = tc_score(predicted, ref)

        results.append({
            "group_id": group["group_id"],
            "ref_class": ref_class,
            "sp_score": sp,
            "tc_score": tc,
            "time_s": elapsed,
            "peak_mb": peak_mb,
        })

    # Aggregate by ref_class
    by_class = defaultdict(list)
    for r in results:
        by_class[r["ref_class"]].append(r)

    summary: dict = {"per_group": results, "by_class": {}}
    for cls, group_results in sorted(by_class.items()):
        sps = [r["sp_score"] for r in group_results]
        tcs = [r["tc_score"] for r in group_results]
        times = [r["time_s"] for r in group_results]
        mems = [r["peak_mb"] for r in group_results]
        summary["by_class"][cls] = {
            "sp_mean": float(np.mean(sps)),
            "sp_std": float(np.std(sps)),
            "tc_mean": float(np.mean(tcs)),
            "tc_std": float(np.std(tcs)),
            "time_mean": float(np.mean(times)),
            "time_std": float(np.std(times)),
            "mem_mean": float(np.mean(mems)),
            "n": len(group_results),
        }

    # Overall
    all_sp = [r["sp_score"] for r in results]
    all_tc = [r["tc_score"] for r in results]
    all_t = [r["time_s"] for r in results]
    summary["overall"] = {
        "sp_mean": float(np.mean(all_sp)),
        "sp_std": float(np.std(all_sp)),
        "tc_mean": float(np.mean(all_tc)),
        "tc_std": float(np.std(all_tc)),
        "time_mean": float(np.mean(all_t)),
        "n": len(results),
    }

    return summary


if __name__ == "__main__":
    # Test 1: Perfect alignment
    ref = ["ACGT--ACGT", "ACGT--ACGT", "--ACGTACGT"]
    pred = ["ACGT--ACGT", "ACGT--ACGT", "--ACGTACGT"]
    score = sp_score(pred, ref)
    assert score == 1.0, f"Perfect alignment should score 1.0, got {score}"

    # Test 2: Imperfect alignment
    pred_wrong = ["ACGTACACGT", "ACGT--ACGT", "--ACGTACGT"]
    score2 = sp_score(pred_wrong, ref)
    assert 0.0 <= score2 <= 1.0, f"Imperfect alignment should score between 0 and 1, got {score2}"
    print(f"sp_score tests passed: perfect={score:.3f}, imperfect={score2:.3f}")

    # Test 3: Sequence order mismatch
    ref3 = ["ACGT--ACGT", "--ACGTACGT", "ACGT--ACGT"]
    pred3_reordered = ["--ACGTACGT", "ACGT--ACGT", "ACGT--ACGT"]  # different order
    score3 = sp_score(pred3_reordered, ref3)
    assert score3 == 1.0, f"Reordered perfect alignment should score 1.0, got {score3}"
    print(f"Reordered SP: {score3:.3f}")

    # Test 4: TC score
    tc = tc_score(pred, ref)
    assert tc == 1.0, f"Perfect TC should be 1.0, got {tc}"
    print(f"TC score test passed: {tc:.3f}")

    # Internal SP
    si = sp_score_internal(ref)
    print(f"Internal SP: {si:.4f}")
    print("All scoring tests passed!")

    # Consensus
    from msa.progressive_msa import build_profile
    profile = build_profile(ref, "dna")
    cons = profile_consensus(profile, "dna")
    print(f"Consensus: '{cons}'")

    print("Smoke test passed!")
