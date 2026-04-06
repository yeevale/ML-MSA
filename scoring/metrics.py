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


def _match_and_reorder(predicted_msa: list[str],
                       reference_msa: list[str]) -> tuple[list[str], dict] | None:
    """Match predicted to reference sequences and return (reordered_pred, ref_to_pred_map).
    Strategy: first match by exact gapped string, then by ungapped content.
    Returns None if matching fails."""
    n = len(reference_msa)
    pred_upper = [s.upper() for s in predicted_msa]
    ref_upper  = [s.upper() for s in reference_msa]

    ref_to_pred = {}
    used_pred = set()

    # Pass 1: match by exact gapped string
    for ri in range(n):
        for pi in range(len(pred_upper)):
            if pi not in used_pred and ref_upper[ri] == pred_upper[pi]:
                ref_to_pred[ri] = pi
                used_pred.add(pi)
                break

    # Pass 2: match remaining by ungapped content
    pred_ungapped = [''.join(c for c in s if c != '-') for s in pred_upper]
    ref_ungapped  = [''.join(c for c in s if c != '-') for s in ref_upper]

    for ri in range(n):
        if ri in ref_to_pred:
            continue
        for pi in range(len(pred_ungapped)):
            if pi not in used_pred and ref_ungapped[ri] == pred_ungapped[pi]:
                ref_to_pred[ri] = pi
                used_pred.add(pi)
                break

    if len(ref_to_pred) < n:
        return None

    reordered = [predicted_msa[ref_to_pred[ri]] for ri in range(n)]
    # After reordering, mapping is identity
    identity_map = {i: i for i in range(n)}
    return reordered, identity_map


def sp_score(predicted_msa: list[str],
             reference_msa: list[str]) -> float:
    """Sum-of-Pairs score (BAliBASE standard).
    Handles sequence order mismatch by matching ungapped content."""
    try:
        # Fix 1: uppercase everything
        predicted_msa = [s.upper() for s in predicted_msa]
        reference_msa = [s.upper() for s in reference_msa]

        n = len(reference_msa)
        if n < 2:
            return 1.0

        # --- DIAGNOSTIC OUTPUT (Problem 1) ---
        pred_ungapped = [''.join(c for c in s if c != '-') for s in predicted_msa]
        ref_ungapped  = [''.join(c for c in s if c != '-') for s in reference_msa]

        if len(predicted_msa) != len(reference_msa):
            print(f"[SP DIAG] seq count mismatch: pred={len(predicted_msa)}, ref={len(reference_msa)}")

        # Check how many ref sequences can be found in pred by ungapped content
        matched_count = 0
        unmatched_refs = []
        used = set()
        for ri, rseq in enumerate(ref_ungapped):
            found = False
            for pi, pseq in enumerate(pred_ungapped):
                if pi not in used and rseq == pseq:
                    used.add(pi)
                    found = True
                    break
            if found:
                matched_count += 1
            else:
                unmatched_refs.append(ri)

        if matched_count < n:
            print(f"[SP DIAG] Only {matched_count}/{n} ref seqs matched in pred by ungapped content")
            for ri in unmatched_refs[:3]:
                print(f"  ref[{ri}] ungapped len={len(ref_ungapped[ri])}, first 60: {ref_ungapped[ri][:60]}")
                # Find closest pred seq by length
                closest = min(range(len(pred_ungapped)),
                              key=lambda pi: abs(len(pred_ungapped[pi]) - len(ref_ungapped[ri])))
                print(f"  closest pred[{closest}] ungapped len={len(pred_ungapped[closest])}, first 60: {pred_ungapped[closest][:60]}")
            print(f"  pred ungapped lengths: {sorted(set(len(s) for s in pred_ungapped))}")
            print(f"  ref  ungapped lengths: {sorted(set(len(s) for s in ref_ungapped))}")
        # --- END DIAGNOSTIC ---

        # Fix 2: match sequences (handles different counts and reordering)
        result = _match_and_reorder(predicted_msa, reference_msa)
        if result is None:
            return 0.0
        predicted_msa, ref_to_pred = result

        # Fix 4: build position maps for each sequence
        # pos_map[seq_idx][ungapped_pos] = gapped_pos in predicted
        def build_pos_map(msa: list[str]) -> list[dict]:
            maps = []
            for seq in msa:
                m = {}
                ungapped_pos = 0
                for gapped_pos, char in enumerate(seq):
                    if char != '-':
                        m[ungapped_pos] = gapped_pos
                        ungapped_pos += 1
                maps.append(m)
            return maps

        pred_pos_maps = build_pos_map(predicted_msa)
        ref_pos_maps  = build_pos_map(reference_msa)

        # Fix 5: compute SP-score using position maps
        correct = 0
        total   = 0

        n = len(reference_msa)
        for i in range(n):
            for j in range(i + 1, n):
                pi = ref_to_pred[i]
                pj = ref_to_pred[j]
                ref_seq_i = reference_msa[i]
                ref_seq_j = reference_msa[j]

                # Track ungapped position in ref sequences
                ungapped_j_counts = {}
                col_j = 0
                for k in range(len(ref_seq_j)):
                    if ref_seq_j[k] != '-':
                        ungapped_j_counts[k] = col_j
                        col_j += 1

                ungapped_i = 0
                for k in range(len(ref_seq_i)):
                    char_i = ref_seq_i[k]
                    char_j = ref_seq_j[k] if k < len(ref_seq_j) else '-'

                    if char_i != '-' and char_j != '-':
                        total += 1
                        # Find these residues in predicted alignment
                        ui = ungapped_i
                        uj = ungapped_j_counts.get(k, -1)
                        if uj == -1:
                            if char_i != '-':
                                ungapped_i += 1
                            continue

                        pred_col_i = pred_pos_maps[pi].get(ui, -1)
                        pred_col_j = pred_pos_maps[pj].get(uj, -1)

                        if (pred_col_i != -1 and pred_col_j != -1 and
                                pred_col_i == pred_col_j):
                            correct += 1

                    if char_i != '-':
                        ungapped_i += 1

        return correct / total if total > 0 else 0.0

    except Exception as e:
        print(f"[sp_score ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 0.0  # never return -1.0


def tc_score(predicted_msa: list[str],
             reference_msa: list[str]) -> float:
    """Total Column score.
    Handles sequence order mismatch by matching ungapped content."""
    try:
        # Uppercase everything
        predicted_msa = [s.upper() for s in predicted_msa]
        reference_msa = [s.upper() for s in reference_msa]

        n = len(reference_msa)
        if n < 2:
            return 1.0

        # Match sequences (handles different counts and reordering)
        result = _match_and_reorder(predicted_msa, reference_msa)
        if result is None:
            return 0.0
        predicted_msa, ref_to_pred = result

        # Build position maps: ungapped_pos -> gapped_pos
        def build_pos_map(msa: list[str]) -> list[dict]:
            maps = []
            for seq in msa:
                m = {}
                ungapped_pos = 0
                for gapped_pos, char in enumerate(seq):
                    if char != '-':
                        m[ungapped_pos] = gapped_pos
                        ungapped_pos += 1
                maps.append(m)
            return maps

        pred_pos_maps = build_pos_map(predicted_msa)
        ref_pos_maps  = build_pos_map(reference_msa)

        # For each reference column, check if ALL non-gap residues
        # end up in the same predicted column
        ref_len = len(reference_msa[0]) if reference_msa else 0
        if ref_len == 0:
            return 0.0

        # Validate ref alignment consistency
        if any(len(s) != ref_len for s in reference_msa):
            return 0.0

        matching = 0
        total = 0

        for k in range(ref_len):
            # Collect (ref_seq_idx, ungapped_pos) for non-gap chars in this column
            residues = []
            ungapped_counts = [0] * n
            for si in range(n):
                # Count ungapped positions up to column k for this sequence
                ug = sum(1 for c in reference_msa[si][:k] if c != '-')
                if reference_msa[si][k] != '-':
                    residues.append((si, ug))

            if len(residues) < 2:
                continue

            total += 1

            # Check if all these residues map to the same predicted column
            pred_cols = set()
            all_found = True
            for seq_idx, ug_pos in residues:
                pi = ref_to_pred[seq_idx]
                pred_col = pred_pos_maps[pi].get(ug_pos, -1)
                if pred_col == -1:
                    all_found = False
                    break
                pred_cols.add(pred_col)

            if all_found and len(pred_cols) == 1:
                matching += 1

        return matching / max(total, 1)

    except Exception as e:
        print(f"[tc_score ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 0.0


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
    # Test 1: perfect alignment
    ref  = ["ACGT--ACGT", "AC--GTACGT", "--ACGTACGT"]
    pred = ["ACGT--ACGT", "AC--GTACGT", "--ACGTACGT"]
    s = sp_score(pred, ref)
    assert s == 1.0, f"Perfect: expected 1.0 got {s}"
    print(f"Test 1 passed: perfect SP = {s:.3f}")

    # Test 2: wrong order — same sequences different order
    pred_reordered = ["--ACGTACGT", "ACGT--ACGT", "AC--GTACGT"]
    s2 = sp_score(pred_reordered, ref)
    assert s2 == 1.0, f"Reordered: expected 1.0 got {s2}"
    print(f"Test 2 passed: reordered SP = {s2:.3f}")

    # Test 3: imperfect alignment (same sequences, different gap placement)
    pred_wrong = ["ACGT-A-CGT", "AC--GTACGT", "--ACGTACGT"]
    s3 = sp_score(pred_wrong, ref)
    assert 0.0 < s3 < 1.0, f"Imperfect: expected 0<x<1 got {s3}"
    assert s3 != -1.0, "Must never return -1.0"
    print(f"Test 3 passed: imperfect SP = {s3:.3f}")

    print("All sp_score tests passed!")
