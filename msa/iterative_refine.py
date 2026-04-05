# msa/iterative_refine.py — MUSCLE-style iterative refinement of MSA.
# 3 passes: on each pass, re-align each sequence against the profile
# of all remaining sequences (sequence-vs-profile mode).
# Accept new alignment only if sp_score_internal improves.
# For N > 100: refine only random 30% of sequences per pass.

import numpy as np
import random

from msa.progressive_msa import build_profile, apply_gaps_to_seqs, _ungap
from model.evaluate import BandPredictorInference
import aligner

N_ITER      = 3
SAMPLE_FRAC = 0.30

DNA_ALPHABET = "ACGT-"
PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"


def sp_score_internal(msa: list[str], seq_type: str = "dna") -> float:
    """SP-score without reference — for iterative refinement.
    Sum of match scores over all pairwise non-gap positions.
    Normalised by number of pairs × alignment length."""
    n = len(msa)
    if n < 2:
        return 0.0
    L = len(msa[0])
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(L):
                ci = msa[i][k]
                cj = msa[j][k]
                if ci != '-' and cj != '-':
                    total += (1.0 if ci == cj else -1.0)
                    count += 1
    return total / max(count, 1)


def profile_consensus(profile: np.ndarray, seq_type: str = "dna") -> str:
    """Consensus string from profile: argmax at each column.
    If argmax is gap column → '-'."""
    alpha = DNA_ALPHABET if seq_type == "dna" else PROTEIN_ALPHABET
    result: list[str] = []
    for i in range(profile.shape[0]):
        idx = int(np.argmax(profile[i]))
        result.append(alpha[idx])
    return "".join(result)


def remove_and_compact(msa: list[str],
                       idx: int) -> tuple[list[str], list[int], str]:
    """Remove row idx from MSA. Delete empty columns (all gaps).
    Returns:
      compact_msa: list[str] without row idx, empty columns removed
      kept_cols:   list[int] — indices of kept columns
      removed_seq: str — row idx without gaps
    """
    removed = msa[idx]
    removed_seq = _ungap(removed)

    rest = msa[:idx] + msa[idx + 1:]
    L = len(rest[0]) if rest else 0

    # Find non-empty columns (at least one non-gap)
    kept_cols: list[int] = []
    for k in range(L):
        if any(row[k] != '-' for row in rest):
            kept_cols.append(k)

    compact_msa = ["".join(row[k] for k in kept_cols) for row in rest]
    return compact_msa, kept_cols, removed_seq


def reinsert_sequence(compact_msa: list[str],
                      aligned_seq: str,
                      aligned_profile_repr: str,
                      kept_cols: list[int],
                      original_len: int) -> list[str]:
    """Reinsert re-aligned sequence back into MSA.
    aligned_seq: from alignment result (with gaps)
    aligned_profile_repr: consensus of rest (with gaps) from same alignment
    kept_cols: indices of columns that were kept from original MSA
    original_len: length of original MSA columns
    """
    # The aligned_profile_repr gives us the gap pattern for compact_msa
    # and aligned_seq gives us the gap pattern for the reinserted sequence
    # Both should have the same length after pairwise alignment

    aln_len = len(aligned_seq)
    result_len = aln_len  # new MSA column count

    # Expand compact_msa according to aligned_profile_repr pattern
    new_rest: list[str] = []
    for row in compact_msa:
        new_row: list[str] = []
        pos = 0
        for ch in aligned_profile_repr:
            if ch == '-':
                new_row.append('-')
            else:
                if pos < len(row):
                    new_row.append(row[pos])
                else:
                    new_row.append('-')
                pos += 1
        new_rest.append("".join(new_row))

    # Build final MSA: rest + reinserted seq
    new_msa = new_rest + [aligned_seq]
    return new_msa


def iterative_refine(msa: list[str],
                     sequences: list[str],
                     predictor: BandPredictorInference,
                     seq_type: str = "dna",
                     n_iter: int = N_ITER) -> list[str]:
    """Iterative refinement of MSA.

    For each pass (n_iter times):
      If N > 100: select random SAMPLE_FRAC*N indices
      For each selected idx:
        1. Remove sequence, compact MSA
        2. Build profile of rest
        3. Predict band parameters
        4. Align sequence vs profile consensus
        5. Reinsert; accept if sp_score_internal improves
    """
    n = len(msa)
    if n <= 2:
        return msa

    current_msa = [s for s in msa]  # copy
    current_score = sp_score_internal(current_msa, seq_type)

    for iteration in range(n_iter):
        if n > 100:
            indices = random.sample(range(n), max(1, int(n * SAMPLE_FRAC)))
        else:
            indices = list(range(n))

        random.shuffle(indices)
        improved = 0

        for idx in indices:
            compact_msa, kept_cols, raw_seq = remove_and_compact(current_msa, idx)

            if not compact_msa or not raw_seq:
                continue

            profile_rest = build_profile(compact_msa, seq_type)
            consensus = profile_consensus(profile_rest, seq_type)
            consensus_nogap = _ungap(consensus)

            if not consensus_nogap or not raw_seq:
                continue

            # Predict band and align
            cd, hw = predictor.predict_single(raw_seq, profile_rest, seq_type)

            try:
                r = aligner.align_with_doubling(
                    raw_seq, consensus_nogap, cd, hw)
                a1 = r.alignment.aligned_seq1
                a2 = r.alignment.aligned_seq2
            except Exception:
                continue

            # Reinsert
            new_msa = reinsert_sequence(
                compact_msa, a1, a2, kept_cols, len(current_msa[0]))

            # Check score improvement
            new_score = sp_score_internal(new_msa, seq_type)
            if new_score > current_score:
                current_msa = new_msa
                current_score = new_score
                improved += 1

        if improved == 0:
            break  # no improvement this pass

    return current_msa


if __name__ == "__main__":
    # Smoke test — test helper functions only (no C++ aligner needed)

    msa = ["ACGT", "AC-T", "A-GT"]
    compact, kept, removed = remove_and_compact(msa, 1)
    print(f"Remove idx=1: compact={compact}, kept={kept}, removed='{removed}'")
    assert removed == "ACT"

    profile = build_profile(compact, "dna")
    cons = profile_consensus(profile, "dna")
    print(f"Consensus of rest: '{cons}'")

    score = sp_score_internal(msa, "dna")
    print(f"SP score internal: {score:.4f}")

    print("Smoke test passed!")
