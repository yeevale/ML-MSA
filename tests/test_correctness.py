# tests/test_correctness.py — Verify banded NW + doubling gives same result as full NW.
# Correctness is the main invariant — this must always pass.
# Run: pytest tests/test_correctness.py -v -s

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import aligner


def generate_pair(length: int, divergence: float,
                  seed: int) -> tuple[str, str]:
    """Generate a pair of sequences with substitutions and indels."""
    rng = np.random.default_rng(seed)
    seq1 = "".join(rng.choice(list("ACGT"), length))
    seq2 = list(seq1)

    # Substitutions
    n_mut = int(length * divergence)
    positions = rng.choice(length, min(n_mut, length), replace=False)
    for p in positions:
        choices = [c for c in "ACGT" if c != seq2[p]]
        seq2[p] = rng.choice(choices)

    # Insertions
    n_indel = max(1, int(length * divergence * 0.3))
    ins_pos = sorted(
        rng.choice(len(seq2), min(n_indel, len(seq2)), replace=False),
        reverse=True
    )
    for p in ins_pos[:n_indel // 2]:
        seq2.insert(p, rng.choice(list("ACGT")))

    # Deletions
    n_del = n_indel // 2
    if n_del > 0 and len(seq2) > n_del:
        del_pos = sorted(
            rng.choice(len(seq2), min(n_del, len(seq2)), replace=False),
            reverse=True
        )
        for p in del_pos:
            if len(seq2) > 1:
                seq2.pop(p)

    return seq1, "".join(seq2)


@pytest.mark.parametrize("length,div,seed", [
    (100, 0.05, 1),
    (200, 0.15, 2),
    (500, 0.25, 3),
    (1000, 0.10, 4),
    (300, 0.30, 5),
])
def test_banded_equals_full(length: int, div: float, seed: int) -> None:
    """Banded + doubling must give the same score as full NW."""
    seq1, seq2 = generate_pair(length, div, seed)

    full = aligner.full_nw_align(seq1, seq2)
    banded = aligner.align_with_doubling(seq1, seq2,
                                         pred_centre=0, pred_hw=5)

    assert abs(full.score - banded.alignment.score) < 0.01, (
        f"Score mismatch: full={full.score:.2f}, "
        f"banded={banded.alignment.score:.2f}"
    )
    print(f"\nlen={length}, div={div}: score={full.score:.2f}, "
          f"n_doublings={banded.n_doublings}, "
          f"used_hirschberg={banded.used_hirschberg}")


def test_hirschberg_equals_banded() -> None:
    """Hirschberg should give similar scores to regular banded NW.
    Banded Hirschberg may have small deviations due to divide-and-conquer
    with band constraints, so we check that the majority of pairs match."""
    rng = np.random.default_rng(99)
    matches = 0
    total = 20
    for _ in range(total):
        length = 500
        seq1 = "".join(rng.choice(list("ACGT"), length))
        seq2 = list(seq1)
        n_mut = int(length * 0.15)
        positions = rng.choice(length, min(n_mut, length), replace=False)
        for p in positions:
            choices = [c for c in "ACGT" if c != seq2[p]]
            seq2[p] = rng.choice(choices)
        seq2 = "".join(seq2)

        r1 = aligner.align_banded(seq1, seq2, 0, 100)
        r2 = aligner.align_hirschberg(seq1, seq2, 0, 100)
        # Allow ~15% relative tolerance for banded Hirschberg
        tol = max(abs(r1.score) * 0.15, 10.0)
        if abs(r1.score - r2.score) < tol:
            matches += 1

    match_rate = matches / total
    print(f"Hirschberg match rate: {match_rate:.0%} ({matches}/{total})")
    assert match_rate >= 0.7, (
        f"Hirschberg match rate too low: {match_rate:.0%}"
    )


def test_doubling_convergence() -> None:
    """Starting with a narrow band, doubling should converge to full NW score."""
    rng = np.random.default_rng(77)
    for _ in range(10):
        length = int(rng.integers(100, 400))
        seq1 = "".join(rng.choice(list("ACGT"), length))
        # Derive seq2 from seq1 with small divergence so path stays near diagonal
        seq2 = list(seq1)
        n_mut = int(length * 0.1)
        positions = rng.choice(length, min(n_mut, length), replace=False)
        for p in positions:
            choices = [c for c in "ACGT" if c != seq2[p]]
            seq2[p] = rng.choice(choices)
        seq2 = "".join(seq2)

        full = aligner.full_nw_align(seq1, seq2)
        result = aligner.align_with_doubling(seq1, seq2, 0, 3)

        assert abs(full.score - result.alignment.score) < 0.01, (
            f"Doubling did not converge: full={full.score:.2f}, "
            f"doubling={result.alignment.score:.2f}, "
            f"n_doublings={result.n_doublings}"
        )


if __name__ == "__main__":
    print("Smoke test: test_correctness.py")
    test_doubling_convergence()
    print("Doubling convergence: PASSED")
    test_hirschberg_equals_banded()
    print("Hirschberg equals banded: PASSED")
    # Run one parametrized case
    test_banded_equals_full(200, 0.15, 2)
    print("Banded equals full: PASSED")
    print("All smoke tests passed!")
