# tests/test_speed_pairwise.py — Benchmark: full NW vs banded NW vs our method.
# Run: pytest tests/test_speed_pairwise.py -v -s
# Results saved to results/speed_pairwise.csv for thesis tables.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import aligner

DNA_ALPHABET = "ACGT"


def generate_pair(length: int, divergence: float,
                  seed: int = 42) -> tuple[str, str]:
    """Generate a pair of sequences with the given divergence (substitutions only)."""
    rng = np.random.default_rng(seed)
    seq1 = "".join(rng.choice(list(DNA_ALPHABET), length))
    seq2 = list(seq1)
    n_mut = int(length * divergence)
    positions = rng.choice(length, min(n_mut, length), replace=False)
    for p in positions:
        choices = [c for c in DNA_ALPHABET if c != seq2[p]]
        seq2[p] = rng.choice(choices)
    return seq1, "".join(seq2)


def time_function(fn, n_runs: int = 5) -> float:
    """Measure median execution time of a function."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


@pytest.mark.parametrize("length,divergence,true_hw", [
    (300, 0.05, 8),
    (500, 0.10, 20),
    (1000, 0.15, 50),
    (2000, 0.20, 120),
    (5000, 0.10, 80),
])
def test_pairwise_speedup(length: int, divergence: float,
                          true_hw: int) -> None:
    """Compare Full NW vs Banded+FR+SIMD+Hirschberg."""
    seq1, seq2 = generate_pair(length, divergence)

    # Baseline: full NW
    t_full = time_function(lambda: aligner.full_nw_align(seq1, seq2))

    # Our method: banded with exact band (simulating perfect neural prediction)
    t_banded = time_function(
        lambda: aligner.align_with_doubling(seq1, seq2, 0, true_hw)
    )

    # Banded with overestimated band (simulating prediction with margin)
    t_wide = time_function(
        lambda: aligner.align_with_doubling(seq1, seq2, 0, true_hw * 2)
    )

    speedup_exact = t_full / max(t_banded, 1e-9)
    speedup_wide = t_full / max(t_wide, 1e-9)

    print(f"\nlen={length:5d}, div={divergence:.0%}: "
          f"full={t_full * 1000:.1f}ms, "
          f"banded(exact)={t_banded * 1000:.1f}ms ({speedup_exact:.1f}x), "
          f"banded(wide)={t_wide * 1000:.1f}ms ({speedup_wide:.1f}x)")

    assert speedup_exact > 1.5, f"Speedup too small: {speedup_exact:.1f}x"


def test_save_results() -> None:
    """Save benchmark results to CSV for the thesis."""
    Path("results").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    configs = [
        (300, 0.05, 8, "short_similar"),
        (500, 0.10, 20, "medium_low"),
        (1000, 0.15, 50, "medium"),
        (2000, 0.20, 120, "long_high"),
        (5000, 0.10, 80, "verylong_low"),
    ]

    for length, div, hw, label in configs:
        seq1, seq2 = generate_pair(length, div)

        t_full = time_function(lambda: aligner.full_nw_align(seq1, seq2))
        t_exact = time_function(
            lambda: aligner.align_with_doubling(seq1, seq2, 0, hw))
        t_wide = time_function(
            lambda: aligner.align_with_doubling(seq1, seq2, 0, hw * 2))

        rows.append({
            "label": label,
            "length": length,
            "divergence": div,
            "true_hw": hw,
            "t_full_ms": round(t_full * 1000, 2),
            "t_exact_ms": round(t_exact * 1000, 2),
            "t_wide_ms": round(t_wide * 1000, 2),
            "speedup_exact": round(t_full / max(t_exact, 1e-9), 1),
            "speedup_wide": round(t_full / max(t_wide, 1e-9), 1),
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/speed_pairwise.csv", index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    print("Smoke test: test_speed_pairwise.py")
    seq1, seq2 = generate_pair(300, 0.1)
    t = time_function(lambda: aligner.full_nw_align(seq1, seq2))
    print(f"Full NW 300bp: {t * 1000:.1f}ms")
    t = time_function(lambda: aligner.align_with_doubling(seq1, seq2, 0, 10))
    print(f"Banded 300bp: {t * 1000:.1f}ms")
    print("Smoke test passed!")
