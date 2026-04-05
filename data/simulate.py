# data/simulate.py — Generate synthetic training data for the band predictor.
# Outputs .parquet files with columns:
#   seq1, seq2, centre_diag, true_half_width, divergence, seq_type
# Stratified equally across 3 divergence groups: low, medium, high.
# Parallelized via multiprocessing.Pool.

import numpy as np
import argparse
import sys
import os
from dataclasses import dataclass, asdict
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
DNA_ALPHABET = "ACGT"
MARGIN = 3


@dataclass
class AlignmentSample:
    seq1: str
    seq2: str
    centre_diag: int
    true_half_width: int
    divergence: float
    seq_type: str


def mutate_with_alignment(
        seq: str, p_sub: float, p_ins: float, p_del: float,
        alphabet: str, rng: np.random.Generator
) -> tuple[str, list[tuple[int, int]]]:
    """Mutate seq and return (mutated_seq, alignment_path).

    alignment_path contains (i, j) pairs where seq1[i] corresponds to seq2[j].
    Computed in O(n) from the known mutation operations — no NW needed.
    Insertions shift j without advancing i; deletions advance i without advancing j.
    """
    result: list[str] = []
    path: list[tuple[int, int]] = []
    j = 0  # current position in seq2 being built
    mismatches = 0

    for i, ch in enumerate(seq):
        # Deletion: seq1[i] is dropped, j does not advance
        if rng.random() < p_del:
            continue

        # Insertions before this match/substitution
        while rng.random() < p_ins:
            result.append(rng.choice(list(alphabet)))
            j += 1  # inserted position in seq2 — not paired with seq1

        # Substitution
        if rng.random() < p_sub:
            choices = [c for c in alphabet if c != ch]
            ch = rng.choice(choices) if choices else ch
            mismatches += 1

        result.append(ch)
        path.append((i, j))
        j += 1

    seq2 = "".join(result) if result else seq[:]
    return seq2, path


def mutate_sequence(seq: str, p_sub: float, p_ins: float, p_del: float,
                   alphabet: str, rng: np.random.Generator) -> str:
    """Thin wrapper kept for backward compatibility."""
    seq2, _ = mutate_with_alignment(seq, p_sub, p_ins, p_del, alphabet, rng)
    return seq2


def compute_band_params(path: list[tuple[int, int]]) -> tuple[int, int]:
    """Compute (centre_diag, true_half_width) from traceback path."""
    if not path:
        return 0, MARGIN

    diagonals = [i - j for i, j in path]
    centre_diag = int(np.median(diagonals))
    true_half_width = max(abs(d - centre_diag) for d in diagonals) + MARGIN
    return centre_diag, true_half_width


def sample_mutation_params(divergence_group: str,
                           rng: np.random.Generator) -> tuple[float, float, float]:
    """Sample (p_sub, p_ins, p_del) for a given divergence group."""
    if divergence_group == "low":
        p_sub = rng.uniform(0.01, 0.08)
        p_ins = rng.uniform(0.0, 0.02)
        p_del = rng.uniform(0.0, 0.02)
    elif divergence_group == "medium":
        p_sub = rng.uniform(0.08, 0.20)
        p_ins = rng.uniform(0.01, 0.05)
        p_del = rng.uniform(0.01, 0.05)
    else:  # high
        p_sub = rng.uniform(0.20, 0.40)
        p_ins = rng.uniform(0.03, 0.10)
        p_del = rng.uniform(0.03, 0.10)
    return p_sub, p_ins, p_del


def simulate_one(args: tuple) -> dict | None:
    """Generate one training sample in O(n) — no NW call.
    args = (length, p_sub, p_ins, p_del, seq_type, seed)

    Band parameters are derived directly from the known mutation path:
    - centre_diag = median diagonal of aligned positions
    - true_half_width = max deviation from centre + MARGIN
    This is exact for the alignment we generated, avoiding O(n²) full NW.
    """
    length, p_sub, p_ins, p_del, seq_type, seed = args
    try:
        rng = np.random.default_rng(seed)
        alphabet = DNA_ALPHABET if seq_type == "dna" else PROTEIN_ALPHABET

        seq1 = "".join(rng.choice(list(alphabet), length))
        seq2, path = mutate_with_alignment(seq1, p_sub, p_ins, p_del, alphabet, rng)

        if len(seq2) < 5 or not path:
            return None

        centre_diag, true_half_width = compute_band_params(path)

        # Divergence = fraction of paired positions that are substitutions
        substitutions = sum(
            1 for (i, j) in path if seq1[i] != seq2[j]
        )
        divergence = round(substitutions / max(len(path), 1), 4)

        sample = AlignmentSample(
            seq1=seq1,
            seq2=seq2,
            centre_diag=centre_diag,
            true_half_width=true_half_width,
            divergence=divergence,
            seq_type=seq_type,
        )
        return asdict(sample)
    except Exception:
        return None


def generate_dataset(n_samples: int, output_path: str,
                     seq_type: str = "dna", n_workers: int = 8,
                     seed: int = 42) -> None:
    """Generate n_samples pairs and save to parquet."""
    rng = np.random.default_rng(seed)
    groups = ["low", "medium", "high"]
    n_per_group = n_samples // 3

    # Length ranges — capped at 500bp: longer seqs use anchor-based alignment
    # which is handled separately; 500bp is sufficient for band predictor training
    if seq_type == "dna":
        len_low, len_high = 50, 500
    else:
        len_low, len_high = 30, 300

    # Prepare all tasks
    tasks: list[tuple] = []
    for g_idx, group in enumerate(groups):
        for i in range(n_per_group):
            task_seed = seed + g_idx * n_per_group + i
            task_rng = np.random.default_rng(task_seed)
            length = int(task_rng.integers(len_low, len_high))
            p_sub, p_ins, p_del = sample_mutation_params(group, task_rng)
            tasks.append((length, p_sub, p_ins, p_del, seq_type, task_seed))

    # Process in batches of 10_000
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    batch_size = 10_000
    total_written = 0
    n_batches = (len(tasks) + batch_size - 1) // batch_size

    try:
        pool = Pool(n_workers) if n_workers > 1 else None
        for batch_idx, batch_start in enumerate(range(0, len(tasks), batch_size)):
            batch_tasks = tasks[batch_start:batch_start + batch_size]
            print(f"  Batch {batch_idx + 1}/{n_batches} "
                  f"({len(batch_tasks)} tasks)...", flush=True)

            if pool is not None:
                results = pool.map(simulate_one, batch_tasks)
            else:
                results = [simulate_one(t) for t in batch_tasks]

            # Filter None results
            valid = [r for r in results if r is not None]
            if not valid:
                print(f"  WARNING: batch {batch_idx + 1} produced 0 valid samples",
                      flush=True)
                continue

            table = pa.Table.from_pylist(valid)

            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)

            writer.write_table(table)
            total_written += len(valid)
            print(f"  Written {total_written} total", flush=True)

    finally:
        if writer is not None:
            writer.close()  # close first so parquet footer is written even if pool hangs
        if pool is not None:
            pool.close()
            pool.join()

    print(f"Generated {total_written} samples -> {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic alignment data")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of samples to generate")
    parser.add_argument("--seq_type", choices=["dna", "protein"], default="dna")
    parser.add_argument("--output", default="data/processed/train_dna.parquet")
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if len(sys.argv) == 1:
        # Smoke test: generate a small batch
        print("Running smoke test (10 samples, single worker)...")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test.parquet")
            generate_dataset(9, out, "dna", n_workers=1, seed=42)
            import pyarrow.parquet as pq
            df = pq.read_table(out).to_pandas()
            print(f"Generated {len(df)} rows")
            print(df[["centre_diag", "true_half_width", "divergence", "seq_type"]].head())
            assert len(df) > 0
            assert "seq1" in df.columns
            assert "centre_diag" in df.columns
            print("Smoke test passed!")
    else:
        generate_dataset(args.n_samples, args.output, args.seq_type,
                         args.n_workers, args.seed)
