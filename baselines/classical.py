# baselines/classical.py — Wrappers around external MSA tools via subprocess.
# All functions accept list[str] sequences and return list[str] aligned sequences.
# Binaries must be in PATH: mafft, muscle, clustalw2.
# Uses tempfile for temporary FASTA I/O.

import subprocess
import tempfile
import os
from pathlib import Path


def _seqs_to_fasta(sequences: list[str],
                   ids: list[str] | None = None) -> str:
    """Format sequences as FASTA string."""
    lines: list[str] = []
    for i, seq in enumerate(sequences):
        header = ids[i] if ids and i < len(ids) else f"seq{i}"
        lines.append(f">{header}")
        lines.append(seq)
    return "\n".join(lines) + "\n"


def _parse_fasta_alignment(text: str) -> list[str]:
    """Parse FASTA alignment (with gaps), return list[str] in input order."""
    result: list[str] = []
    current: list[str] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if current:
                result.append("".join(current))
            current = []
        else:
            current.append(line)
    if current:
        result.append("".join(current))
    return result


def run_mafft(sequences: list[str],
              ids: list[str] | None = None,
              extra_args: list[str] | None = None) -> list[str]:
    """Run MAFFT: mafft --auto --quiet [extra_args] input.fasta
    Parse stdout as FASTA. Raises RuntimeError if returncode != 0."""
    fasta = _seqs_to_fasta(sequences, ids)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta",
                                     delete=False) as f:
        f.write(fasta)
        input_path = f.name

    try:
        cmd = ["mafft", "--auto", "--quiet"]
        if extra_args:
            cmd.extend(extra_args)
        cmd.append(input_path)

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            raise RuntimeError(
                f"MAFFT failed (rc={result.returncode}): {result.stderr[:500]}")

        return _parse_fasta_alignment(result.stdout)
    finally:
        os.unlink(input_path)


def run_muscle(sequences: list[str],
               ids: list[str] | None = None) -> list[str]:
    """Run MUSCLE: muscle -align input.fasta -output output.fasta"""
    fasta = _seqs_to_fasta(sequences, ids)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta",
                                     delete=False) as f:
        f.write(fasta)
        input_path = f.name

    output_path = input_path + ".out"

    try:
        # Try MUSCLE v5 syntax first
        result = subprocess.run(
            ["muscle", "-align", input_path, "-output", output_path],
            capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            # Fallback to MUSCLE v3 syntax
            result = subprocess.run(
                ["muscle", "-in", input_path, "-out", output_path, "-quiet"],
                capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            raise RuntimeError(
                f"MUSCLE failed (rc={result.returncode}): {result.stderr[:500]}")

        with open(output_path) as f:
            return _parse_fasta_alignment(f.read())
    finally:
        for p in [input_path, output_path]:
            if os.path.exists(p):
                os.unlink(p)


def run_clustalw(sequences: list[str],
                 ids: list[str] | None = None) -> list[str]:
    """Run ClustalW: try 'clustalw2' then 'clustalw'."""
    fasta = _seqs_to_fasta(sequences, ids)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta",
                                     delete=False) as f:
        f.write(fasta)
        input_path = f.name

    output_path = input_path + ".aln.fasta"

    try:
        for cmd_name in ["clustalw2", "clustalw"]:
            try:
                result = subprocess.run(
                    [cmd_name,
                     f"-INFILE={input_path}",
                     f"-OUTFILE={output_path}",
                     "-OUTPUT=FASTA",
                     "-QUIET"],
                    capture_output=True, text=True, timeout=600)

                if result.returncode == 0 and os.path.exists(output_path):
                    with open(output_path) as f:
                        return _parse_fasta_alignment(f.read())
            except FileNotFoundError:
                continue

        raise RuntimeError(
            "ClustalW not found. Tried 'clustalw2' and 'clustalw'.")
    finally:
        for p in [input_path, output_path]:
            if os.path.exists(p):
                os.unlink(p)
        # ClustalW also creates .dnd file
        dnd = input_path.rsplit(".", 1)[0] + ".dnd"
        if os.path.exists(dnd):
            os.unlink(dnd)


if __name__ == "__main__":
    # Smoke test — just test FASTA formatting/parsing (no external tools needed)
    seqs = ["ACGTACGT", "ACGAACGT", "ACGTACGA"]
    fasta = _seqs_to_fasta(seqs, ["s1", "s2", "s3"])
    print("Generated FASTA:")
    print(fasta)

    # Parse it back
    parsed = _parse_fasta_alignment(fasta)
    assert len(parsed) == 3
    assert parsed[0] == "ACGTACGT"
    print(f"Parsed {len(parsed)} sequences")

    # Test with gaps
    aligned_fasta = ">s1\nACGT-ACGT\n>s2\nACG-AACGT\n>s3\nACGTACG-A\n"
    aligned = _parse_fasta_alignment(aligned_fasta)
    assert len(aligned) == 3
    assert aligned[0] == "ACGT-ACGT"
    print(f"Parsed aligned: {aligned}")

    print("Smoke test passed!")
