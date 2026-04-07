# data/loaders.py — FASTA file loaders.

from pathlib import Path


def load_fasta(path: str) -> list[tuple[str, str]]:
    """Parse .fasta/.tfa file. Returns list[(header, sequence_without_gaps)]."""
    records: list[tuple[str, str]] = []
    header = ""
    seq_parts: list[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if header:
                    records.append((header, "".join(seq_parts).replace("-", "")))
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line.strip())
    if header:
        records.append((header, "".join(seq_parts).replace("-", "")))
    return records


def load_fasta_with_gaps(path: str) -> list[tuple[str, str]]:
    """Parse .fasta/.tfa file preserving gaps. Returns list[(header, aligned_seq)]."""
    records: list[tuple[str, str]] = []
    header = ""
    seq_parts: list[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if header:
                    records.append((header, "".join(seq_parts)))
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line.strip())
    if header:
        records.append((header, "".join(seq_parts)))
    return records


if __name__ == "__main__":
    import sys

    # Smoke test with a fake FASTA
    import tempfile, os

    fasta_content = ">seq1\nACGT-ACGT\n>seq2\nACG-TACGT\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tfa", delete=False) as f:
        f.write(fasta_content)
        tmp_path = f.name

    try:
        records = load_fasta(tmp_path)
        assert len(records) == 2
        assert records[0] == ("seq1", "ACGTACGT")
        assert records[1] == ("seq2", "ACGTACGT")

        records_gaps = load_fasta_with_gaps(tmp_path)
        assert records_gaps[0] == ("seq1", "ACGT-ACGT")
        assert records_gaps[1] == ("seq2", "ACG-TACGT")

        print("load_fasta: OK")
        print("load_fasta_with_gaps: OK")
        print("All smoke tests passed!")
    finally:
        os.unlink(tmp_path)
