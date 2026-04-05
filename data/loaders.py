# data/loaders.py — BAliBASE 3.0 loader and synthetic data loaders.
# BAliBASE structure: folders RV11..RV50, each contains .tfa files.
# .tfa = FASTA format with gaps (reference alignment).
# Split: RV11+RV12+RV20+RV30 → train, RV40 → val, RV50 → test.

from pathlib import Path
from typing import Optional
from tqdm import tqdm


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


class BAliBASELoader:
    """Loader for BAliBASE 3.0 benchmark dataset."""

    TRAIN_CLASSES = ["RV11", "RV12", "RV20", "RV30"]
    VAL_CLASSES = ["RV40"]
    TEST_CLASSES = ["RV50"]

    def __init__(self, data_dir: str, max_seq_len: int = 2000,
                 max_num_seqs: int = 50):
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.max_num_seqs = max_num_seqs

    def _find_tfa_files(self, ref_classes: list[str] | None = None) -> list[Path]:
        """Find all .tfa files, optionally filtered by reference class."""
        tfa_files: list[Path] = []
        if ref_classes is None:
            ref_classes = ["RV11", "RV12", "RV20", "RV30", "RV40", "RV50"]
        for rc in ref_classes:
            rc_dir = self.data_dir / rc
            if rc_dir.exists():
                tfa_files.extend(sorted(rc_dir.glob("*.tfa")))
            # Also try bb* subdirectories (some BAliBASE layouts)
            for sub in sorted(self.data_dir.glob(f"{rc}*")):
                if sub.is_dir():
                    tfa_files.extend(sorted(sub.glob("*.tfa")))
        return list(dict.fromkeys(tfa_files))  # deduplicate preserving order

    def load_group(self, tfa_path: Path) -> dict | None:
        """Load one alignment group from a .tfa file.

        Returns dict with keys:
            group_id, ref_class, sequences, seq_ids, reference
        or None if the group is filtered out.
        """
        records = load_fasta_with_gaps(str(tfa_path))
        if not records:
            return None

        seq_ids = [r[0] for r in records]
        reference = [r[1] for r in records]  # with gaps
        sequences = [r[1].replace("-", "") for r in records]  # without gaps

        # Filter by constraints
        if len(sequences) > self.max_num_seqs:
            return None
        if any(len(s) > self.max_seq_len for s in sequences):
            return None

        # Determine ref_class from path
        ref_class = "unknown"
        for part in tfa_path.parts:
            for rc in ["RV11", "RV12", "RV20", "RV30", "RV40", "RV50"]:
                if rc in part:
                    ref_class = rc
                    break

        group_id = f"{ref_class}/{tfa_path.stem}"

        return {
            "group_id": group_id,
            "ref_class": ref_class,
            "sequences": sequences,
            "seq_ids": seq_ids,
            "reference": reference,
        }

    def load_all(self, ref_classes: list[str] | None = None) -> list[dict]:
        """Load all groups, optionally filtered by ref_classes."""
        tfa_files = self._find_tfa_files(ref_classes)
        groups: list[dict] = []
        for tfa_path in tqdm(tfa_files, desc="Loading BAliBASE"):
            g = self.load_group(tfa_path)
            if g is not None:
                groups.append(g)
        return groups

    def train_val_test_split(self) -> tuple[list[dict], list[dict], list[dict]]:
        """RV11+RV12+RV20+RV30 → train, RV40 → val, RV50 → test."""
        train = self.load_all(self.TRAIN_CLASSES)
        val = self.load_all(self.VAL_CLASSES)
        test = self.load_all(self.TEST_CLASSES)
        return train, val, test


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
