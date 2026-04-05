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

    # Map filename prefix (BB11 / bb11) to RV class
    _PREFIX_TO_CLASS = {
        "bb11": "RV11", "bb12": "RV12", "bb20": "RV20",
        "bb30": "RV30", "bb40": "RV40", "bb50": "RV50",
    }

    @staticmethod
    def _stem_to_class(stem: str) -> str:
        """Derive RVxx class from filename stem like BB11001 or RV11_..."""
        s = stem.lower()
        for prefix, cls in BAliBASELoader._PREFIX_TO_CLASS.items():
            if s.startswith(prefix):
                return cls
        for cls in ["RV11", "RV12", "RV20", "RV30", "RV40", "RV50"]:
            if cls.lower() in s:
                return cls
        return "unknown"

    def _find_tfa_files(self, ref_classes: list[str] | None = None) -> list[Path]:
        """Find all .tfa files, optionally filtered by reference class."""
        if ref_classes is None:
            ref_classes = ["RV11", "RV12", "RV20", "RV30", "RV40", "RV50"]
        ref_classes_lower = {rc.lower() for rc in ref_classes}
        tfa_files: list[Path] = []

        # Layout 1: DATASET-BALiBASE/Unaligned sequences/*.tfa  (new layout)
        unaligned_dir = self.data_dir / "Unaligned sequences"
        if unaligned_dir.exists():
            for f in sorted(unaligned_dir.glob("*.tfa")):
                cls = self._stem_to_class(f.stem)
                if cls.lower() in ref_classes_lower or cls == "unknown":
                    tfa_files.append(f)

        # Layout 2: DATASET-BALiBASE/RV11/*.tfa  (classic layout)
        for rc in ref_classes:
            rc_dir = self.data_dir / rc
            if rc_dir.exists():
                tfa_files.extend(sorted(rc_dir.glob("*.tfa")))
            for sub in sorted(self.data_dir.glob(f"{rc}*")):
                if sub.is_dir():
                    tfa_files.extend(sorted(sub.glob("*.tfa")))

        return list(dict.fromkeys(tfa_files))  # deduplicate preserving order

    def _load_xml_reference(self, xml_path: Path) -> list[str] | None:
        """Parse BAliBASE XML alignment file. Returns list of aligned sequences."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            aligned: list[str] = []
            # Try all common BAliBASE XML element/attribute patterns
            for seq_el in root.iter():
                if seq_el.tag.lower() in ("seq", "sequence", "aligned", "aln"):
                    # Data may be in text, or child 'data'/'seq'/'aligned'
                    data_el = (seq_el.find("data") or seq_el.find("seq")
                               or seq_el.find("aligned"))
                    text = (data_el.text if data_el is not None
                            else seq_el.text) or ""
                    text = text.strip().replace(" ", "").replace("\n", "")
                    if text:
                        aligned.append(text)
            if aligned:
                return aligned
        except Exception:
            pass
        # Fallback: try reading as FASTA (some BAliBASE sets use .fasta/.aln)
        for ext in [".fasta", ".fa", ".aln", ".msf"]:
            alt = xml_path.with_suffix(ext)
            if alt.exists():
                try:
                    records = load_fasta_with_gaps(str(alt))
                    if records:
                        return [r[1] for r in records]
                except Exception:
                    pass
        return None

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
        sequences = [r[1].replace("-", "") for r in records]  # without gaps

        # For the DATASET-BALiBASE layout, .tfa files in "Unaligned sequences/"
        # contain unaligned sequences. Try to find the reference alignment from
        # the corresponding .xml in "Aligned sequences/".
        reference = [r[1] for r in records]  # fallback: use tfa gaps if present
        aligned_dir = tfa_path.parent.parent / "Aligned sequences"
        if aligned_dir.exists():
            xml_path = aligned_dir / (tfa_path.stem + ".xml")
            if xml_path.exists():
                xml_ref = self._load_xml_reference(xml_path)
                if xml_ref and len(xml_ref) == len(sequences):
                    reference = xml_ref
            else:
                # Try other extensions
                for ext in [".fasta", ".fa", ".aln", ".tfa"]:
                    alt = aligned_dir / (tfa_path.stem + ext)
                    if alt.exists():
                        try:
                            recs = load_fasta_with_gaps(str(alt))
                            if recs and len(recs) == len(sequences):
                                reference = [r[1] for r in recs]
                        except Exception:
                            pass
                        break

        # Validate: reference sequences must all have the same length
        if reference:
            ref_lens = set(len(s) for s in reference)
            if len(ref_lens) > 1:
                # Unaligned reference — unusable for scoring
                reference = None

        # Filter by constraints
        if len(sequences) > self.max_num_seqs:
            return None
        if any(len(s) > self.max_seq_len for s in sequences):
            return None

        # Determine ref_class from path or filename
        ref_class = self._stem_to_class(tfa_path.stem)
        if ref_class == "unknown":
            for part in tfa_path.parts:
                for rc in ["RV11", "RV12", "RV20", "RV30", "RV40", "RV50"]:
                    if rc in part.upper():
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
