# features/kmer.py — k-mer and minimizer-based scalar features for the band predictor.
# Produces a SCALAR_DIM=70 feature vector for a pair of sequences.
# All features normalized to [0,1] range.

import numpy as np
from collections import Counter, defaultdict
from typing import Literal

SeqType = Literal["dna", "protein"]

SCALAR_DIM = 70

# Alphabet helpers
DNA_CHARS = set("ACGT")
PROTEIN_CHARS = set("ACDEFGHIKLMNPQRSTVWY")
CHARGED_AA = set("DEKRH")


def _alphabet_size(seq_type: SeqType) -> int:
    return 4 if seq_type == "dna" else 20


def _kmer_index(kmer: str, alpha: str) -> int | None:
    """Convert k-mer to integer index, return None if invalid char."""
    idx = 0
    a_size = len(alpha)
    for ch in kmer:
        p = alpha.find(ch)
        if p < 0:
            return None
        idx = idx * a_size + p
    return idx


def kmer_freq(seq: str, k: int, seq_type: SeqType = "dna") -> np.ndarray:
    """Normalised k-mer frequency vector.
    DNA k=4 → length 256.  Protein k=3 → length 8000.
    K-mers containing ambiguous chars are skipped."""
    alpha = "ACGT" if seq_type == "dna" else "ACDEFGHIKLMNPQRSTVWY"
    total = len(alpha) ** k
    freq = np.zeros(total, dtype=np.float32)
    seq_upper = seq.upper()
    count = 0
    for i in range(len(seq_upper) - k + 1):
        kmer = seq_upper[i:i + k]
        idx = _kmer_index(kmer, alpha)
        if idx is not None:
            freq[idx] += 1.0
            count += 1
    if count > 0:
        freq /= count
    return freq


def minimizers(seq: str, w: int, k: int) -> set[str]:
    """Standard minimizer algorithm (Roberts et al., 2004).
    For each window of size w pick the lexicographically smallest k-mer.
    Returns set of unique minimizer strings."""
    seq_upper = seq.upper()
    n = len(seq_upper)
    if n < k:
        return set()

    kmers = [seq_upper[i:i + k] for i in range(n - k + 1)]
    result: set[str] = set()
    for start in range(len(kmers) - w + 1):
        window = kmers[start:start + w]
        result.add(min(window))
    return result


def _minimizer_positions(seq: str, w: int, k: int) -> dict[str, list[int]]:
    """Return dict {kmer_string: [positions]}."""
    seq_upper = seq.upper()
    n = len(seq_upper)
    if n < k:
        return {}
    kmers = [seq_upper[i:i + k] for i in range(n - k + 1)]
    result: dict[str, list[int]] = defaultdict(list)
    for start in range(len(kmers) - w + 1):
        window = kmers[start:start + w]
        best = min(window)
        best_idx = start + window.index(best)
        result[best].append(best_idx)
    # deduplicate positions
    return {k: sorted(set(v)) for k, v in result.items()}


def _gc_content(seq: str) -> float:
    seq = seq.upper()
    gc = sum(1 for c in seq if c in "GC")
    return gc / max(len(seq), 1)


def _charged_fraction(seq: str) -> float:
    seq = seq.upper()
    charged = sum(1 for c in seq if c in CHARGED_AA)
    return charged / max(len(seq), 1)


def _jaccard(s1: set, s2: set) -> float:
    if not s1 and not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _l1_norm(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.sum(np.abs(v1 - v2)) / 2.0)


def _entropy(freq: np.ndarray) -> float:
    """Shannon entropy in bits."""
    p = freq[freq > 0]
    return float(-np.sum(p * np.log2(p + 1e-9)))


def kmer_features(seq1: str, seq2: str,
                  seq_type: SeqType = "dna") -> np.ndarray:
    """Compute ~17 scalar features padded to SCALAR_DIM=70. dtype=float32."""
    feat: list[float] = []
    len1, len2 = len(seq1), len(seq2)
    max_len = max(len1, len2, 1)

    # Basic length features (4)
    feat.append(len1 / 10000.0)  # normalise
    feat.append(len2 / 10000.0)
    feat.append(len1 / max(len2, 1))
    feat.append(abs(len1 - len2) / max_len)

    # Composition features (2)
    if seq_type == "dna":
        feat.append(_gc_content(seq1))
        feat.append(_gc_content(seq2))
    else:
        feat.append(_charged_fraction(seq1))
        feat.append(_charged_fraction(seq2))

    # K-mer statistics for two k values (6)
    if seq_type == "dna":
        ks = [3, 4]
    else:
        ks = [2, 3]

    for k in ks:
        f1 = kmer_freq(seq1, k, seq_type)
        f2 = kmer_freq(seq2, k, seq_type)
        s1 = set(i for i, v in enumerate(f1) if v > 0)
        s2 = set(i for i, v in enumerate(f2) if v > 0)
        feat.append(_jaccard(s1, s2))
        feat.append(_cosine(f1, f2))
        feat.append(_l1_norm(f1, f2))

    # Minimizer features w=5, k=8 (3)
    min1 = minimizers(seq1, w=5, k=8)
    min2 = minimizers(seq2, w=5, k=8)
    feat.append(_jaccard(min1, min2))
    feat.append(len(min1) / max(len1, 1))
    feat.append(len(min2) / max(len2, 1))

    # Entropy features (2)
    k_entropy = 4 if seq_type == "dna" else 3
    feat.append(_entropy(kmer_freq(seq1, k_entropy, seq_type)))
    feat.append(_entropy(kmer_freq(seq2, k_entropy, seq_type)))

    # Pad to SCALAR_DIM
    result = np.zeros(SCALAR_DIM, dtype=np.float32)
    result[:len(feat)] = feat
    return result


if __name__ == "__main__":
    # Smoke test
    s1 = "ACGTACGTACGTACGTACGTAAACCCGGGTTT"
    s2 = "ACGTACATACGTACCTACGTAAACCCGGGTTT"
    v = kmer_features(s1, s2, "dna")
    assert v.shape == (SCALAR_DIM,), f"Bad shape: {v.shape}"
    assert v.dtype == np.float32
    print(f"kmer_features shape={v.shape}, nonzero={np.count_nonzero(v)}")
    print(f"First 17 features: {v[:17]}")

    mins = minimizers(s1, w=5, k=8)
    print(f"Minimizers count: {len(mins)}")

    # Protein test
    ps1 = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK"
    ps2 = "MVLSGEDKSNIKAAWGKIGGHGAEYGAEALERMFLSFPTTK"
    vp = kmer_features(ps1, ps2, "protein")
    assert vp.shape == (SCALAR_DIM,)
    print(f"Protein features nonzero={np.count_nonzero(vp)}")
    print("Smoke test passed!")
