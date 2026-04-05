# features/profile_features.py — CENTRAL feature file for the neural network.
# Implements make_input() — unified interface that works for:
#   Mode 1: obj = str  (leaf sequences in guide tree)
#   Mode 2: obj = np.ndarray profile (internal nodes)
# Output tensors always (1,64,64) + (SCALAR_DIM,) — architecture is unchanged.
#
# PROFILE: np.ndarray shape (L, A) where
#   L = alignment length (columns), A = DNA_PROF_SIZE=5 or PROTEIN_PROF_SIZE=21
#   profile[i, a] = frequency of symbol a at position i, sum(profile[i]) = 1.0

import numpy as np
from pathlib import Path
from scipy.ndimage import zoom

from features.kmer import kmer_features, SCALAR_DIM
from features.dotplot import dotplot_tensor

# ---- Substitution matrices ----
DNA_SUBST = np.array([
    [ 1., -1., -1., -1.],  # A
    [-1.,  1., -1., -1.],  # C
    [-1., -1.,  1., -1.],  # G
    [-1., -1., -1.,  1.],  # T
], dtype=np.float32)

MATRIX_SIZE = 64

# Standard amino acid order for BLOSUM parsing
_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"

BLOSUM62: np.ndarray | None = None
_BLOSUM62_PATH = Path(__file__).resolve().parent.parent / "data" / "blosum62.txt"


def load_blosum62(path: str | Path | None = None) -> np.ndarray:
    """Load BLOSUM62 from a standard NCBI-format text file.
    Lines starting with '#' are comments.
    First non-comment line is the header (amino acid order).
    Remaining lines are matrix rows."""
    global BLOSUM62
    if path is None:
        path = _BLOSUM62_PATH
    path = Path(path)

    lines = [l.strip() for l in path.read_text().splitlines()
             if l.strip() and not l.startswith("#")]

    header = lines[0].split()
    n = len(header)
    mat = np.zeros((20, 20), dtype=np.float32)

    # Map header AA to our canonical order
    header_to_idx = {aa: i for i, aa in enumerate(header)}

    for line in lines[1:]:
        parts = line.split()
        row_aa = parts[0]
        if row_aa not in _AA_ORDER:
            continue
        row_i = _AA_ORDER.index(row_aa)
        values = [float(x) for x in parts[1:n + 1]]
        for col_j, col_aa in enumerate(header):
            if col_aa not in _AA_ORDER:
                continue
            mat[row_i, _AA_ORDER.index(col_aa)] = values[col_j]

    BLOSUM62 = mat
    return mat


def _ensure_blosum62() -> np.ndarray:
    """Load BLOSUM62 lazily."""
    global BLOSUM62
    if BLOSUM62 is None:
        if _BLOSUM62_PATH.exists():
            return load_blosum62()
        # Fallback: identity-like matrix
        BLOSUM62 = np.eye(20, dtype=np.float32)
    return BLOSUM62


def column_entropy(col: np.ndarray) -> float:
    """Shannon entropy of one profile column.
    0 = fully conserved, log2(A) = random."""
    p = col[col > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p + 1e-9)))


def profile_scalar_features(profile1: np.ndarray, profile2: np.ndarray,
                             seq_type: str = "dna") -> np.ndarray:
    """Scalar features for two profiles → shape (SCALAR_DIM,) float32.

    Features:
      L1, L2, L1/L2, |L1-L2|/max                — 4
      mean_entropy(p1), mean_entropy(p2)          — 2
      std_entropy(p1), std_entropy(p2)            — 2
      gap_fraction(p1), gap_fraction(p2)          — 2  (last col > 0.5)
      mean_profile_sim, max_profile_sim           — 2
    Pad to SCALAR_DIM=70."""
    feat: list[float] = []
    L1, L2 = profile1.shape[0], profile2.shape[0]
    max_L = max(L1, L2, 1)

    # Length features (4)
    feat.append(L1 / 10000.0)
    feat.append(L2 / 10000.0)
    feat.append(L1 / max(L2, 1))
    feat.append(abs(L1 - L2) / max_L)

    # Entropy features (4)
    ent1 = np.array([column_entropy(profile1[i]) for i in range(L1)])
    ent2 = np.array([column_entropy(profile2[i]) for i in range(L2)])
    feat.append(float(ent1.mean()) if len(ent1) > 0 else 0.0)
    feat.append(float(ent2.mean()) if len(ent2) > 0 else 0.0)
    feat.append(float(ent1.std()) if len(ent1) > 0 else 0.0)
    feat.append(float(ent2.std()) if len(ent2) > 0 else 0.0)

    # Gap fraction (2) — last column index is the gap column
    A1 = profile1.shape[1]
    A2 = profile2.shape[1]
    if A1 > 1:
        gap_frac1 = float(np.mean(profile1[:, -1] > 0.5))
    else:
        gap_frac1 = 0.0
    if A2 > 1:
        gap_frac2 = float(np.mean(profile2[:, -1] > 0.5))
    else:
        gap_frac2 = 0.0
    feat.append(gap_frac1)
    feat.append(gap_frac2)

    # Profile similarity via sampling (2)
    subst = DNA_SUBST if seq_type == "dna" else _ensure_blosum62()
    # Take A columns without gap column for similarity
    cols1 = min(A1, subst.shape[0])
    cols2 = min(A2, subst.shape[1])

    n_samples = min(100, L1 * L2)
    if n_samples > 0 and L1 > 0 and L2 > 0:
        rng = np.random.default_rng(0)
        ii = rng.integers(0, L1, size=n_samples)
        jj = rng.integers(0, L2, size=n_samples)
        sims = []
        for i_idx, j_idx in zip(ii, jj):
            p1 = profile1[i_idx, :cols1]
            p2 = profile2[j_idx, :cols2]
            s = float(p1 @ subst[:cols1, :cols2] @ p2)
            sims.append(s)
        feat.append(float(np.mean(sims)))
        feat.append(float(np.max(sims)))
    else:
        feat.append(0.0)
        feat.append(0.0)

    # Pad to SCALAR_DIM
    result = np.zeros(SCALAR_DIM, dtype=np.float32)
    result[:len(feat)] = feat
    return result


def profile_similarity_matrix(profile1: np.ndarray, profile2: np.ndarray,
                               subst: np.ndarray,
                               target_size: int = MATRIX_SIZE) -> np.ndarray:
    """Profile similarity matrix → shape (1, 64, 64) float32.

    sim[i,j] = profile1[i] @ subst @ profile2[j]
    Normalised to [0, 1], resized to target_size."""
    L1 = profile1.shape[0]
    L2 = profile2.shape[0]
    cols1 = min(profile1.shape[1], subst.shape[0])
    cols2 = min(profile2.shape[1], subst.shape[1])

    if L1 == 0 or L2 == 0:
        return np.zeros((1, target_size, target_size), dtype=np.float32)

    # sim = profile1[:, :cols1] @ subst[:cols1, :cols2] @ profile2[:, :cols2].T
    p1 = profile1[:, :cols1]  # (L1, cols1)
    p2 = profile2[:, :cols2]  # (L2, cols2)
    s = subst[:cols1, :cols2]  # (cols1, cols2)
    sim = np.einsum('ia,ab,jb->ij', p1, s, p2)  # (L1, L2)

    # Min-max normalise
    vmin, vmax = sim.min(), sim.max()
    if vmax - vmin > 1e-9:
        sim = (sim - vmin) / (vmax - vmin)
    else:
        sim = np.zeros_like(sim)

    # Resize
    if sim.shape[0] != target_size or sim.shape[1] != target_size:
        zr = target_size / sim.shape[0]
        zc = target_size / sim.shape[1]
        sim = zoom(sim, (zr, zc), order=1)

    np.clip(sim, 0.0, 1.0, out=sim)
    return sim.reshape(1, target_size, target_size).astype(np.float32)


def make_input(obj1: str | np.ndarray,
               obj2: str | np.ndarray,
               seq_type: str = "dna") -> tuple[np.ndarray, np.ndarray]:
    """MAIN FUNCTION — unified interface for the neural network.

    Mode detection by type of obj1:
      str       → Mode 1 (sequences): dotplot + kmer_features
      ndarray   → Mode 2 (profiles):  profile_similarity_matrix + profile_scalar_features

    Returns:
      matrix:  (1, 64, 64) float32
      scalars: (SCALAR_DIM,) float32
    """
    if isinstance(obj1, str) and isinstance(obj2, str):
        # Mode 1: raw sequences
        k = 4 if seq_type == "dna" else 3
        matrix = dotplot_tensor(obj1, obj2, target_size=MATRIX_SIZE, k=k)
        scalars = kmer_features(obj1, obj2, seq_type)
    else:
        # Mode 2: profile arrays (or mixed — convert string to 1-row profile)
        if isinstance(obj1, str):
            alpha = "ACGT-" if seq_type == "dna" else "ACDEFGHIKLMNPQRSTVWY-"
            char_to_idx = {c: i for i, c in enumerate(alpha)}
            prof = np.zeros((len(obj1), len(alpha)), dtype=np.float32)
            for i, ch in enumerate(obj1.upper()):
                prof[i, char_to_idx.get(ch, len(alpha) - 1)] = 1.0
            obj1 = prof
        if isinstance(obj2, str):
            alpha = "ACGT-" if seq_type == "dna" else "ACDEFGHIKLMNPQRSTVWY-"
            char_to_idx = {c: i for i, c in enumerate(alpha)}
            prof = np.zeros((len(obj2), len(alpha)), dtype=np.float32)
            for i, ch in enumerate(obj2.upper()):
                prof[i, char_to_idx.get(ch, len(alpha) - 1)] = 1.0
            obj2 = prof
        subst = DNA_SUBST if seq_type == "dna" else _ensure_blosum62()
        matrix = profile_similarity_matrix(obj1, obj2, subst, MATRIX_SIZE)
        scalars = profile_scalar_features(obj1, obj2, seq_type)

    return matrix, scalars


if __name__ == "__main__":
    # Smoke test — Mode 1 (sequences)
    s1 = "ACGTACGTACGTACGTACGTAAACCCGGGTTT" * 5
    s2 = "ACGTACATACGTACCTACGTAAACCCGGGTTT" * 5
    mat, scl = make_input(s1, s2, "dna")
    assert mat.shape == (1, 64, 64), f"Bad matrix shape: {mat.shape}"
    assert scl.shape == (SCALAR_DIM,), f"Bad scalar shape: {scl.shape}"
    assert mat.dtype == np.float32
    assert scl.dtype == np.float32
    print(f"Mode 1 (seq): matrix={mat.shape} scalars={scl.shape}")

    # Smoke test — Mode 2 (profiles)
    p1 = np.random.dirichlet(np.ones(5), size=100).astype(np.float32)
    p2 = np.random.dirichlet(np.ones(5), size=80).astype(np.float32)
    mat2, scl2 = make_input(p1, p2, "dna")
    assert mat2.shape == (1, 64, 64), f"Bad profile matrix shape: {mat2.shape}"
    assert scl2.shape == (SCALAR_DIM,), f"Bad profile scalar shape: {scl2.shape}"
    print(f"Mode 2 (profile): matrix={mat2.shape} scalars={scl2.shape}")

    # Protein test
    ps1 = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK" * 3
    ps2 = "MVLSGEDKSNIKAAWGKIGGHGAEYGAEALERMFLSFPTTK" * 3
    mat3, scl3 = make_input(ps1, ps2, "protein")
    assert mat3.shape == (1, 64, 64)
    print(f"Protein seq: matrix={mat3.shape}")

    print("All smoke tests passed!")
