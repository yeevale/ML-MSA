# features/dotplot.py — Compressed dot-plot tensor for the CNN branch.
# Output shape is ALWAYS (1, 64, 64) float32 regardless of input lengths.

import numpy as np
from collections import defaultdict
from scipy.ndimage import zoom

MATRIX_SIZE = 64


def dotplot_tensor(seq1: str, seq2: str,
                   target_size: int = MATRIX_SIZE,
                   k: int = 4) -> np.ndarray:
    """Build a compressed dot-plot tensor.

    1. Hash all k-mers of seq2 into kmer_to_pos2
    2. Build binary dot matrix (len1-k+1, len2-k+1)
    3. Resize to (target_size, target_size) via bilinear zoom
    4. Min-max normalise to [0, 1]
    5. Return shape (1, target_size, target_size) float32
    """
    seq1 = seq1.upper()
    seq2 = seq2.upper()
    len1 = len(seq1)
    len2 = len(seq2)

    if len1 < k or len2 < k:
        return np.zeros((1, target_size, target_size), dtype=np.float32)

    rows = len1 - k + 1
    cols = len2 - k + 1

    # Step 1: hash k-mers of seq2
    kmer_to_pos2: dict[str, list[int]] = defaultdict(list)
    for j in range(cols):
        kmer_to_pos2[seq2[j:j + k]].append(j)

    # Step 2: build binary dot matrix
    dot = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        kmer = seq1[i:i + k]
        for j in kmer_to_pos2.get(kmer, []):
            dot[i, j] = 1.0

    # Step 3: resize to target_size × target_size
    if rows == target_size and cols == target_size:
        dot_small = dot
    else:
        zoom_r = target_size / rows
        zoom_c = target_size / cols
        dot_small = zoom(dot, (zoom_r, zoom_c), order=1)

    # Step 4: min-max normalise
    vmax = dot_small.max()
    if vmax > 0:
        dot_small = dot_small / vmax

    # Clip to [0,1] (zoom can produce slight negatives)
    np.clip(dot_small, 0.0, 1.0, out=dot_small)

    # Step 5: add channel dim
    return dot_small.reshape(1, target_size, target_size).astype(np.float32)


if __name__ == "__main__":
    # Smoke test
    s1 = "ACGTACGTACGTACGTACGTAAACCCGGGTTT" * 5
    s2 = "ACGTACATACGTACCTACGTAAACCCGGGTTT" * 5

    t = dotplot_tensor(s1, s2, k=4)
    assert t.shape == (1, 64, 64), f"Bad shape: {t.shape}"
    assert t.dtype == np.float32
    assert t.min() >= 0.0 and t.max() <= 1.0
    print(f"dotplot shape={t.shape} min={t.min():.3f} max={t.max():.3f} "
          f"nonzero={np.count_nonzero(t)}")

    # Very short sequences → zero tensor
    t_short = dotplot_tensor("AC", "GT", k=4)
    assert t_short.shape == (1, 64, 64)
    assert t_short.max() == 0.0
    print("Short sequence → zeros: OK")

    # Protein test k=3
    ps1 = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK" * 3
    ps2 = "MVLSGEDKSNIKAAWGKIGGHGAEYGAEALERMFLSFPTTK" * 3
    tp = dotplot_tensor(ps1, ps2, k=3)
    assert tp.shape == (1, 64, 64)
    print(f"Protein dotplot nonzero={np.count_nonzero(tp)}")
    print("Smoke test passed!")
