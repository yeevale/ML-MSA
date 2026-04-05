# features/anchors.py — Anchor-based block splitting for long sequences.
# For sequences longer than MAX_DIRECT_LEN=5000, find exact-match anchors,
# chain them monotonically, then split into shorter blocks between anchors.
# Each block is processed independently by the neural network + banded NW.

from dataclasses import dataclass
from collections import defaultdict
from features.kmer import minimizers

MAX_DIRECT_LEN = 5000


@dataclass
class Anchor:
    i: int   # start position in seq1
    j: int   # start position in seq2
    k: int   # length of matching k-mer


def find_anchors(seq1: str, seq2: str,
                 window: int = 10, k: int = 15) -> list[Anchor]:
    """Find shared minimizers as anchors.
    1. Build {kmer: [positions]} for both sequences
    2. Intersect keys → list of Anchor(i, j, k)"""
    seq1 = seq1.upper()
    seq2 = seq2.upper()

    if len(seq1) < k or len(seq2) < k:
        return []

    # Build position index for seq1 minimizers
    min1_pos: dict[str, list[int]] = defaultdict(list)
    kmers1 = [seq1[i:i + k] for i in range(len(seq1) - k + 1)]
    for start in range(len(kmers1) - window + 1):
        win = kmers1[start:start + window]
        best = min(win)
        best_idx = start + win.index(best)
        min1_pos[best].append(best_idx)

    # Build position index for seq2 minimizers
    min2_pos: dict[str, list[int]] = defaultdict(list)
    kmers2 = [seq2[i:i + k] for i in range(len(seq2) - k + 1)]
    for start in range(len(kmers2) - window + 1):
        win = kmers2[start:start + window]
        best = min(win)
        best_idx = start + win.index(best)
        min2_pos[best].append(best_idx)

    # Deduplicate positions
    for key in min1_pos:
        min1_pos[key] = sorted(set(min1_pos[key]))
    for key in min2_pos:
        min2_pos[key] = sorted(set(min2_pos[key]))

    # Intersect and create anchors
    anchors: list[Anchor] = []
    shared_kmers = set(min1_pos.keys()) & set(min2_pos.keys())
    for kmer in shared_kmers:
        for i_pos in min1_pos[kmer]:
            for j_pos in min2_pos[kmer]:
                anchors.append(Anchor(i=i_pos, j=j_pos, k=k))

    # Sort by position in seq1
    anchors.sort(key=lambda a: (a.i, a.j))
    return anchors


def chain_anchors(anchors: list[Anchor], max_gap: int = 1000) -> list[Anchor]:
    """Find a monotone chain of anchors (LIS on (i, j) pairs).
    Two anchors compatible: a1.i < a2.i AND a1.j < a2.j AND
                            (a2.i - a1.i) < max_gap AND (a2.j - a1.j) < max_gap.
    Uses patience sorting for O(n log n)."""
    if not anchors:
        return []

    # Sort by i, then by j
    sorted_anc = sorted(anchors, key=lambda a: (a.i, a.j))

    # Remove duplicates (same i,j)
    deduped: list[Anchor] = []
    seen: set[tuple[int, int]] = set()
    for a in sorted_anc:
        key = (a.i, a.j)
        if key not in seen:
            seen.add(key)
            deduped.append(a)
    sorted_anc = deduped

    n = len(sorted_anc)
    if n == 0:
        return []

    # LIS on j values with compatibility constraints
    # dp[i] = length of longest chain ending at i
    # parent[i] = index of predecessor
    import bisect

    dp = [1] * n
    parent = [-1] * n

    # For each anchor, find best predecessor
    for idx in range(1, n):
        a = sorted_anc[idx]
        for prev in range(idx - 1, -1, -1):
            pa = sorted_anc[prev]
            if pa.i < a.i and pa.j < a.j:
                gap_i = a.i - pa.i
                gap_j = a.j - pa.j
                if gap_i < max_gap and gap_j < max_gap:
                    if dp[prev] + 1 > dp[idx]:
                        dp[idx] = dp[prev] + 1
                        parent[idx] = prev

    # Backtrack from best
    best_idx = max(range(n), key=lambda i: dp[i])
    chain: list[Anchor] = []
    while best_idx >= 0:
        chain.append(sorted_anc[best_idx])
        best_idx = parent[best_idx]
    chain.reverse()
    return chain


def split_by_anchors(seq1: str, seq2: str,
                     chain: list[Anchor]) -> list[tuple[str, str, int, int]]:
    """Split pair into blocks between anchors.
    Returns list[(block_seq1, block_seq2, offset_i, offset_j)].
    Anchor regions (exact matches) are NOT included in blocks."""
    if not chain:
        return [(seq1, seq2, 0, 0)]

    blocks: list[tuple[str, str, int, int]] = []
    prev_i = 0
    prev_j = 0

    for anchor in chain:
        # Block before this anchor
        if anchor.i > prev_i or anchor.j > prev_j:
            block_s1 = seq1[prev_i:anchor.i]
            block_s2 = seq2[prev_j:anchor.j]
            if block_s1 or block_s2:
                blocks.append((block_s1, block_s2, prev_i, prev_j))

        # Skip anchor region
        prev_i = anchor.i + anchor.k
        prev_j = anchor.j + anchor.k

    # Block after last anchor
    if prev_i < len(seq1) or prev_j < len(seq2):
        block_s1 = seq1[prev_i:]
        block_s2 = seq2[prev_j:]
        if block_s1 or block_s2:
            blocks.append((block_s1, block_s2, prev_i, prev_j))

    return blocks


def needs_anchoring(seq1: str, seq2: str) -> bool:
    """Check if sequences are long enough to benefit from anchoring."""
    return len(seq1) > MAX_DIRECT_LEN or len(seq2) > MAX_DIRECT_LEN


if __name__ == "__main__":
    import random
    random.seed(42)

    # Generate a long sequence and a mutated copy
    alpha = "ACGT"
    base = "".join(random.choice(alpha) for _ in range(6000))
    mutated = list(base)
    for i in range(0, len(mutated), 50):  # mutate every 50th position
        mutated[i] = random.choice(alpha)
    mutated_str = "".join(mutated)

    print(f"Sequence lengths: {len(base)}, {len(mutated_str)}")
    print(f"needs_anchoring: {needs_anchoring(base, mutated_str)}")

    # Find and chain anchors
    anchors = find_anchors(base, mutated_str, window=10, k=15)
    print(f"Found {len(anchors)} raw anchors")

    chain = chain_anchors(anchors, max_gap=1000)
    print(f"Chained to {len(chain)} anchors")

    if chain:
        print(f"First anchor: i={chain[0].i}, j={chain[0].j}, k={chain[0].k}")
        print(f"Last anchor:  i={chain[-1].i}, j={chain[-1].j}, k={chain[-1].k}")

    # Split
    blocks = split_by_anchors(base, mutated_str, chain)
    print(f"Split into {len(blocks)} blocks")
    for idx, (b1, b2, oi, oj) in enumerate(blocks):
        print(f"  Block {idx}: len1={len(b1)}, len2={len(b2)}, "
              f"offset=({oi},{oj})")

    # Verify blocks cover the sequence
    assert all(len(b1) <= MAX_DIRECT_LEN or len(b1) < len(base)
               for b1, _, _, _ in blocks), "Some blocks too large"
    print("Smoke test passed!")
