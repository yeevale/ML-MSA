# msa/progressive_msa.py — Main progressive MSA pipeline.
# Post-order traversal of guide tree (bottom-up), N-1 merge steps.
# At each step: NN predicts band → C++ aligns within band.
#
# CONSENSUS-BASED ALIGNMENT: each node stores a consensus string
# (majority-vote per column). Alignment uses fast sequence NW (SIMD),
# not slow profile-profile DP.
#
# BATCHED INFERENCE BY LEVEL: tree_levels() groups same-depth nodes
# into a single batch for the neural net — critical for GPU efficiency.
#
# ANCHOR MODE: for sequences longer than MAX_DIRECT_LEN,
# split via anchors before aligning blocks.

import numpy as np

from msa.guide_tree import (
    pairwise_distance_matrix, build_guide_tree,
    tree_levels, TreeNode, assign_node_ids, get_leaves,
)
from features.anchors import (
    MAX_DIRECT_LEN, find_anchors, chain_anchors, split_by_anchors, needs_anchoring,
)
from model.evaluate import BandPredictorInference
import aligner

DNA_ALPHABET = "ACGT-"
PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"


def build_profile(aligned_seqs: list[str],
                  seq_type: str = "dna") -> np.ndarray:
    """Build frequency profile from aligned sequences.
    Returns shape (alignment_length, alphabet_size) float32.
    alphabet = ACGT- (5) for DNA, 20aa+- (21) for protein.
    Kept for iterative_refine compatibility."""
    if not aligned_seqs:
        return np.zeros((0, 5 if seq_type == "dna" else 21), dtype=np.float32)

    alpha = DNA_ALPHABET if seq_type == "dna" else PROTEIN_ALPHABET
    A = len(alpha)
    char_to_idx = {c: i for i, c in enumerate(alpha)}
    L = len(aligned_seqs[0])
    n_seqs = len(aligned_seqs)

    profile = np.zeros((L, A), dtype=np.float32)
    for seq in aligned_seqs:
        for col, ch in enumerate(seq.upper()):
            idx = char_to_idx.get(ch, A - 1)  # unknown → gap column
            profile[col, idx] += 1.0

    profile /= max(n_seqs, 1)
    return profile


def _consensus_seq(aligned_seqs: list[str], seq_type: str = "dna") -> str:
    """Majority-vote consensus from aligned sequences.
    Gap-only columns get 'N' (DNA) or 'X' (protein) to preserve column count.
    This ensures len(consensus) == len(aligned_seqs[0])."""
    if not aligned_seqs:
        return ""
    L = len(aligned_seqs[0])
    alpha = set("ACGT") if seq_type == "dna" else set("ACDEFGHIKLMNPQRSTVWY")
    gap_rep = 'N' if seq_type == "dna" else 'X'
    result: list[str] = []
    for col in range(L):
        counts: dict[str, int] = {}
        for seq in aligned_seqs:
            c = seq[col].upper()
            if c in alpha:
                counts[c] = counts.get(c, 0) + 1
        if counts:
            result.append(max(counts, key=counts.get))
        else:
            result.append(gap_rep)
    return "".join(result)


def apply_gaps_to_seqs(seqs: list[str], aligned_repr: str) -> list[str]:
    """Apply gap pattern from aligned_repr to all sequences in seqs.
    Where aligned_repr has '-', insert '-' in each seq.
    Where aligned_repr has a char, consume next char from each seq."""
    result: list[str] = []
    for seq in seqs:
        # seq has its own gaps; we need to insert additional gaps from aligned_repr
        new_chars: list[str] = []
        seq_pos = 0
        for ch in aligned_repr:
            if ch == '-':
                new_chars.append('-')
            else:
                if seq_pos < len(seq):
                    new_chars.append(seq[seq_pos])
                else:
                    new_chars.append('-')
                seq_pos += 1
        result.append("".join(new_chars))
    return result


def _ungap(seq: str) -> str:
    """Remove gap characters."""
    return seq.replace("-", "")


def align_pair_with_anchors(seq1: str, seq2: str,
                            predictor: BandPredictorInference,
                            seq_type: str) -> tuple[str, str]:
    """Align a long pair via anchor mode.
    1. find_anchors → chain_anchors → split_by_anchors
    2. For each block: predict band + align_with_doubling
    3. Concatenate: anchors (perfect match) + aligned blocks."""
    raw1 = _ungap(seq1)
    raw2 = _ungap(seq2)

    anchors = find_anchors(raw1, raw2, window=10, k=15)
    chain = chain_anchors(anchors, max_gap=1000)

    if not chain:
        # No anchors found, align directly
        cd, hw = predictor.predict_single(raw1, raw2, seq_type)
        r = aligner.align_with_doubling(raw1, raw2, cd, hw)
        return r.alignment.aligned_seq1, r.alignment.aligned_seq2

    blocks = split_by_anchors(raw1, raw2, chain)

    aligned1_parts: list[str] = []
    aligned2_parts: list[str] = []
    prev_anchor_end_i = 0
    prev_anchor_end_j = 0

    for block_idx, (blk1, blk2, off_i, off_j) in enumerate(blocks):
        # Find which anchor (if any) precedes this block
        if block_idx > 0 or (chain and chain[0].i > 0):
            pass  # gap between blocks handled by block iteration

        if len(blk1) == 0 and len(blk2) == 0:
            continue

        if len(blk1) == 0:
            aligned1_parts.append("-" * len(blk2))
            aligned2_parts.append(blk2)
        elif len(blk2) == 0:
            aligned1_parts.append(blk1)
            aligned2_parts.append("-" * len(blk1))
        else:
            cd, hw = predictor.predict_single(blk1, blk2, seq_type)
            r = aligner.align_with_doubling(blk1, blk2, cd, hw)
            aligned1_parts.append(r.alignment.aligned_seq1)
            aligned2_parts.append(r.alignment.aligned_seq2)

        # Check if an anchor follows
        # Find the anchor that starts at off_i + len(blk1) in seq1
        for anc in chain:
            if anc.i == off_i + len(blk1):
                anchor_seq = raw1[anc.i:anc.i + anc.k]
                aligned1_parts.append(anchor_seq)
                aligned2_parts.append(anchor_seq)
                break

    # Handle trailing anchor if it follows last block
    if chain:
        last_anc = chain[-1]
        last_end_i = last_anc.i + last_anc.k
        last_end_j = last_anc.j + last_anc.k
        # Check if last anchor was already appended
        total1 = sum(len(p.replace("-", "")) for p in aligned1_parts)
        if total1 < len(raw1):
            # Missing the last anchor + trailing
            remaining1 = raw1[total1:]
            total2 = sum(len(p.replace("-", "")) for p in aligned2_parts)
            remaining2 = raw2[total2:]
            if remaining1 or remaining2:
                if remaining1 and remaining2:
                    cd, hw = predictor.predict_single(remaining1, remaining2, seq_type)
                    r = aligner.align_with_doubling(remaining1, remaining2, cd, hw)
                    aligned1_parts.append(r.alignment.aligned_seq1)
                    aligned2_parts.append(r.alignment.aligned_seq2)
                elif remaining1:
                    aligned1_parts.append(remaining1)
                    aligned2_parts.append("-" * len(remaining1))
                else:
                    aligned1_parts.append("-" * len(remaining2))
                    aligned2_parts.append(remaining2)

    return "".join(aligned1_parts), "".join(aligned2_parts)


def progressive_msa(sequences: list[str],
                    seq_ids: list[str],
                    predictor: BandPredictorInference,
                    seq_type: str = "dna",
                    tree_method: str = "nj",
                    n_jobs: int = -1) -> list[str]:
    """Main progressive MSA function.

    1. Build distance matrix + guide tree
    2. Traverse bottom-up by level, batching NN predictions
    3. Align consensus sequences at each internal node (fast SIMD NW)
    4. Apply gap pattern to all child sequences
    5. Return final MSA as list of aligned strings
    """
    n = len(sequences)
    if n == 0:
        return []
    if n == 1:
        return [sequences[0]]

    # 1. Distance matrix and guide tree
    dist_matrix = pairwise_distance_matrix(sequences, seq_type, n_jobs)
    tree = build_guide_tree(dist_matrix, tree_method)
    assign_node_ids(tree)

    # 2. Initialise leaf data
    leaves = get_leaves(tree)
    # node_objects stores consensus strings (for NN features + alignment)
    node_objects: dict[int, str] = {}
    seq_groups: dict[int, list[str]] = {}

    for leaf in leaves:
        node_objects[leaf.node_id] = sequences[leaf.seq_idx]
        seq_groups[leaf.node_id] = [sequences[leaf.seq_idx]]

    # 3. Process levels bottom-up
    levels = tree_levels(tree)

    for level in levels:
        # Collect pairs for batched NN inference
        pairs: list[tuple] = []
        for node in level:
            obj1 = node_objects[node.left.node_id]
            obj2 = node_objects[node.right.node_id]
            pairs.append((obj1, obj2))

        # Batched prediction
        predictions = predictor.predict_batch(pairs, seq_type)

        # Process each node in this level
        for node, (centre, hw) in zip(level, predictions):
            cons_left = node_objects[node.left.node_id]
            cons_right = node_objects[node.right.node_id]
            left_seqs = seq_groups[node.left.node_id]
            right_seqs = seq_groups[node.right.node_id]

            # Align consensus sequences (fast SIMD-accelerated NW)
            if needs_anchoring(cons_left, cons_right):
                a1, a2 = align_pair_with_anchors(
                    cons_left, cons_right, predictor, seq_type)
            else:
                r = aligner.align_with_doubling(
                    cons_left, cons_right, centre, hw)
                a1 = r.alignment.aligned_seq1
                a2 = r.alignment.aligned_seq2

            # Apply gap pattern from consensus alignment to all sequences
            new_left = apply_gaps_to_seqs(left_seqs, a1)
            new_right = apply_gaps_to_seqs(right_seqs, a2)

            # Merge
            new_seqs = new_left + new_right

            # Store consensus for next level (for NN features + alignment)
            node_objects[node.node_id] = _consensus_seq(new_seqs, seq_type)
            seq_groups[node.node_id] = new_seqs

            # Free children memory
            del node_objects[node.left.node_id]
            del node_objects[node.right.node_id]
            del seq_groups[node.left.node_id]
            del seq_groups[node.right.node_id]

    # 4. Return final MSA
    return seq_groups[tree.node_id]


if __name__ == "__main__":
    # Smoke test — test build_profile, consensus, and apply_gaps
    seqs = ["ACGT", "AC-T", "A-GT"]
    profile = build_profile(seqs, "dna")
    print(f"Profile shape: {profile.shape}")
    assert profile.shape == (4, 5)  # 4 cols, 5 = ACGT-
    print(f"Column 0 freqs: {profile[0]}")  # A should be 1.0

    # Test consensus
    cons = _consensus_seq(seqs, "dna")
    print(f"Consensus: '{cons}'")
    assert len(cons) == 4  # same as aligned length

    # Test apply_gaps_to_seqs
    aligned = "A-C-GT"
    original = ["ACGT"]
    result = apply_gaps_to_seqs(original, aligned)
    print(f"apply_gaps: '{original[0]}' with pattern '{aligned}' → '{result[0]}'")
    assert len(result[0]) == len(aligned)

    print("Smoke test passed!")
