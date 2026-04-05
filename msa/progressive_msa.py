# msa/progressive_msa.py — Main progressive MSA pipeline.
# Post-order traversal of guide tree (bottom-up), N-1 merge steps.
# At each step: NN predicts band → C++ aligns within band.
#
# LAZY PROFILES: after merging left+right, delete both children.
# Peak memory: O(log N) profiles simultaneously.
#
# BATCHED INFERENCE BY LEVEL: tree_levels() groups same-depth nodes
# into a single batch for the neural net — critical for GPU efficiency.
#
# ANCHOR MODE: for sequences longer than MAX_DIRECT_LEN,
# split via anchors before aligning blocks.

import gc
import numpy as np

from msa.guide_tree import (
    pairwise_distance_matrix, build_guide_tree,
    tree_levels, TreeNode, assign_node_ids, get_leaves,
)
from features.profile_features import make_input
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
    alphabet = ACGT- (5) for DNA, 20aa+- (21) for protein."""
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


def _profile_gap_pattern(profile: np.ndarray, aligned_len: int,
                         other_len: int) -> str:
    """Build a gap pattern string for a profile after alignment.
    If C++ aligner returns empty aligned strings (score-only mode),
    fall back to a simple diagonal alignment representation."""
    L = profile.shape[0]
    # Create a string with L non-gap chars + (aligned_len - L) gaps
    return 'X' * L + '-' * max(0, aligned_len - L)


def _align_profiles_with_fallback(obj1: np.ndarray, obj2: np.ndarray,
                                   subst_np: np.ndarray,
                                   centre: int, hw: int) -> tuple[str, str]:
    """Align two profiles, handling C++ returning empty alignment strings.
    Returns (aligned_repr1, aligned_repr2) gap pattern strings."""
    r = aligner.align_profiles_with_doubling(
        obj1, obj2, subst_np, centre, hw)
    a1 = r.alignment.aligned_seq1
    a2 = r.alignment.aligned_seq2

    if a1 and a2:
        return a1, a2

    # C++ profile aligner returned empty strings (score-only mode).
    # Build gap patterns from a simple NW alignment of "consensus" strings.
    L1 = obj1.shape[0]
    L2 = obj2.shape[0]

    # Create dummy consensus sequences from profiles (argmax at each position)
    alpha = "ACGT-" if obj1.shape[1] == 5 else "ACDEFGHIKLMNPQRSTVWY-"
    cons1 = "".join(alpha[min(int(obj1[i].argmax()), len(alpha) - 1)]
                    for i in range(L1))
    cons2 = "".join(alpha[min(int(obj2[i].argmax()), len(alpha) - 1)]
                    for i in range(L2))

    # Remove gap characters from consensus for alignment
    cons1_nogap = cons1.replace("-", "")
    cons2_nogap = cons2.replace("-", "")

    if not cons1_nogap or not cons2_nogap:
        # Edge case: all-gap profiles — produce simple concatenation
        total = L1 + L2
        return 'X' * L1 + '-' * L2, '-' * L1 + 'X' * L2

    # Align consensus sequences using the banded aligner
    r2 = aligner.align_with_doubling(cons1_nogap, cons2_nogap, centre, hw)
    a1 = r2.alignment.aligned_seq1
    a2 = r2.alignment.aligned_seq2

    # Re-insert internal gaps from profiles into aligned consensus
    if cons1 != cons1_nogap:
        a1 = _reinsert_gaps(a1, cons1)
    if cons2 != cons2_nogap:
        a2 = _reinsert_gaps(a2, cons2)

    return a1, a2


def _reinsert_gaps(aligned: str, original_with_gaps: str) -> str:
    """Re-insert gaps from the original gapped string into the aligned version."""
    result: list[str] = []
    aligned_pos = 0
    for ch in original_with_gaps:
        if ch == '-':
            result.append('-')
        else:
            if aligned_pos < len(aligned):
                result.append(aligned[aligned_pos])
            else:
                result.append('-')
            aligned_pos += 1
    # Append any remaining chars from aligned (extra gaps from alignment)
    while aligned_pos < len(aligned):
        result.append(aligned[aligned_pos])
        aligned_pos += 1
    return "".join(result)


    """Return list[bool]: True where aligned string has a character, False for gap."""
    return [c != '-' for c in aligned]


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
    3. Merge sequences/profiles at each internal node
    4. Return final MSA as list of aligned strings
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
    node_objects: dict[int, str | np.ndarray] = {}
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
            obj1 = node_objects[node.left.node_id]
            obj2 = node_objects[node.right.node_id]
            left_seqs = seq_groups[node.left.node_id]
            right_seqs = seq_groups[node.right.node_id]

            # Align the pair
            if isinstance(obj1, str) and isinstance(obj2, str):
                raw1 = _ungap(obj1)
                raw2 = _ungap(obj2)
                if needs_anchoring(raw1, raw2):
                    a1, a2 = align_pair_with_anchors(
                        raw1, raw2, predictor, seq_type)
                else:
                    r = aligner.align_with_doubling(
                        raw1, raw2, centre, hw)
                    a1 = r.alignment.aligned_seq1
                    a2 = r.alignment.aligned_seq2

                new_left = apply_gaps_to_seqs(left_seqs, a1)
                new_right = apply_gaps_to_seqs(right_seqs, a2)

            elif isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray):
                # Profile-profile alignment
                subst = _get_subst_matrix(seq_type)
                subst_np = np.ascontiguousarray(subst, dtype=np.float32)
                a1, a2 = _align_profiles_with_fallback(
                    obj1, obj2, subst_np, centre, hw)

                new_left = apply_gaps_to_seqs(left_seqs, a1)
                new_right = apply_gaps_to_seqs(right_seqs, a2)
            else:
                # Mixed: one is str, other is profile
                # Convert str to profile first
                if isinstance(obj1, str):
                    obj1_p = build_profile([obj1], seq_type)
                    obj2_p = obj2
                else:
                    obj1_p = obj1
                    obj2_p = build_profile([obj2], seq_type)

                subst = _get_subst_matrix(seq_type)
                subst_np = np.ascontiguousarray(subst, dtype=np.float32)
                a1, a2 = _align_profiles_with_fallback(
                    obj1_p, obj2_p, subst_np, centre, hw)

                new_left = apply_gaps_to_seqs(left_seqs, a1)
                new_right = apply_gaps_to_seqs(right_seqs, a2)

            # Merge
            new_seqs = new_left + new_right
            new_profile = build_profile(new_seqs, seq_type)

            node_objects[node.node_id] = new_profile
            seq_groups[node.node_id] = new_seqs

            # Free children memory
            del node_objects[node.left.node_id]
            del node_objects[node.right.node_id]
            del seq_groups[node.left.node_id]
            del seq_groups[node.right.node_id]
            gc.collect()

    # 4. Return final MSA
    return seq_groups[tree.node_id]


def _get_subst_matrix(seq_type: str) -> np.ndarray:
    """Get substitution matrix for profile-profile alignment."""
    from features.profile_features import DNA_SUBST, _ensure_blosum62
    if seq_type == "dna":
        return DNA_SUBST
    else:
        return _ensure_blosum62()


if __name__ == "__main__":
    # Smoke test — just test build_profile and apply_gaps
    seqs = ["ACGT", "AC-T", "A-GT"]
    profile = build_profile(seqs, "dna")
    print(f"Profile shape: {profile.shape}")
    assert profile.shape == (4, 5)  # 4 cols, 5 = ACGT-
    print(f"Column 0 freqs: {profile[0]}")  # A should be 1.0

    # Test apply_gaps_to_seqs
    aligned = "A-C-GT"
    original = ["ACGT"]
    result = apply_gaps_to_seqs(original, aligned)
    print(f"apply_gaps: '{original[0]}' with pattern '{aligned}' → '{result[0]}'")
    assert len(result[0]) == len(aligned)

    print("Smoke test passed!")
