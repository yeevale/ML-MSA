# msa/guide_tree.py — Guide tree construction for progressive MSA.
# Distance matrix from k-mer Jaccard (no alignment needed).
# NJ (Neighbour-Joining) or UPGMA via scipy.
# tree_levels() groups internal nodes by BFS level for batched NN inference.

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

from features.kmer import minimizers


@dataclass
class TreeNode:
    """Guide tree node."""
    left:     Optional['TreeNode'] = None
    right:    Optional['TreeNode'] = None
    seq_idx:  Optional[int] = None       # leaf only
    distance: float = 0.0
    node_id:  int = -1                   # unique ID for node_objects dict


def kmer_jaccard_dist(seq1: str, seq2: str, k: int = 4) -> float:
    """Jaccard distance from k-mer sets: 1 - |A∩B| / |A∪B|."""
    s1_upper = seq1.upper()
    s2_upper = seq2.upper()
    if len(s1_upper) < k or len(s2_upper) < k:
        return 1.0
    set1 = {s1_upper[i:i + k] for i in range(len(s1_upper) - k + 1)}
    set2 = {s2_upper[i:i + k] for i in range(len(s2_upper) - k + 1)}
    union = set1 | set2
    if not union:
        return 1.0
    return 1.0 - len(set1 & set2) / len(union)


def pairwise_distance_matrix(sequences: list[str],
                              seq_type: str = "dna",
                              n_jobs: int = -1) -> np.ndarray:
    """Symmetric distance matrix (N, N) with zeros on diagonal.
    k=4 for DNA, k=3 for protein. Parallelised via joblib."""
    n = len(sequences)
    k = 4 if seq_type == "dna" else 3

    # Compute upper triangle
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    try:
        from joblib import Parallel, delayed
        dists = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(kmer_jaccard_dist)(sequences[i], sequences[j], k)
            for i, j in pairs
        )
    except ImportError:
        dists = [kmer_jaccard_dist(sequences[i], sequences[j], k)
                 for i, j in pairs]

    dist_matrix = np.zeros((n, n), dtype=np.float64)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist_matrix[i, j] = dists[idx]
            dist_matrix[j, i] = dists[idx]
            idx += 1
    return dist_matrix


def _scipy_tree_to_treenode(cnode, n_leaves: int) -> TreeNode:
    """Recursively convert scipy ClusterNode to our TreeNode."""
    if cnode.is_leaf():
        return TreeNode(seq_idx=cnode.id)

    left = _scipy_tree_to_treenode(cnode.get_left(), n_leaves)
    right = _scipy_tree_to_treenode(cnode.get_right(), n_leaves)
    return TreeNode(left=left, right=right, distance=cnode.dist)


def build_guide_tree(dist_matrix: np.ndarray,
                     method: str = "nj") -> TreeNode:
    """Build guide tree from distance matrix.
    method='upgma': scipy linkage average
    method='nj': Neighbour-Joining (approximated via scipy 'ward' for robustness,
                 or BioPython NJ if available)"""
    n = dist_matrix.shape[0]
    if n <= 1:
        return TreeNode(seq_idx=0)

    # Ensure no zeros on off-diagonal (scipy needs positive distances for condensed form)
    dm = dist_matrix.copy()
    np.fill_diagonal(dm, 0.0)
    # Clamp small values
    dm = np.maximum(dm, 1e-10)
    np.fill_diagonal(dm, 0.0)

    condensed = squareform(dm, checks=False)

    if method == "upgma":
        Z = linkage(condensed, method="average")
    elif method == "nj":
        # Try BioPython NJ first
        try:
            from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
            names = [str(i) for i in range(n)]
            # BioPython DistanceMatrix takes lower-triangle lists
            matrix_list = []
            for i in range(n):
                row = [dm[i, j] for j in range(i + 1)]
                matrix_list.append(row)
            bio_dm = DistanceMatrix(names, matrix_list)
            constructor = DistanceTreeConstructor()
            bio_tree = constructor.nj(bio_dm)

            # Convert BioPython tree to our TreeNode
            return _biopython_to_treenode(bio_tree, n)
        except (ImportError, Exception):
            # Fallback to UPGMA via scipy
            Z = linkage(condensed, method="average")
    else:
        Z = linkage(condensed, method="average")

    root_cnode = to_tree(Z)
    return _scipy_tree_to_treenode(root_cnode, n)


def _biopython_to_treenode(bio_tree, n_leaves: int) -> TreeNode:
    """Convert BioPython NJ tree to our TreeNode format."""
    from Bio.Phylo.BaseTree import Tree, Clade

    root_clade = bio_tree.root

    def convert(clade: Clade) -> TreeNode:
        children = list(clade.clades)
        if not children:
            # Leaf
            try:
                idx = int(clade.name)
            except (ValueError, TypeError):
                idx = 0
            return TreeNode(seq_idx=idx,
                           distance=clade.branch_length or 0.0)

        if len(children) == 1:
            return convert(children[0])

        # For NJ, root might have 2 or 3 children
        # Merge into binary tree by pairing left-to-right
        nodes = [convert(c) for c in children]
        while len(nodes) > 1:
            left = nodes.pop(0)
            right = nodes.pop(0)
            parent = TreeNode(left=left, right=right,
                            distance=max(left.distance, right.distance))
            nodes.append(parent)
        return nodes[0]

    return convert(root_clade)


def assign_node_ids(root: TreeNode) -> int:
    """DFS: assign unique node_id to all nodes. Returns max_id + 1."""
    counter = [0]

    def dfs(node: TreeNode) -> None:
        if node is None:
            return
        dfs(node.left)
        dfs(node.right)
        node.node_id = counter[0]
        counter[0] += 1

    dfs(root)
    return counter[0]


def get_leaves(root: TreeNode) -> list[TreeNode]:
    """Collect all leaf nodes via DFS."""
    leaves: list[TreeNode] = []

    def dfs(node: TreeNode) -> None:
        if node is None:
            return
        if node.left is None and node.right is None:
            leaves.append(node)
        else:
            dfs(node.left)
            dfs(node.right)

    dfs(root)
    return leaves


def tree_levels(root: TreeNode) -> list[list[TreeNode]]:
    """BFS: return list of levels from leaves to root.
    levels[0] = internal nodes whose both children are leaves
    levels[-1] = [root]
    Used for batched NN inference by level."""
    if root.left is None and root.right is None:
        return []  # single leaf, no internal nodes

    # First compute depth of each node
    depth_map: dict[int, int] = {}

    def compute_depth(node: TreeNode) -> int:
        if node is None:
            return 0
        if node.left is None and node.right is None:
            depth_map[node.node_id] = 0
            return 0
        ld = compute_depth(node.left)
        rd = compute_depth(node.right)
        d = max(ld, rd) + 1
        depth_map[node.node_id] = d
        return d

    compute_depth(root)

    # Collect internal nodes by depth (bottom-up)
    max_depth = depth_map.get(root.node_id, 0)
    levels: list[list[TreeNode]] = [[] for _ in range(max_depth)]

    def collect(node: TreeNode) -> None:
        if node is None:
            return
        if node.left is None and node.right is None:
            return  # skip leaves
        d = depth_map[node.node_id]
        levels[d - 1].append(node)  # depth 1 → levels[0], etc.
        collect(node.left)
        collect(node.right)

    collect(root)

    # Remove empty levels
    levels = [lev for lev in levels if lev]
    return levels


if __name__ == "__main__":
    # Smoke test
    seqs = [
        "ACGTACGTACGTACGT",
        "ACGTACATACGTACGT",
        "GGGCCCTTTAAAGGG",
        "GGGCCCTTAAAAGGG",
        "ATATATATATATATATAT",
    ]

    dm = pairwise_distance_matrix(seqs, "dna", n_jobs=1)
    print(f"Distance matrix shape: {dm.shape}")
    print(f"Distance range: [{dm.min():.3f}, {dm[dm > 0].max():.3f}]")

    tree = build_guide_tree(dm, method="upgma")
    n_ids = assign_node_ids(tree)
    print(f"Tree nodes: {n_ids}")

    leaves = get_leaves(tree)
    print(f"Leaves: {len(leaves)}, seq_idxs: {[l.seq_idx for l in leaves]}")

    levels = tree_levels(tree)
    print(f"Levels (bottom-up): {len(levels)}")
    for i, lev in enumerate(levels):
        print(f"  Level {i}: {len(lev)} internal nodes "
              f"(ids: {[n.node_id for n in lev]})")

    assert len(leaves) == 5
    assert n_ids == 9  # 5 leaves + 4 internal
    print("Smoke test passed!")
