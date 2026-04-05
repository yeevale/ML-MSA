// anchored_align.cpp — Alignment of blocks for long sequences.
// When max(len(seq1), len(seq2)) > MAX_DIRECT_LEN:
//   anchors → split_by_anchors → align blocks → concatenate.
// Each block is aligned independently using align_with_doubling.
// Anchor regions are perfect matches → no alignment needed.
// This file provides C++ helpers; the main logic is in features/anchors.py.

// NOTE: This file is #included from band_doubling.cpp.
// All symbols from the include chain are already available.

// Align a list of block pairs and concatenate results
// blocks: vector of (seq1_block, seq2_block, offset_i, offset_j)
// anchors: vector of (anchor_seq, offset_i, offset_j)
// Returns concatenated alignment

struct AnchorBlock {
    std::string seq1_block;
    std::string seq2_block;
    int offset_i;
    int offset_j;
    bool is_anchor;  // true = perfect match (no alignment needed)
};

struct AnchoredAlignResult {
    std::string aligned_seq1;
    std::string aligned_seq2;
    float total_score;
    int n_blocks;
    int n_doublings_total;
};

AnchoredAlignResult align_anchored_blocks(
    const std::vector<AnchorBlock>& blocks,
    float gap_open,
    float gap_extend,
    bool is_protein)
{
    AnchoredAlignResult result;
    result.total_score = 0.0f;
    result.n_blocks = static_cast<int>(blocks.size());
    result.n_doublings_total = 0;

    for (const auto& block : blocks) {
        if (block.is_anchor) {
            // Perfect match — just append
            result.aligned_seq1 += block.seq1_block;
            result.aligned_seq2 += block.seq2_block;
            // Score: sum of match scores
            float block_score = 0.0f;
            for (size_t k = 0; k < block.seq1_block.size(); ++k) {
                int a = is_protein ? encode_protein(block.seq1_block[k])
                                   : encode_dna(block.seq1_block[k]);
                int b = is_protein ? encode_protein(block.seq2_block[k])
                                   : encode_dna(block.seq2_block[k]);
                block_score += (a >= 0 && b >= 0 && a == b) ? 1.0f : -1.0f;
            }
            result.total_score += block_score;
        } else {
            // Align this block
            if (block.seq1_block.empty() && block.seq2_block.empty()) {
                continue;
            }
            if (block.seq1_block.empty()) {
                result.aligned_seq1 += std::string(block.seq2_block.size(), '-');
                result.aligned_seq2 += block.seq2_block;
                result.total_score += gap_open + gap_extend * block.seq2_block.size();
                continue;
            }
            if (block.seq2_block.empty()) {
                result.aligned_seq1 += block.seq1_block;
                result.aligned_seq2 += std::string(block.seq1_block.size(), '-');
                result.total_score += gap_open + gap_extend * block.seq1_block.size();
                continue;
            }

            // Use align_with_doubling with initial band guess
            int len1 = static_cast<int>(block.seq1_block.size());
            int len2 = static_cast<int>(block.seq2_block.size());
            int pred_centre = 0;
            int pred_hw = std::max(10, std::abs(len1 - len2) + 10);

            DoublingResult dr = align_with_doubling(
                block.seq1_block, block.seq2_block,
                pred_centre, pred_hw,
                gap_open, gap_extend, is_protein, nullptr);

            result.aligned_seq1 += dr.alignment.aligned_seq1;
            result.aligned_seq2 += dr.alignment.aligned_seq2;
            result.total_score += dr.alignment.score;
            result.n_doublings_total += dr.n_doublings;
        }
    }

    return result;
}

// Helper: find k-mer matches between two sequences (C++ version for speed)
// Returns list of (pos_in_seq1, pos_in_seq2, k) tuples
std::vector<std::tuple<int, int, int>> find_kmer_matches(
    const std::string& seq1,
    const std::string& seq2,
    int k)
{
    if (k <= 0 || static_cast<int>(seq1.size()) < k || static_cast<int>(seq2.size()) < k) {
        return {};
    }

    // Build hash map for seq2
    std::unordered_map<std::string, std::vector<int>> kmer_positions;
    for (int j = 0; j <= static_cast<int>(seq2.size()) - k; ++j) {
        kmer_positions[seq2.substr(j, k)].push_back(j);
    }

    // Find matches
    std::vector<std::tuple<int, int, int>> matches;
    for (int i = 0; i <= static_cast<int>(seq1.size()) - k; ++i) {
        std::string kmer = seq1.substr(i, k);
        auto it = kmer_positions.find(kmer);
        if (it != kmer_positions.end()) {
            for (int j : it->second) {
                matches.emplace_back(i, j, k);
            }
        }
    }

    return matches;
}

// LIS on (i, j) pairs for anchor chaining
// Returns indices of selected anchors in monotone increasing order
std::vector<int> lis_chain(const std::vector<std::pair<int, int>>& pts) {
    if (pts.empty()) return {};

    int n = static_cast<int>(pts.size());
    // Sort by first coordinate
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return pts[a].first < pts[b].first ||
               (pts[a].first == pts[b].first && pts[a].second < pts[b].second);
    });

    // Patience sorting on second coordinate
    std::vector<int> tails;      // smallest tail element of each pile
    std::vector<int> tails_idx;  // index in order[] of smallest tail
    std::vector<int> prev(n, -1);

    for (int idx : order) {
        int j_val = pts[idx].second;
        // Binary search for position
        int lo = 0, hi = static_cast<int>(tails.size());
        while (lo < hi) {
            int mid_pos = (lo + hi) / 2;
            if (tails[mid_pos] < j_val) lo = mid_pos + 1;
            else hi = mid_pos;
        }

        if (lo == static_cast<int>(tails.size())) {
            tails.push_back(j_val);
            tails_idx.push_back(idx);
        } else {
            tails[lo] = j_val;
            tails_idx[lo] = idx;
        }
        prev[idx] = (lo > 0) ? tails_idx[lo - 1] : -1;
    }

    // Reconstruct
    std::vector<int> result;
    int cur = tails_idx.back();
    while (cur >= 0) {
        result.push_back(cur);
        cur = prev[cur];
    }
    std::reverse(result.begin(), result.end());
    return result;
}
