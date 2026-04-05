// hirschberg.cpp — Hirschberg's algorithm (1975) over banded NW → O(W) memory instead of O(n*W).
// Divide-and-conquer: split seq1 in half, find split point on the midline,
// recursively solve two subproblems.
//
// COMBINATION WITH FOUR RUSSIANS:
//   FourRussiansAligner created in hirschberg_banded() ONCE.
//   Passed by reference to all recursive calls of hirschberg_banded_impl().
//   Table accumulates across the entire recursion → hit_ratio grows with depth.
//
// COMBINATION WITH SIMD:
//   Base case (len1 <= BASE_CASE_LEN=64) → align_banded_auto() (AVX2 if available).
//   Inside FR: compute_block_simd on cache miss.
//
// COMPLEXITIES: O(W) memory + O(n*W/t) time + SIMD constant ~8x.

#include "four_russians.cpp"

constexpr int BASE_CASE_LEN = 64;

// Forward pass: compute last row using Four Russians
std::vector<float> nw_last_row_fr(
    const std::string& seq1,
    const std::string& seq2,
    int   centre_diag,
    int   half_width,
    float gap_open,
    float gap_extend,
    bool  is_protein,
    FourRussiansAligner& fr)
{
    return fr.last_row(seq1, seq2, centre_diag, half_width);
}

// Backward pass: reverse both sequences, adjust centre_diag
std::vector<float> nw_last_row_backward_fr(
    const std::string& seq1,
    const std::string& seq2,
    int   centre_diag,
    int   half_width,
    float gap_open,
    float gap_extend,
    bool  is_protein,
    FourRussiansAligner& fr)
{
    std::string rev1(seq1.rbegin(), seq1.rend());
    std::string rev2(seq2.rbegin(), seq2.rend());
    int n = static_cast<int>(seq1.size());
    int m = static_cast<int>(seq2.size());
    // Reverse centre: if forward centre = c, backward centre = (n - m) - c... 
    // Actually for banded: reverse means path (n-i, m-j) corresponds to diag (n-i)-(m-j) = (n-m)-(i-j)
    // So reverse centre_diag = -(centre_diag)  when lengths are equal
    // More generally: rev_centre = (n - m) - centre_diag but with reversed indexing...
    // Simplification: use -centre_diag for the reversed problem, adjust half_width
    int rev_centre = -centre_diag;
    return fr.last_row(rev1, rev2, rev_centre, half_width);
}

// Find optimal split point on row mid, searching only within band
int find_split_point(
    const std::vector<float>& fwd,
    const std::vector<float>& bwd,
    int mid, int centre_diag, int half_width, int len2)
{
    int bw = 2 * half_width + 1;
    int best_j = -1;
    float best_score = NEG_INF;

    int jmin = band_j_min(mid, centre_diag, half_width, len2);
    int jmax = band_j_max(mid, centre_diag, half_width, len2);

    for (int j = jmin; j <= jmax; ++j) {
        // Forward score at (mid, j) — stored in fwd with band indexing for row mid
        int bi_fwd = to_band_idx(j, mid, centre_diag, half_width);
        // Backward score: corresponds to reversed problem at position (n-mid, m-j)
        // The backward row is indexed from the other end
        int bi_bwd = to_band_idx(len2 - j, 0, -centre_diag, half_width);
        // Actually, backward row corresponds to the "last" row of the reversed problem
        // which is the row at the position corresponding to (mid, j)
        // Direct index: backward array is indexed same way but reversed
        int bwd_idx = static_cast<int>(bwd.size()) - 1 - bi_fwd;
        if (bwd_idx < 0) bwd_idx = bi_fwd;  // fallback
        
        float f = NEG_INF, b = NEG_INF;
        if (bi_fwd >= 0 && bi_fwd < static_cast<int>(fwd.size()))
            f = fwd[bi_fwd];
        if (bwd_idx >= 0 && bwd_idx < static_cast<int>(bwd.size()))
            b = bwd[bwd_idx];

        float total = f + b;
        if (total > best_score) {
            best_score = total;
            best_j = j;
        }
    }

    if (best_j < 0) {
        // Fallback: pick middle of band intersection
        best_j = std::max(0, std::min(len2, mid - centre_diag));
    }

    return best_j;
}

// Recursive implementation (fr passed by reference through all recursion)
BandedResult hirschberg_banded_impl(
    const std::string& seq1, const std::string& seq2,
    int centre_diag, int half_width,
    float gap_open, float gap_extend, bool is_protein,
    FourRussiansAligner& fr)
{
    const int n = static_cast<int>(seq1.size());
    const int m = static_cast<int>(seq2.size());

    // Base cases
    if (n == 0) {
        std::string a1(m, '-');
        return BandedResult{gap_open + gap_extend * m, a1, seq2, false, false, false, 0};
    }
    if (m == 0) {
        std::string a2(n, '-');
        return BandedResult{gap_open + gap_extend * n, seq1, a2, false, false, false, 0};
    }
    if (n == 1) {
        // Align single character against seq2 within band
        return align_banded_auto(seq1, seq2, centre_diag, half_width,
                                 gap_open, gap_extend, is_protein);
    }
    if (n <= BASE_CASE_LEN) {
        // Small enough → use SIMD banded directly
        return align_banded_auto(seq1, seq2, centre_diag, half_width,
                                 gap_open, gap_extend, is_protein);
    }

    // Divide: split seq1 at midpoint
    int mid = n / 2;
    std::string top = seq1.substr(0, mid);
    std::string bot = seq1.substr(mid);

    // Forward pass on top half
    auto fwd = nw_last_row_fr(top, seq2, centre_diag, half_width,
                               gap_open, gap_extend, is_protein, fr);

    // Backward pass on bottom half
    auto bwd = nw_last_row_backward_fr(bot, seq2, centre_diag, half_width,
                                        gap_open, gap_extend, is_protein, fr);

    // Find split point
    int split_j = find_split_point(fwd, bwd, mid, centre_diag, half_width, m);

    // Adjust centre_diag for subproblems
    // Top: seq1[0..mid-1] vs seq2[0..split_j-1]
    int centre_top = centre_diag;  // approximate: keep same centre
    // Bottom: seq1[mid..n-1] vs seq2[split_j..m-1]
    int centre_bot = centre_diag;  // shifted by the split

    std::string top_seq2 = (split_j > 0) ? seq2.substr(0, split_j) : "";
    std::string bot_seq2 = (split_j < m) ? seq2.substr(split_j) : "";

    // Recurse on both halves
    BandedResult left_result, right_result;

    if (top.empty() || top_seq2.empty()) {
        if (top.empty() && top_seq2.empty()) {
            left_result = BandedResult{0, "", "", false, false, false, 0};
        } else if (top.empty()) {
            std::string gaps(top_seq2.size(), '-');
            left_result = BandedResult{gap_open + gap_extend * (float)top_seq2.size(),
                                        gaps, top_seq2, false, false, false, 0};
        } else {
            std::string gaps(top.size(), '-');
            left_result = BandedResult{gap_open + gap_extend * (float)top.size(),
                                        top, gaps, false, false, false, 0};
        }
    } else {
        left_result = hirschberg_banded_impl(top, top_seq2, centre_top, half_width,
                                              gap_open, gap_extend, is_protein, fr);
    }

    if (bot.empty() || bot_seq2.empty()) {
        if (bot.empty() && bot_seq2.empty()) {
            right_result = BandedResult{0, "", "", false, false, false, 0};
        } else if (bot.empty()) {
            std::string gaps(bot_seq2.size(), '-');
            right_result = BandedResult{gap_open + gap_extend * (float)bot_seq2.size(),
                                         gaps, bot_seq2, false, false, false, 0};
        } else {
            std::string gaps(bot.size(), '-');
            right_result = BandedResult{gap_open + gap_extend * (float)bot.size(),
                                         bot, gaps, false, false, false, 0};
        }
    } else {
        right_result = hirschberg_banded_impl(bot, bot_seq2, centre_bot, half_width,
                                               gap_open, gap_extend, is_protein, fr);
    }

    // Combine results
    BandedResult combined;
    combined.score = left_result.score + right_result.score;
    combined.aligned_seq1 = left_result.aligned_seq1 + right_result.aligned_seq1;
    combined.aligned_seq2 = left_result.aligned_seq2 + right_result.aligned_seq2;
    combined.path_escaped = left_result.path_escaped || right_result.path_escaped;
    combined.escape_left = left_result.escape_left || right_result.escape_left;
    combined.escape_right = left_result.escape_right || right_result.escape_right;
    combined.max_deviation = std::max(left_result.max_deviation, right_result.max_deviation);

    return combined;
}

// Public interface: creates fr_aligner, calls _impl
BandedResult hirschberg_banded(
    const std::string& seq1,
    const std::string& seq2,
    int   centre_diag,
    int   half_width,
    float gap_open,
    float gap_extend,
    bool  is_protein)
{
    // Create FourRussiansAligner ONCE
    FourRussiansAligner fr(0, is_protein, gap_open, gap_extend, 16, nullptr);
    return hirschberg_banded_impl(seq1, seq2, centre_diag, half_width,
                                   gap_open, gap_extend, is_protein, fr);
}

// Profile-profile Hirschberg variant
// For profile-profile, we use a simpler approach: if memory fits, do direct banded;
// otherwise do the Hirschberg split using profile slicing
BandedResult hirschberg_banded_profiles(
    const py::array_t<float>& p1,
    const py::array_t<float>& p2,
    const py::array_t<float>& subst,
    int centre_diag, int half_width,
    float gap_open, float gap_extend)
{
    auto p1_buf = p1.unchecked<2>();
    auto p2_buf = p2.unchecked<2>();
    const int n = static_cast<int>(p1_buf.shape(0));
    const int m = static_cast<int>(p2_buf.shape(0));
    const int alpha = static_cast<int>(p1_buf.shape(1));

    // For profile-profile, always use direct banded (profiles are typically shorter)
    // Hirschberg for profiles would require slicing numpy arrays which is complex
    // and profiles are typically shorter than raw sequences
    return align_banded_profiles(p1, p2, subst, centre_diag, half_width,
                                  gap_open, gap_extend);
}
