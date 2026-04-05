// profile_dp.cpp — Profile-profile versions of alignment functions.
// score(col_i, col_j) = einsum('a,b,ab->', p1[i], p2[j], subst)
// Used when aligning internal guide tree nodes (profiles vs profiles).
// The core profile-profile banded DP is already in banded_nw.cpp (align_banded_profiles).
// This file provides additional helpers for profile manipulation in C++.

// NOTE: This file is #included from band_doubling.cpp.
// All symbols from the include chain are already available.

// Compute substitution score between two profile columns
// p1: profile column of length alpha, p2: same, subst: alpha x alpha matrix
inline float profile_column_score(const float* p1, const float* p2,
                                   const float* subst, int alpha) {
    float s = 0.0f;
    for (int a = 0; a < alpha; ++a) {
        for (int b = 0; b < alpha; ++b) {
            s += p1[a] * p2[b] * subst[a * alpha + b];
        }
    }
    return s;
}

// Compute full score matrix between two profiles
// Returns: flat vector of size L1 * L2 (row-major)
std::vector<float> profile_score_matrix(
    const float* p1, int L1,
    const float* p2, int L2,
    const float* subst, int alpha)
{
    std::vector<float> scores(L1 * L2);
    for (int i = 0; i < L1; ++i) {
        for (int j = 0; j < L2; ++j) {
            scores[i * L2 + j] = profile_column_score(
                &p1[i * alpha], &p2[j * alpha], subst, alpha);
        }
    }
    return scores;
}

// Merge two aligned profiles into a new profile
// aln1: aligned columns from profile1 (with gap rows)
// aln2: aligned columns from profile2 (with gap rows)
// gap_char_idx: index of gap character in alphabet
// Returns: merged profile as flat vector, shape (aln_len, alpha)
std::vector<float> merge_profiles(
    const float* p1, int L1, int alpha1,
    const float* p2, int L2, int alpha2,
    const std::string& aligned1,  // gap pattern for p1 side
    const std::string& aligned2,  // gap pattern for p2 side
    int n_seqs1, int n_seqs2)
{
    int aln_len = static_cast<int>(aligned1.size());
    int alpha = alpha1;  // should be same
    int total_seqs = n_seqs1 + n_seqs2;

    std::vector<float> merged(aln_len * alpha, 0.0f);

    int pos1 = 0, pos2 = 0;
    for (int k = 0; k < aln_len; ++k) {
        bool gap1 = (aligned1[k] == '-');
        bool gap2 = (aligned2[k] == '-');

        if (!gap1 && pos1 < L1) {
            for (int a = 0; a < alpha; ++a) {
                merged[k * alpha + a] += p1[pos1 * alpha + a] * n_seqs1;
            }
            pos1++;
        } else {
            // Gap in profile1: add gap character weight
            if (alpha > 0) {
                merged[k * alpha + (alpha - 1)] += static_cast<float>(n_seqs1);
            }
        }

        if (!gap2 && pos2 < L2) {
            for (int a = 0; a < alpha; ++a) {
                merged[k * alpha + a] += p2[pos2 * alpha + a] * n_seqs2;
            }
            pos2++;
        } else {
            if (alpha > 0) {
                merged[k * alpha + (alpha - 1)] += static_cast<float>(n_seqs2);
            }
        }

        // Normalize
        float sum = 0.0f;
        for (int a = 0; a < alpha; ++a) sum += merged[k * alpha + a];
        if (sum > 0.0f) {
            for (int a = 0; a < alpha; ++a) merged[k * alpha + a] /= sum;
        }
    }

    return merged;
}

// Last row for profile banded DP (for Hirschberg on profiles)
// Returns the last row of the DP matrix (M values only) as a vector
std::vector<float> profile_banded_last_row(
    const float* p1, int L1,
    const float* p2, int L2,
    const float* subst, int alpha,
    int centre_diag, int half_width,
    float gap_open, float gap_extend)
{
    const int bw = 2 * half_width + 1;
    std::vector<float> prev(bw + 2, NEG_INF);
    std::vector<float> curr(bw + 2, NEG_INF);

    // Initialize
    {
        int bi = to_band_idx(0, 0, centre_diag, half_width);
        if (bi >= 0 && bi < bw + 2) prev[bi] = 0.0f;
    }
    {
        int jmax = band_j_max(0, centre_diag, half_width, L2);
        for (int j = 1; j <= jmax; ++j) {
            int bi = to_band_idx(j, 0, centre_diag, half_width);
            if (bi >= 0 && bi < bw + 2) prev[bi] = gap_open + gap_extend * j;
        }
    }

    for (int i = 1; i <= L1; ++i) {
        std::fill(curr.begin(), curr.end(), NEG_INF);
        int jmin = band_j_min(i, centre_diag, half_width, L2);
        int jmax = band_j_max(i, centre_diag, half_width, L2);

        if (0 >= jmin && 0 <= jmax) {
            int bi = to_band_idx(0, i, centre_diag, half_width);
            if (bi >= 0 && bi < bw + 2) curr[bi] = gap_open + gap_extend * i;
        }

        for (int j = std::max(1, jmin); j <= jmax; ++j) {
            int bi = to_band_idx(j, i, centre_diag, half_width);
            if (bi < 0 || bi >= bw + 2) continue;

            float s = profile_column_score(
                &p1[(i - 1) * alpha], &p2[(j - 1) * alpha], subst, alpha);

            int bd = to_band_idx(j - 1, i - 1, centre_diag, half_width);
            float diag = (bd >= 0 && bd < bw + 2) ? prev[bd] : NEG_INF;

            int bu = to_band_idx(j, i - 1, centre_diag, half_width);
            float up = (bu >= 0 && bu < bw + 2) ? prev[bu] : NEG_INF;

            int bl = to_band_idx(j - 1, i, centre_diag, half_width);
            float left = (bl >= 0 && bl < bw + 2) ? curr[bl] : NEG_INF;

            curr[bi] = std::max({diag + s,
                                 up + gap_open + gap_extend,
                                 left + gap_open + gap_extend});
        }

        std::swap(prev, curr);
    }

    return prev;
}
