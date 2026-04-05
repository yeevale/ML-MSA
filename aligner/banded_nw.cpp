// banded_nw.cpp — Core banded Needleman-Wunsch with affine gaps.
// Computes only cells within band: j in [i - centre - hw, i - centre + hw].
// Stores only band in memory: matrix (len1+1) x (2*hw+1), not full (len1+1)x(len2+1).
// Three matrices M, X, Y for affine gaps (Gotoh-Smith algorithm).
// Traceback tracks escape_left, escape_right, max_deviation.

#include "full_nw.cpp"

// ─── Helper: clamp j to valid range within band ───

inline int band_j_min(int i, int centre_diag, int half_width, int len2) {
    // j range: i - centre - hw to i - centre + hw, clamped to [0, len2]
    return std::max(0, i - centre_diag - half_width);
}

inline int band_j_max(int i, int centre_diag, int half_width, int len2) {
    return std::min(len2, i - centre_diag + half_width);
}

// Map absolute j to band-local index
inline int to_band_idx(int j, int i, int centre_diag, int half_width) {
    return j - (i - centre_diag - half_width);
}

// ─── Banded NW alignment ───

BandedResult align_banded(
    const std::string& seq1,
    const std::string& seq2,
    int   centre_diag,
    int   half_width,
    float gap_open,
    float gap_extend,
    bool  is_protein,
    const py::array_t<float>* subst_matrix)
{
    const int n = static_cast<int>(seq1.size());
    const int m = static_cast<int>(seq2.size());
    const int bw = 2 * half_width + 1;  // band width

    // Ensure half_width is at least 1
    half_width = std::max(half_width, 1);

    const float* subst_ptr = nullptr;
    int subst_cols = 0;
    if (subst_matrix && subst_matrix->size() > 0) {
        subst_ptr = subst_matrix->data();
        subst_cols = static_cast<int>(subst_matrix->unchecked<2>().shape(1));
    }

    auto score_fn = [&](int i, int j) -> float {
        int a = is_protein ? encode_protein(seq1[i]) : encode_dna(seq1[i]);
        int b = is_protein ? encode_protein(seq2[j]) : encode_dna(seq2[j]);
        if (a < 0 || b < 0) return -1.0f;
        if (subst_ptr && subst_cols > 0)
            return subst_ptr[a * subst_cols + b];
        return dna_subst_score(a, b);
    };

    // DP matrices: rows [0..n], columns stored as band-local indices [0..bw-1]
    // We store (n+1) rows of width bw for each of M, X, Y
    auto make_dp = [&]() {
        return std::vector<std::vector<float>>(n + 1, std::vector<float>(bw + 2, NEG_INF));
    };
    auto M = make_dp();
    auto X = make_dp();
    auto Y = make_dp();

    // Traceback: for each cell, store direction (encoded as int)
    // 3 bits: [matrix_source (0=M,1=X,2=Y)] for the three matrices
    auto make_tb = [&]() {
        return std::vector<std::vector<int>>(n + 1, std::vector<int>(bw + 2, -1));
    };
    auto tb_M = make_tb();
    auto tb_X = make_tb();
    auto tb_Y = make_tb();

    // Initialize (0,0)
    {
        int jmin = band_j_min(0, centre_diag, half_width, m);
        int jmax = band_j_max(0, centre_diag, half_width, m);
        if (0 >= jmin && 0 <= jmax) {
            int bi = to_band_idx(0, 0, centre_diag, half_width);
            if (bi >= 0 && bi < bw + 2) M[0][bi] = 0.0f;
        }
    }

    // Initialize first column (j=0, gaps in seq2)
    for (int i = 1; i <= n; ++i) {
        int jmin = band_j_min(i, centre_diag, half_width, m);
        int jmax = band_j_max(i, centre_diag, half_width, m);
        if (0 >= jmin && 0 <= jmax) {
            int bi = to_band_idx(0, i, centre_diag, half_width);
            if (bi >= 0 && bi < bw + 2) {
                X[i][bi] = gap_open + gap_extend * i;
                tb_X[i][bi] = (i == 1) ? 0 : 1;
            }
        }
    }

    // Initialize first row (i=0, gaps in seq1)
    {
        int jmin = band_j_min(0, centre_diag, half_width, m);
        int jmax = band_j_max(0, centre_diag, half_width, m);
        for (int j = std::max(1, jmin); j <= jmax; ++j) {
            int bi = to_band_idx(j, 0, centre_diag, half_width);
            if (bi >= 0 && bi < bw + 2) {
                Y[0][bi] = gap_open + gap_extend * j;
                tb_Y[0][bi] = (j == 1) ? 0 : 2;
            }
        }
    }

    // Fill DP within band
    for (int i = 1; i <= n; ++i) {
        int jmin = band_j_min(i, centre_diag, half_width, m);
        int jmax = band_j_max(i, centre_diag, half_width, m);

        for (int j = std::max(1, jmin); j <= jmax; ++j) {
            int bi = to_band_idx(j, i, centre_diag, half_width);
            if (bi < 0 || bi >= bw + 2) continue;

            float s = score_fn(i - 1, j - 1);

            // X[i][j] = max(M[i-1][j] + gap_open + gap_extend, X[i-1][j] + gap_extend)
            {
                int bi_prev_row = to_band_idx(j, i - 1, centre_diag, half_width);
                float x_from_m = NEG_INF, x_from_x = NEG_INF;
                if (bi_prev_row >= 0 && bi_prev_row < bw + 2) {
                    if (M[i - 1][bi_prev_row] > NEG_INF)
                        x_from_m = M[i - 1][bi_prev_row] + gap_open + gap_extend;
                    if (X[i - 1][bi_prev_row] > NEG_INF)
                        x_from_x = X[i - 1][bi_prev_row] + gap_extend;
                }
                if (x_from_m >= x_from_x) { X[i][bi] = x_from_m; tb_X[i][bi] = 0; }
                else                       { X[i][bi] = x_from_x; tb_X[i][bi] = 1; }
            }

            // Y[i][j] = max(M[i][j-1] + gap_open + gap_extend, Y[i][j-1] + gap_extend)
            {
                int bi_prev_col = to_band_idx(j - 1, i, centre_diag, half_width);
                float y_from_m = NEG_INF, y_from_y = NEG_INF;
                if (bi_prev_col >= 0 && bi_prev_col < bw + 2) {
                    if (M[i][bi_prev_col] > NEG_INF)
                        y_from_m = M[i][bi_prev_col] + gap_open + gap_extend;
                    if (Y[i][bi_prev_col] > NEG_INF)
                        y_from_y = Y[i][bi_prev_col] + gap_extend;
                }
                if (y_from_m >= y_from_y) { Y[i][bi] = y_from_m; tb_Y[i][bi] = 0; }
                else                       { Y[i][bi] = y_from_y; tb_Y[i][bi] = 2; }
            }

            // M[i][j] = max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1]) + s
            {
                int bi_diag = to_band_idx(j - 1, i - 1, centre_diag, half_width);
                float best_val = NEG_INF;
                int best_tb = 0;
                if (bi_diag >= 0 && bi_diag < bw + 2) {
                    if (M[i - 1][bi_diag] > best_val) { best_val = M[i - 1][bi_diag]; best_tb = 0; }
                    if (X[i - 1][bi_diag] > best_val) { best_val = X[i - 1][bi_diag]; best_tb = 1; }
                    if (Y[i - 1][bi_diag] > best_val) { best_val = Y[i - 1][bi_diag]; best_tb = 2; }
                }
                M[i][bi] = best_val + s;
                tb_M[i][bi] = best_tb;
            }
        }
    }

    // Score at (n, m)
    int bi_end = to_band_idx(m, n, centre_diag, half_width);
    float best_score = NEG_INF;
    int cur_mat = 0;
    bool reachable = bi_end >= 0 && bi_end < bw + 2;

    if (reachable) {
        best_score = M[n][bi_end]; cur_mat = 0;
        if (X[n][bi_end] > best_score) { best_score = X[n][bi_end]; cur_mat = 1; }
        if (Y[n][bi_end] > best_score) { best_score = Y[n][bi_end]; cur_mat = 2; }
    }

    // Traceback
    std::string aln1, aln2;
    bool escape_left = false, escape_right = false;
    int max_deviation = 0;
    int i = n, j = m;

    if (!reachable || best_score <= NEG_INF + 1e6f) {
        // Path cannot reach (n,m) within band → escaped
        return BandedResult{NEG_INF, "", "", true,
                            (n - m) < centre_diag - half_width,
                            (n - m) > centre_diag + half_width,
                            half_width + 1};
    }

    while (i > 0 || j > 0) {
        int dev = std::abs((i - j) - centre_diag);
        max_deviation = std::max(max_deviation, dev);

        // Check escape conditions at band boundary
        if (dev >= half_width) {
            if ((i - j) < centre_diag - half_width + 1) escape_left = true;
            if ((i - j) > centre_diag + half_width - 1) escape_right = true;
        }

        int bi_cur = to_band_idx(j, i, centre_diag, half_width);

        if (cur_mat == 0) {
            if (i > 0 && j > 0) {
                int prev = (bi_cur >= 0 && bi_cur < bw + 2) ? tb_M[i][bi_cur] : 0;
                aln1 += seq1[i - 1];
                aln2 += seq2[j - 1];
                --i; --j;
                cur_mat = prev;
            } else if (i > 0) {
                aln1 += seq1[--i]; aln2 += '-'; cur_mat = 1;
            } else {
                aln1 += '-'; aln2 += seq2[--j]; cur_mat = 2;
            }
        } else if (cur_mat == 1) {
            if (i > 0) {
                int prev = (bi_cur >= 0 && bi_cur < bw + 2) ? tb_X[i][bi_cur] : 0;
                aln1 += seq1[i - 1]; aln2 += '-';
                --i;
                cur_mat = prev;
            } else {
                aln1 += '-'; aln2 += seq2[--j]; cur_mat = 2;
            }
        } else {
            if (j > 0) {
                int prev = (bi_cur >= 0 && bi_cur < bw + 2) ? tb_Y[i][bi_cur] : 0;
                aln1 += '-'; aln2 += seq2[j - 1];
                --j;
                cur_mat = (prev == 2) ? 2 : 0;
            } else {
                aln1 += seq1[--i]; aln2 += '-'; cur_mat = 1;
            }
        }
    }

    std::reverse(aln1.begin(), aln1.end());
    std::reverse(aln2.begin(), aln2.end());

    bool path_esc = escape_left || escape_right || (max_deviation >= half_width);

    return BandedResult{best_score, aln1, aln2, path_esc, escape_left, escape_right, max_deviation};
}

// ─── Profile-profile banded alignment ───

BandedResult align_banded_profiles(
    const py::array_t<float>& profile1,
    const py::array_t<float>& profile2,
    const py::array_t<float>& subst,
    int   centre_diag,
    int   half_width,
    float gap_open,
    float gap_extend)
{
    auto p1 = profile1.unchecked<2>();
    auto p2 = profile2.unchecked<2>();
    auto sm = subst.unchecked<2>();

    const int n = static_cast<int>(p1.shape(0));
    const int m = static_cast<int>(p2.shape(0));
    const int alpha = static_cast<int>(p1.shape(1));
    const int bw = 2 * half_width + 1;

    half_width = std::max(half_width, 1);

    // score(i, j) = sum_a sum_b p1[i,a] * p2[j,b] * subst[a,b]
    auto score_fn = [&](int i, int j) -> float {
        float s = 0.0f;
        for (int a = 0; a < alpha; ++a)
            for (int b = 0; b < alpha; ++b)
                s += p1(i, a) * p2(j, b) * sm(a, b);
        return s;
    };

    auto make_dp = [&]() {
        return std::vector<std::vector<float>>(n + 1, std::vector<float>(bw + 2, NEG_INF));
    };
    auto M = make_dp(), X_ = make_dp(), Y_ = make_dp();

    // Initialize
    {
        int bi0 = to_band_idx(0, 0, centre_diag, half_width);
        if (bi0 >= 0 && bi0 < bw + 2) M[0][bi0] = 0.0f;
    }
    for (int i = 1; i <= n; ++i) {
        int jmin = band_j_min(i, centre_diag, half_width, m);
        int jmax = band_j_max(i, centre_diag, half_width, m);
        if (0 >= jmin && 0 <= jmax) {
            int bi = to_band_idx(0, i, centre_diag, half_width);
            if (bi >= 0 && bi < bw + 2) X_[i][bi] = gap_open + gap_extend * i;
        }
    }
    {
        int jmin = band_j_min(0, centre_diag, half_width, m);
        int jmax = band_j_max(0, centre_diag, half_width, m);
        for (int j = std::max(1, jmin); j <= jmax; ++j) {
            int bi = to_band_idx(j, 0, centre_diag, half_width);
            if (bi >= 0 && bi < bw + 2) Y_[0][bi] = gap_open + gap_extend * j;
        }
    }

    // Fill DP
    for (int i = 1; i <= n; ++i) {
        int jmin = band_j_min(i, centre_diag, half_width, m);
        int jmax = band_j_max(i, centre_diag, half_width, m);

        for (int j = std::max(1, jmin); j <= jmax; ++j) {
            int bi = to_band_idx(j, i, centre_diag, half_width);
            if (bi < 0 || bi >= bw + 2) continue;

            float s = score_fn(i - 1, j - 1);

            // X
            {
                int bip = to_band_idx(j, i - 1, centre_diag, half_width);
                float xm = NEG_INF, xx = NEG_INF;
                if (bip >= 0 && bip < bw + 2) {
                    if (M[i-1][bip] > NEG_INF) xm = M[i-1][bip] + gap_open + gap_extend;
                    if (X_[i-1][bip] > NEG_INF) xx = X_[i-1][bip] + gap_extend;
                }
                X_[i][bi] = std::max(xm, xx);
            }
            // Y
            {
                int bip = to_band_idx(j-1, i, centre_diag, half_width);
                float ym = NEG_INF, yy = NEG_INF;
                if (bip >= 0 && bip < bw + 2) {
                    if (M[i][bip] > NEG_INF) ym = M[i][bip] + gap_open + gap_extend;
                    if (Y_[i][bip] > NEG_INF) yy = Y_[i][bip] + gap_extend;
                }
                Y_[i][bi] = std::max(ym, yy);
            }
            // M
            {
                int bid = to_band_idx(j-1, i-1, centre_diag, half_width);
                float best = NEG_INF;
                if (bid >= 0 && bid < bw + 2) {
                    best = std::max({M[i-1][bid], X_[i-1][bid], Y_[i-1][bid]});
                }
                M[i][bi] = best + s;
            }
        }
    }

    // Score
    int bi_end = to_band_idx(m, n, centre_diag, half_width);
    bool reachable = bi_end >= 0 && bi_end < bw + 2;
    float best_score = NEG_INF;
    if (reachable)
        best_score = std::max({M[n][bi_end], X_[n][bi_end], Y_[n][bi_end]});

    if (!reachable || best_score <= NEG_INF + 1e6f) {
        return BandedResult{NEG_INF, "", "", true,
                            (n - m) < centre_diag - half_width,
                            (n - m) > centre_diag + half_width,
                            half_width + 1};
    }

    // For profile-profile, return empty alignment strings (only score matters)
    bool el = false, er = false;
    int bi_check = to_band_idx(m, n, centre_diag, half_width);
    int dev = std::abs((n - m) - centre_diag);
    bool esc = dev >= half_width;

    return BandedResult{best_score, "", "", esc,
                        (n - m) < centre_diag,
                        (n - m) > centre_diag,
                        dev};
}
