// full_nw.cpp — Reference full Needleman-Wunsch with affine gaps (Gotoh-Smith).
// Used ONLY for: (1) verification of banded NW, (2) generating training data (traceback paths).
// NOT used in the production MSA pipeline.

#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

constexpr float NEG_INF = -1e9f;

// ─── Structs shared across all aligner files ───

struct BandedResult {
    float       score;
    std::string aligned_seq1;
    std::string aligned_seq2;
    bool        path_escaped;
    bool        escape_left;
    bool        escape_right;
    int         max_deviation;
};

struct DoublingResult {
    BandedResult alignment;
    int  n_doublings;
    int  final_left_bound;
    int  final_right_bound;
    bool used_hirschberg;
    bool used_four_russians;
    bool used_simd;
};

// ─── Character encoders ───

inline int encode_dna(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return -1;  // N or unknown
    }
}

inline int encode_protein(char c) {
    static const char aa[] = "ACDEFGHIKLMNPQRSTVWY";
    for (int i = 0; i < 20; ++i)
        if (aa[i] == c || aa[i] == (c & ~32)) return i;
    return -1;  // X or unknown
}

// Default DNA substitution: match=+1, mismatch=-1
inline float dna_subst_score(int a, int b) {
    if (a < 0 || b < 0) return -1.0f;
    return (a == b) ? 1.0f : -1.0f;
}

// ─── Full NW alignment (O(n*m) time and memory) ───

BandedResult full_nw_align(
    const std::string& seq1,
    const std::string& seq2,
    float gap_open,
    float gap_extend,
    bool  is_protein,
    const py::array_t<float>* subst_matrix)
{
    const int n = static_cast<int>(seq1.size());
    const int m = static_cast<int>(seq2.size());

    // Access substitution matrix if provided
    const float* subst_ptr = nullptr;
    int subst_cols = 0;
    if (subst_matrix && subst_matrix->size() > 0) {
        auto buf = subst_matrix->unchecked<2>();
        subst_ptr = subst_matrix->data();
        subst_cols = static_cast<int>(buf.shape(1));
    }

    auto score_fn = [&](int i, int j) -> float {
        int a = is_protein ? encode_protein(seq1[i]) : encode_dna(seq1[i]);
        int b = is_protein ? encode_protein(seq2[j]) : encode_dna(seq2[j]);
        if (a < 0 || b < 0) return -1.0f;
        if (subst_ptr && subst_cols > 0) {
            return subst_ptr[a * subst_cols + b];
        }
        return dna_subst_score(a, b);
    };

    // Three DP matrices: M (match/mismatch), X (gap in seq2), Y (gap in seq1)
    // M[i][j] = best score ending with seq1[i-1] aligned to seq2[j-1]
    // X[i][j] = best score ending with a gap in seq2 (insertion in seq1)
    // Y[i][j] = best score ending with a gap in seq1 (insertion in seq2)
    std::vector<std::vector<float>> M(n + 1, std::vector<float>(m + 1, NEG_INF));
    std::vector<std::vector<float>> X(n + 1, std::vector<float>(m + 1, NEG_INF));
    std::vector<std::vector<float>> Y(n + 1, std::vector<float>(m + 1, NEG_INF));

    // Traceback direction: 0=diag(M), 1=up(X), 2=left(Y)
    // For each matrix, store where it came from
    // tb_M[i][j]: 0 = from M[i-1][j-1], 1 = from X[i-1][j-1], 2 = from Y[i-1][j-1]
    // tb_X[i][j]: 0 = from M[i-1][j](open), 1 = from X[i-1][j](extend)
    // tb_Y[i][j]: 0 = from M[i][j-1](open), 2 = from Y[i][j-1](extend)
    std::vector<std::vector<int>> tb_M(n + 1, std::vector<int>(m + 1, -1));
    std::vector<std::vector<int>> tb_X(n + 1, std::vector<int>(m + 1, -1));
    std::vector<std::vector<int>> tb_Y(n + 1, std::vector<int>(m + 1, -1));

    M[0][0] = 0.0f;
    // Initialize first column (gaps in seq2)
    for (int i = 1; i <= n; ++i) {
        X[i][0] = gap_open + gap_extend * i;
        tb_X[i][0] = (i == 1) ? 0 : 1;
    }
    // Initialize first row (gaps in seq1)
    for (int j = 1; j <= m; ++j) {
        Y[0][j] = gap_open + gap_extend * j;
        tb_Y[0][j] = (j == 1) ? 0 : 2;
    }

    // Fill DP
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            float s = score_fn(i - 1, j - 1);

            // X[i][j] = max(M[i-1][j] + gap_open + gap_extend, X[i-1][j] + gap_extend)
            float x_from_m = M[i - 1][j] + gap_open + gap_extend;
            float x_from_x = X[i - 1][j] + gap_extend;
            if (x_from_m >= x_from_x) {
                X[i][j] = x_from_m;
                tb_X[i][j] = 0;
            } else {
                X[i][j] = x_from_x;
                tb_X[i][j] = 1;
            }

            // Y[i][j] = max(M[i][j-1] + gap_open + gap_extend, Y[i][j-1] + gap_extend)
            float y_from_m = M[i][j - 1] + gap_open + gap_extend;
            float y_from_y = Y[i][j - 1] + gap_extend;
            if (y_from_m >= y_from_y) {
                Y[i][j] = y_from_m;
                tb_Y[i][j] = 0;
            } else {
                Y[i][j] = y_from_y;
                tb_Y[i][j] = 2;
            }

            // M[i][j] = max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1]) + s
            float m_from_m = M[i - 1][j - 1];
            float m_from_x = X[i - 1][j - 1];
            float m_from_y = Y[i - 1][j - 1];
            float best = m_from_m;
            int best_tb = 0;
            if (m_from_x > best) { best = m_from_x; best_tb = 1; }
            if (m_from_y > best) { best = m_from_y; best_tb = 2; }
            M[i][j] = best + s;
            tb_M[i][j] = best_tb;
        }
    }

    // Find best score at (n, m) across all three matrices
    float best_score = M[n][m];
    int cur_mat = 0;  // 0=M, 1=X, 2=Y
    if (X[n][m] > best_score) { best_score = X[n][m]; cur_mat = 1; }
    if (Y[n][m] > best_score) { best_score = Y[n][m]; cur_mat = 2; }

    // Traceback
    std::string aln1, aln2;
    int i = n, j = m;

    while (i > 0 || j > 0) {
        if (cur_mat == 0) {
            // In M matrix — diagonal move
            if (i == 0 || j == 0) {
                // Edge case: move along border
                if (i > 0) { aln1 += seq1[--i]; aln2 += '-'; cur_mat = 1; }
                else       { aln1 += '-'; aln2 += seq2[--j]; cur_mat = 2; }
            } else {
                int prev = tb_M[i][j];
                aln1 += seq1[i - 1];
                aln2 += seq2[j - 1];
                --i; --j;
                cur_mat = prev;
            }
        } else if (cur_mat == 1) {
            // In X matrix — gap in seq2 (move up)
            if (i == 0) {
                aln1 += '-'; aln2 += seq2[--j]; cur_mat = 2;
            } else {
                int prev = tb_X[i][j];
                aln1 += seq1[i - 1];
                aln2 += '-';
                --i;
                cur_mat = prev;
            }
        } else {
            // In Y matrix — gap in seq1 (move left)
            if (j == 0) {
                aln1 += seq1[--i]; aln2 += '-'; cur_mat = 1;
            } else {
                int prev = tb_Y[i][j];
                aln1 += '-';
                aln2 += seq2[j - 1];
                --j;
                cur_mat = (prev == 2) ? 2 : 0;
            }
        }
    }

    std::reverse(aln1.begin(), aln1.end());
    std::reverse(aln2.begin(), aln2.end());

    return BandedResult{best_score, aln1, aln2, false, false, false, 0};
}

// ─── Full NW traceback: returns path as list of (i,j) ───

std::vector<std::pair<int, int>> full_nw_traceback(
    const std::string& seq1,
    const std::string& seq2,
    float gap_open,
    float gap_extend,
    bool  is_protein,
    const py::array_t<float>* subst_matrix)
{
    const int n = static_cast<int>(seq1.size());
    const int m = static_cast<int>(seq2.size());

    const float* subst_ptr = nullptr;
    int subst_cols = 0;
    if (subst_matrix && subst_matrix->size() > 0) {
        auto buf = subst_matrix->unchecked<2>();
        subst_ptr = subst_matrix->data();
        subst_cols = static_cast<int>(buf.shape(1));
    }

    auto score_fn = [&](int i, int j) -> float {
        int a = is_protein ? encode_protein(seq1[i]) : encode_dna(seq1[i]);
        int b = is_protein ? encode_protein(seq2[j]) : encode_dna(seq2[j]);
        if (a < 0 || b < 0) return -1.0f;
        if (subst_ptr && subst_cols > 0)
            return subst_ptr[a * subst_cols + b];
        return dna_subst_score(a, b);
    };

    // Three DP matrices
    std::vector<std::vector<float>> M(n + 1, std::vector<float>(m + 1, NEG_INF));
    std::vector<std::vector<float>> X(n + 1, std::vector<float>(m + 1, NEG_INF));
    std::vector<std::vector<float>> Y(n + 1, std::vector<float>(m + 1, NEG_INF));
    std::vector<std::vector<int>> tb_M(n + 1, std::vector<int>(m + 1, -1));
    std::vector<std::vector<int>> tb_X(n + 1, std::vector<int>(m + 1, -1));
    std::vector<std::vector<int>> tb_Y(n + 1, std::vector<int>(m + 1, -1));

    M[0][0] = 0.0f;
    for (int i = 1; i <= n; ++i) {
        X[i][0] = gap_open + gap_extend * i;
        tb_X[i][0] = (i == 1) ? 0 : 1;
    }
    for (int j = 1; j <= m; ++j) {
        Y[0][j] = gap_open + gap_extend * j;
        tb_Y[0][j] = (j == 1) ? 0 : 2;
    }

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            float s = score_fn(i - 1, j - 1);

            float x_from_m = M[i - 1][j] + gap_open + gap_extend;
            float x_from_x = X[i - 1][j] + gap_extend;
            if (x_from_m >= x_from_x) { X[i][j] = x_from_m; tb_X[i][j] = 0; }
            else                       { X[i][j] = x_from_x; tb_X[i][j] = 1; }

            float y_from_m = M[i][j - 1] + gap_open + gap_extend;
            float y_from_y = Y[i][j - 1] + gap_extend;
            if (y_from_m >= y_from_y) { Y[i][j] = y_from_m; tb_Y[i][j] = 0; }
            else                       { Y[i][j] = y_from_y; tb_Y[i][j] = 2; }

            float best = M[i - 1][j - 1]; int bt = 0;
            if (X[i - 1][j - 1] > best) { best = X[i - 1][j - 1]; bt = 1; }
            if (Y[i - 1][j - 1] > best) { best = Y[i - 1][j - 1]; bt = 2; }
            M[i][j] = best + s;
            tb_M[i][j] = bt;
        }
    }

    // Determine starting matrix
    float best_score = M[n][m]; int cur_mat = 0;
    if (X[n][m] > best_score) { best_score = X[n][m]; cur_mat = 1; }
    if (Y[n][m] > best_score) { best_score = Y[n][m]; cur_mat = 2; }

    // Traceback — collect (i,j) positions
    std::vector<std::pair<int, int>> path;
    int i = n, j = m;
    path.push_back({i, j});

    while (i > 0 || j > 0) {
        if (cur_mat == 0) {
            if (i > 0 && j > 0) {
                int prev = tb_M[i][j];
                --i; --j;
                cur_mat = prev;
            } else if (i > 0) {
                --i; cur_mat = 1;
            } else {
                --j; cur_mat = 2;
            }
        } else if (cur_mat == 1) {
            if (i > 0) {
                int prev = tb_X[i][j];
                --i;
                cur_mat = prev;
            } else {
                --j; cur_mat = 2;
            }
        } else {
            if (j > 0) {
                int prev = tb_Y[i][j];
                --j;
                cur_mat = (prev == 2) ? 2 : 0;
            } else {
                --i; cur_mat = 1;
            }
        }
        path.push_back({i, j});
    }

    std::reverse(path.begin(), path.end());
    return path;
}
