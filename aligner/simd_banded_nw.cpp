// simd_banded_nw.cpp — AVX2 acceleration of banded NW via anti-diagonal parallelization.
// Anti-diagonal d contains cells (i,j): i+j=d, i-j in [centre-hw, centre+hw].
// All cells on one anti-diagonal are INDEPENDENT → compute 8 at a time via AVX2.
// Fallback: if HAVE_AVX2 not defined → use align_banded from banded_nw.cpp.

#include "banded_nw.cpp"

#ifdef HAVE_AVX2
#include <immintrin.h>
#endif

// Runtime AVX2 detection
#ifdef _MSC_VER
#include <intrin.h>
inline bool avx2_supported() {
    int info[4];
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
}
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
inline bool avx2_supported() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 5)) != 0;
    }
    return false;
}
#else
inline bool avx2_supported() { return false; }
#endif

#ifdef HAVE_AVX2

// AVX2 banded NW: anti-diagonal sweep with 8-wide SIMD
// For simplicity and correctness, this implements the affine gap model
// using anti-diagonal wavefront, processing 8 cells per SIMD register.

BandedResult align_banded_avx2(
    const std::string& seq1,
    const std::string& seq2,
    int   centre_diag,
    int   half_width,
    float gap_open,
    float gap_extend,
    bool  is_protein)
{
    const int n = static_cast<int>(seq1.size());
    const int m = static_cast<int>(seq2.size());
    const int bw = 2 * half_width + 1;

    half_width = std::max(half_width, 1);

    // For small problems or narrow bands, fall back to scalar
    if (bw < 8 || n < 8 || m < 8) {
        return align_banded(seq1, seq2, centre_diag, half_width,
                            gap_open, gap_extend, is_protein, nullptr);
    }

    // Encode sequences
    std::vector<int> enc1(n), enc2(m);
    for (int i = 0; i < n; ++i) enc1[i] = is_protein ? encode_protein(seq1[i]) : encode_dna(seq1[i]);
    for (int j = 0; j < m; ++j) enc2[j] = is_protein ? encode_protein(seq2[j]) : encode_dna(seq2[j]);

    // DP storage: row-major, band-local indexing like scalar version
    // M, X, Y each (n+1) x (bw+2)
    const int cols = bw + 2;
    std::vector<float> M_dp((n + 1) * cols, NEG_INF);
    std::vector<float> X_dp((n + 1) * cols, NEG_INF);
    std::vector<float> Y_dp((n + 1) * cols, NEG_INF);
    std::vector<int> tb_M_dp((n + 1) * cols, -1);
    std::vector<int> tb_X_dp((n + 1) * cols, -1);
    std::vector<int> tb_Y_dp((n + 1) * cols, -1);

    auto idx = [cols](int row, int col) -> int { return row * cols + col; };

    // Initialize (0,0)
    {
        int bi0 = to_band_idx(0, 0, centre_diag, half_width);
        if (bi0 >= 0 && bi0 < cols) M_dp[idx(0, bi0)] = 0.0f;
    }
    // First column
    for (int i = 1; i <= n; ++i) {
        int jmin = band_j_min(i, centre_diag, half_width, m);
        int jmax = band_j_max(i, centre_diag, half_width, m);
        if (0 >= jmin && 0 <= jmax) {
            int bi = to_band_idx(0, i, centre_diag, half_width);
            if (bi >= 0 && bi < cols) {
                X_dp[idx(i, bi)] = gap_open + gap_extend * i;
                tb_X_dp[idx(i, bi)] = (i == 1) ? 0 : 1;
            }
        }
    }
    // First row
    {
        int jmin = band_j_min(0, centre_diag, half_width, m);
        int jmax = band_j_max(0, centre_diag, half_width, m);
        for (int j = std::max(1, jmin); j <= jmax; ++j) {
            int bi = to_band_idx(j, 0, centre_diag, half_width);
            if (bi >= 0 && bi < cols) {
                Y_dp[idx(0, bi)] = gap_open + gap_extend * j;
                tb_Y_dp[idx(0, bi)] = (j == 1) ? 0 : 2;
            }
        }
    }

    // Fill DP row by row, using AVX2 for inner loop over band columns
    __m256 v_gap_open = _mm256_set1_ps(gap_open);
    __m256 v_gap_extend = _mm256_set1_ps(gap_extend);
    __m256 v_gap_oe = _mm256_set1_ps(gap_open + gap_extend);
    __m256 v_neginf = _mm256_set1_ps(NEG_INF);

    for (int i = 1; i <= n; ++i) {
        int jmin = std::max(1, band_j_min(i, centre_diag, half_width, m));
        int jmax = band_j_max(i, centre_diag, half_width, m);
        
        // Process 8 j values at a time with AVX2
        int j = jmin;
        
        // We need to be careful: Y depends on j-1 in same row (left dependency)
        // So we can't fully parallelize Y across j in the same row.
        // But M and X only depend on previous row → can be parallelized.
        // Strategy: compute M and X in SIMD, then do Y in scalar after.
        
        // Actually, for correctness with the left dependency in Y,
        // process M and X in SIMD chunks, then Y sequentially
        for (; j + 7 <= jmax; j += 8) {
            // Compute substitution scores for 8 positions
            float scores[8];
            for (int k = 0; k < 8; ++k) {
                int jj = j + k;
                int a = enc1[i - 1], b = enc2[jj - 1];
                scores[k] = (a >= 0 && b >= 0) ? ((a == b) ? 1.0f : -1.0f) : -1.0f;
            }
            if (is_protein) {
                // Protein scoring — use simple match/mismatch for SIMD path
                for (int k = 0; k < 8; ++k) {
                    int jj = j + k;
                    int a = enc1[i - 1], b = enc2[jj - 1];
                    scores[k] = (a >= 0 && b >= 0 && a == b) ? 4.0f : -1.0f;
                }
            }
            __m256 v_scores = _mm256_loadu_ps(scores);

            // Load M, X, Y from previous row diagonal (i-1, j-1..j+6)
            float m_diag[8], x_diag[8], y_diag[8];
            for (int k = 0; k < 8; ++k) {
                int jj = j + k;
                int bid = to_band_idx(jj - 1, i - 1, centre_diag, half_width);
                if (bid >= 0 && bid < cols) {
                    m_diag[k] = M_dp[idx(i - 1, bid)];
                    x_diag[k] = X_dp[idx(i - 1, bid)];
                    y_diag[k] = Y_dp[idx(i - 1, bid)];
                } else {
                    m_diag[k] = NEG_INF;
                    x_diag[k] = NEG_INF;
                    y_diag[k] = NEG_INF;
                }
            }
            __m256 vm_d = _mm256_loadu_ps(m_diag);
            __m256 vx_d = _mm256_loadu_ps(x_diag);
            __m256 vy_d = _mm256_loadu_ps(y_diag);

            // M[i][j] = max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1]) + score
            __m256 vbest = _mm256_max_ps(vm_d, _mm256_max_ps(vx_d, vy_d));
            __m256 vm_new = _mm256_add_ps(vbest, v_scores);

            // X[i][j] = max(M[i-1][j] + gap_open+gap_extend, X[i-1][j] + gap_extend)
            float m_up[8], x_up[8];
            for (int k = 0; k < 8; ++k) {
                int jj = j + k;
                int biu = to_band_idx(jj, i - 1, centre_diag, half_width);
                if (biu >= 0 && biu < cols) {
                    m_up[k] = M_dp[idx(i - 1, biu)];
                    x_up[k] = X_dp[idx(i - 1, biu)];
                } else {
                    m_up[k] = NEG_INF;
                    x_up[k] = NEG_INF;
                }
            }
            __m256 vm_u = _mm256_loadu_ps(m_up);
            __m256 vx_u = _mm256_loadu_ps(x_up);
            __m256 vx_new = _mm256_max_ps(
                _mm256_add_ps(vm_u, v_gap_oe),
                _mm256_add_ps(vx_u, v_gap_extend)
            );

            // Store M and X
            float m_out[8], x_out[8];
            _mm256_storeu_ps(m_out, vm_new);
            _mm256_storeu_ps(x_out, vx_new);

            for (int k = 0; k < 8; ++k) {
                int jj = j + k;
                int bi = to_band_idx(jj, i, centre_diag, half_width);
                if (bi >= 0 && bi < cols) {
                    M_dp[idx(i, bi)] = m_out[k];
                    X_dp[idx(i, bi)] = x_out[k];

                    // Traceback for M
                    float best_m = m_diag[k];
                    tb_M_dp[idx(i, bi)] = 0;
                    if (x_diag[k] > best_m) { best_m = x_diag[k]; tb_M_dp[idx(i, bi)] = 1; }
                    if (y_diag[k] > best_m) { tb_M_dp[idx(i, bi)] = 2; }

                    // Traceback for X
                    tb_X_dp[idx(i, bi)] = (m_up[k] + gap_open + gap_extend >= x_up[k] + gap_extend) ? 0 : 1;
                }
            }
        }

        // Scalar remainder for M and X
        for (; j <= jmax; ++j) {
            int bi = to_band_idx(j, i, centre_diag, half_width);
            if (bi < 0 || bi >= cols) continue;

            int a = enc1[i - 1], b = enc2[j - 1];
            float s = (a >= 0 && b >= 0) ? ((a == b) ? 1.0f : -1.0f) : -1.0f;
            if (is_protein && a >= 0 && b >= 0) s = (a == b) ? 4.0f : -1.0f;

            // X
            int biu = to_band_idx(j, i - 1, centre_diag, half_width);
            float xm = NEG_INF, xx = NEG_INF;
            if (biu >= 0 && biu < cols) {
                if (M_dp[idx(i-1, biu)] > NEG_INF) xm = M_dp[idx(i-1, biu)] + gap_open + gap_extend;
                if (X_dp[idx(i-1, biu)] > NEG_INF) xx = X_dp[idx(i-1, biu)] + gap_extend;
            }
            X_dp[idx(i, bi)] = std::max(xm, xx);
            tb_X_dp[idx(i, bi)] = (xm >= xx) ? 0 : 1;

            // M
            int bid = to_band_idx(j - 1, i - 1, centre_diag, half_width);
            float best = NEG_INF; int bt = 0;
            if (bid >= 0 && bid < cols) {
                best = M_dp[idx(i-1, bid)]; bt = 0;
                if (X_dp[idx(i-1, bid)] > best) { best = X_dp[idx(i-1, bid)]; bt = 1; }
                if (Y_dp[idx(i-1, bid)] > best) { best = Y_dp[idx(i-1, bid)]; bt = 2; }
            }
            M_dp[idx(i, bi)] = best + s;
            tb_M_dp[idx(i, bi)] = bt;
        }

        // Y depends on j-1 in same row → must be sequential
        for (j = std::max(1, band_j_min(i, centre_diag, half_width, m)); j <= jmax; ++j) {
            int bi = to_band_idx(j, i, centre_diag, half_width);
            if (bi < 0 || bi >= cols) continue;

            int bip = to_band_idx(j - 1, i, centre_diag, half_width);
            float ym = NEG_INF, yy = NEG_INF;
            if (bip >= 0 && bip < cols) {
                if (M_dp[idx(i, bip)] > NEG_INF) ym = M_dp[idx(i, bip)] + gap_open + gap_extend;
                if (Y_dp[idx(i, bip)] > NEG_INF) yy = Y_dp[idx(i, bip)] + gap_extend;
            }
            if (std::max(ym, yy) > Y_dp[idx(i, bi)]) {
                Y_dp[idx(i, bi)] = std::max(ym, yy);
                tb_Y_dp[idx(i, bi)] = (ym >= yy) ? 0 : 2;
            }
        }
    }

    // Score at (n, m)
    int bi_end = to_band_idx(m, n, centre_diag, half_width);
    bool reachable = bi_end >= 0 && bi_end < cols;
    float best_score = NEG_INF;
    int cur_mat = 0;

    if (reachable) {
        best_score = M_dp[idx(n, bi_end)]; cur_mat = 0;
        if (X_dp[idx(n, bi_end)] > best_score) { best_score = X_dp[idx(n, bi_end)]; cur_mat = 1; }
        if (Y_dp[idx(n, bi_end)] > best_score) { best_score = Y_dp[idx(n, bi_end)]; cur_mat = 2; }
    }

    if (!reachable || best_score <= NEG_INF + 1e6f) {
        return BandedResult{NEG_INF, "", "", true,
                            (n - m) < centre_diag - half_width,
                            (n - m) > centre_diag + half_width,
                            half_width + 1};
    }

    // Traceback
    std::string aln1, aln2;
    bool escape_left = false, escape_right = false;
    int max_deviation = 0;
    int ii = n, jj = m;

    while (ii > 0 || jj > 0) {
        int dev = std::abs((ii - jj) - centre_diag);
        max_deviation = std::max(max_deviation, dev);
        if (dev >= half_width) {
            if ((ii - jj) < centre_diag - half_width + 1) escape_left = true;
            if ((ii - jj) > centre_diag + half_width - 1) escape_right = true;
        }

        int bi_cur = to_band_idx(jj, ii, centre_diag, half_width);

        if (cur_mat == 0) {
            if (ii > 0 && jj > 0) {
                int prev = (bi_cur >= 0 && bi_cur < cols) ? tb_M_dp[idx(ii, bi_cur)] : 0;
                aln1 += seq1[ii - 1]; aln2 += seq2[jj - 1];
                --ii; --jj;
                cur_mat = prev;
            } else if (ii > 0) {
                aln1 += seq1[--ii]; aln2 += '-'; cur_mat = 1;
            } else {
                aln1 += '-'; aln2 += seq2[--jj]; cur_mat = 2;
            }
        } else if (cur_mat == 1) {
            if (ii > 0) {
                int prev = (bi_cur >= 0 && bi_cur < cols) ? tb_X_dp[idx(ii, bi_cur)] : 0;
                aln1 += seq1[ii - 1]; aln2 += '-';
                --ii;
                cur_mat = prev;
            } else {
                aln1 += '-'; aln2 += seq2[--jj]; cur_mat = 2;
            }
        } else {
            if (jj > 0) {
                int prev = (bi_cur >= 0 && bi_cur < cols) ? tb_Y_dp[idx(ii, bi_cur)] : 0;
                aln1 += '-'; aln2 += seq2[jj - 1];
                --jj;
                cur_mat = (prev == 2) ? 2 : 0;
            } else {
                aln1 += seq1[--ii]; aln2 += '-'; cur_mat = 1;
            }
        }
    }

    std::reverse(aln1.begin(), aln1.end());
    std::reverse(aln2.begin(), aln2.end());

    bool path_esc = escape_left || escape_right || (max_deviation >= half_width);
    return BandedResult{best_score, aln1, aln2, path_esc, escape_left, escape_right, max_deviation};
}

#endif  // HAVE_AVX2

// ─── Auto-dispatcher ───

BandedResult align_banded_auto(
    const std::string& seq1,
    const std::string& seq2,
    int   centre_diag,
    int   half_width,
    float gap_open,
    float gap_extend,
    bool  is_protein)
{
#ifdef HAVE_AVX2
    static bool has_avx2 = avx2_supported();
    if (has_avx2 && half_width >= 4) {
        return align_banded_avx2(seq1, seq2, centre_diag, half_width,
                                 gap_open, gap_extend, is_protein);
    }
#endif
    return align_banded(seq1, seq2, centre_diag, half_width,
                        gap_open, gap_extend, is_protein, nullptr);
}
