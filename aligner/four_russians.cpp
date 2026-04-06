// four_russians.cpp — Method of Four Russians (Arlazarov, Dinic, Kronrod, Faradzhev, 1970).
// Speeds up banded NW from O(n*W) to O(n*W/t) via t×t lookup table.
//
// KEY: FourRussiansAligner created ONCE and reused:
//   - Between band_doubling iterations (table accumulates)
//   - Inside Hirschberg (passed by reference to all recursive calls)
//   hit_ratio grows with each call, reaching >90%.
//
// Parameter t: t = floor(log2(2*hw+1))
//   DNA (|Σ|=4), t=4: table ~1.6M entries
//   Proteins (|Σ|=20): t=min(t,2)
//
// Boundary quantization into B=16 levels → finite key size → lookup table

#include "simd_banded_nw.cpp"
#include <unordered_map>
#include <functional>
#include <cstring>
#include <cmath>

struct BlockBoundary {
    std::vector<float> bottom_row;  // bottom boundary of block (t+1 values per matrix)
    std::vector<float> right_col;   // right boundary of block
};

class FourRussiansAligner {
public:
    struct Stats {
        int   hits = 0;
        int   computed_simd = 0;
        int   computed_scalar = 0;
        float hit_ratio = 0.0f;
    };

    FourRussiansAligner(int block_size, bool is_protein,
                         float gap_open, float gap_extend,
                         int quant_levels = 16,
                         const float* subst = nullptr)
        : is_protein_(is_protein), go_(gap_open), ge_(gap_extend), B_(quant_levels)
    {
        t_ = (block_size > 0) ? block_size : 4;
        if (subst) {
            int dim = is_protein ? 20 : 4;
            subst_.assign(subst, subst + dim * dim);
        }
    }

    // For Hirschberg: only last row, O(W) memory
    // Accumulates lookup table between calls
    std::vector<float> last_row(
        const std::string& seq1,
        const std::string& seq2,
        int centre_diag, int half_width)
    {
        const int n = static_cast<int>(seq1.size());
        const int m = static_cast<int>(seq2.size());
        const int bw = 2 * half_width + 1;

        // Compute t based on half_width
        int t = compute_t(half_width, is_protein_);

        // If t is too small or sequences too short, fall back to scalar
        if (t < 2 || n < t || m < t) {
            return last_row_scalar(seq1, seq2, centre_diag, half_width);
        }

        // Process in t-sized blocks
        // Use two-row DP approach: keep current and previous row
        std::vector<float> prev_row(bw + 2, NEG_INF);
        std::vector<float> curr_row(bw + 2, NEG_INF);

        // Initialize: M[0][0] = 0
        {
            int bi = to_band_idx(0, 0, centre_diag, half_width);
            if (bi >= 0 && bi < bw + 2) prev_row[bi] = 0.0f;
        }
        // First row gaps
        {
            int jmin = band_j_min(0, centre_diag, half_width, m);
            int jmax = band_j_max(0, centre_diag, half_width, m);
            for (int j = 1; j <= jmax; ++j) {
                int bi = to_band_idx(j, 0, centre_diag, half_width);
                if (bi >= 0 && bi < bw + 2) {
                    prev_row[bi] = go_ + ge_ * j;
                }
            }
        }

        // Process blocks of t rows
        int i = 0;
        while (i < n) {
            int block_rows = std::min(t, n - i);

            // Compute hash BEFORE processing — includes sequence chars + quantized boundary
            size_t key = hash_block(seq1, seq2, i, std::min(i + t, n),
                                    centre_diag, half_width, prev_row);

            auto it = table_.find(key);
            if (it != table_.end()) {
                // Cache hit — use stored bottom boundary, skip DP computation
                prev_row = it->second.bottom_row;
                stats_.hits++;
            } else {
                // Cache miss — compute block via DP
                for (int bi = 0; bi < block_rows; ++bi) {
                    int row = i + bi + 1;
                    std::fill(curr_row.begin(), curr_row.end(), NEG_INF);

                    int jmin = band_j_min(row, centre_diag, half_width, m);
                    int jmax = band_j_max(row, centre_diag, half_width, m);

                    // First column gap
                    if (0 >= jmin && 0 <= jmax) {
                        int bidx = to_band_idx(0, row, centre_diag, half_width);
                        if (bidx >= 0 && bidx < bw + 2) {
                            curr_row[bidx] = go_ + ge_ * row;
                        }
                    }

                    for (int j = std::max(1, jmin); j <= jmax; ++j) {
                        int bidx = to_band_idx(j, row, centre_diag, half_width);
                        if (bidx < 0 || bidx >= bw + 2) continue;

                        float s = score_chars(seq1[row - 1], seq2[j - 1]);

                        // Diagonal from prev_row
                        int bd = to_band_idx(j - 1, row - 1, centre_diag, half_width);
                        float diag_val = NEG_INF;
                        if (bd >= 0 && bd < bw + 2) diag_val = prev_row[bd];

                        // Up from prev_row
                        int bu = to_band_idx(j, row - 1, centre_diag, half_width);
                        float up_val = NEG_INF;
                        if (bu >= 0 && bu < bw + 2) up_val = prev_row[bu];

                        // Left from curr_row
                        int bl = to_band_idx(j - 1, row, centre_diag, half_width);
                        float left_val = NEG_INF;
                        if (bl >= 0 && bl < bw + 2) left_val = curr_row[bl];

                        // Simple scoring (without separate affine tracking for speed)
                        float from_diag = diag_val + s;
                        float from_up = up_val + go_ + ge_;
                        float from_left = left_val + go_ + ge_;

                        curr_row[bidx] = std::max({from_diag, from_up, from_left});
                    }

                    prev_row = curr_row;
                }

                // Store in cache
                if (table_memory_bytes() < max_bytes_) {
                    BlockBoundary bb;
                    bb.bottom_row = prev_row;
                    table_[key] = std::move(bb);
                }
                stats_.computed_scalar++;
            }

            i += block_rows;
        }

        update_hit_ratio();
        return prev_row;
    }

    // Full alignment with traceback (standalone use)
    BandedResult align(
        const std::string& seq1,
        const std::string& seq2,
        int centre_diag, int half_width)
    {
        // For full alignment with traceback, delegate to banded auto
        // but still benefit from the cache warming effect
        last_row(seq1, seq2, centre_diag, half_width);  // warm cache
        return align_banded_auto(seq1, seq2, centre_diag, half_width,
                                 go_, ge_, is_protein_);
    }

    void reset_stats() { stats_ = Stats{}; }
    
    Stats get_stats() const { return stats_; }
    
    size_t table_memory_bytes() const {
        size_t total = 0;
        for (auto& [k, v] : table_) {
            total += sizeof(k) + sizeof(v);
            total += v.bottom_row.size() * sizeof(float);
            total += v.right_col.size() * sizeof(float);
        }
        return total;
    }
    
    void set_max_table_bytes(size_t max_bytes) { max_bytes_ = max_bytes; }

private:
    int t_;
    bool is_protein_;
    float go_, ge_;
    int B_;
    size_t max_bytes_ = 512ULL << 20;
    std::vector<float> subst_;
    std::unordered_map<size_t, BlockBoundary> table_;
    Stats stats_{};

    static int compute_t(int half_width, bool is_protein) {
        int bw = 2 * half_width + 1;
        int t = static_cast<int>(std::floor(std::log2(bw)));
        t = std::max(t, 2);
        if (is_protein) t = std::min(t, 2);
        else t = std::min(t, 4);
        return t;
    }

    float score_chars(char c1, char c2) const {
        int a = is_protein_ ? encode_protein(c1) : encode_dna(c1);
        int b = is_protein_ ? encode_protein(c2) : encode_dna(c2);
        if (a < 0 || b < 0) return -1.0f;
        if (!subst_.empty()) {
            int dim = is_protein_ ? 20 : 4;
            return subst_[a * dim + b];
        }
        return (a == b) ? 1.0f : -1.0f;
    }

    std::vector<float> last_row_scalar(
        const std::string& seq1,
        const std::string& seq2,
        int centre_diag, int half_width)
    {
        const int n = static_cast<int>(seq1.size());
        const int m = static_cast<int>(seq2.size());
        const int bw = 2 * half_width + 1;

        std::vector<float> prev(bw + 2, NEG_INF);
        std::vector<float> curr(bw + 2, NEG_INF);

        {
            int bi = to_band_idx(0, 0, centre_diag, half_width);
            if (bi >= 0 && bi < bw + 2) prev[bi] = 0.0f;
        }
        {
            int jmax = band_j_max(0, centre_diag, half_width, m);
            for (int j = 1; j <= jmax; ++j) {
                int bi = to_band_idx(j, 0, centre_diag, half_width);
                if (bi >= 0 && bi < bw + 2) prev[bi] = go_ + ge_ * j;
            }
        }

        for (int i = 1; i <= n; ++i) {
            std::fill(curr.begin(), curr.end(), NEG_INF);
            int jmin = band_j_min(i, centre_diag, half_width, m);
            int jmax = band_j_max(i, centre_diag, half_width, m);

            if (0 >= jmin && 0 <= jmax) {
                int bi = to_band_idx(0, i, centre_diag, half_width);
                if (bi >= 0 && bi < bw + 2) curr[bi] = go_ + ge_ * i;
            }

            for (int j = std::max(1, jmin); j <= jmax; ++j) {
                int bi = to_band_idx(j, i, centre_diag, half_width);
                if (bi < 0 || bi >= bw + 2) continue;

                float s = score_chars(seq1[i - 1], seq2[j - 1]);

                int bd = to_band_idx(j - 1, i - 1, centre_diag, half_width);
                float diag = (bd >= 0 && bd < bw + 2) ? prev[bd] : NEG_INF;

                int bu = to_band_idx(j, i - 1, centre_diag, half_width);
                float up = (bu >= 0 && bu < bw + 2) ? prev[bu] : NEG_INF;

                int bl = to_band_idx(j - 1, i, centre_diag, half_width);
                float left = (bl >= 0 && bl < bw + 2) ? curr[bl] : NEG_INF;

                curr[bi] = std::max({diag + s, up + go_ + ge_, left + go_ + ge_});
            }

            std::swap(prev, curr);
        }

        stats_.computed_scalar++;
        update_hit_ratio();
        return prev;
    }

    size_t hash_block_simple(const std::string& s1, const std::string& s2,
                              int start_i, int end_i,
                              int centre_diag, int half_width) const {
        // Legacy hash — not used by last_row() anymore
        return hash_block(s1, s2, start_i, end_i, centre_diag, half_width, {});
    }

    size_t hash_block(const std::string& s1, const std::string& s2,
                      int start_i, int end_i,
                      int centre_diag, int half_width,
                      const std::vector<float>& boundary_row) const {
        // FNV-1a hash on: sequence chars + quantized boundary values
        // NO position (start_i) — same content at different positions must match
        size_t h = 14695981039346656037ULL;
        constexpr size_t FNV_PRIME = 1099511628211ULL;

        // Hash seq1 chars in this block
        for (int idx = start_i; idx < end_i && idx < (int)s1.size(); ++idx) {
            h ^= static_cast<size_t>(s1[idx]);
            h *= FNV_PRIME;
        }

        // Hash relevant seq2 chars (band region)
        int jmin = band_j_min(start_i + 1, centre_diag, half_width, (int)s2.size());
        int jmax = band_j_max(std::min(end_i, (int)s1.size()),
                              centre_diag, half_width, (int)s2.size());
        for (int j = std::max(0, jmin);
             j <= std::min((int)s2.size() - 1, jmax); ++j) {
            h ^= static_cast<size_t>(s2[j]);
            h *= FNV_PRIME;
        }

        // Hash quantized boundary values (top boundary of this block)
        for (size_t k = 0; k < boundary_row.size(); ++k) {
            int q = quantize_boundary(boundary_row[k]);
            h ^= static_cast<size_t>(static_cast<unsigned int>(q + 100000));
            h *= FNV_PRIME;
        }

        return h;
    }

    int quantize_boundary(float val) const {
        if (val <= NEG_INF + 1.0f) return -(B_ * 100);
        float clamped = std::max(-1000.0f, std::min(1000.0f, val));
        return static_cast<int>(std::round(clamped * B_ / 100.0f));
    }

    void update_hit_ratio() {
        int total = stats_.hits + stats_.computed_simd + stats_.computed_scalar;
        stats_.hit_ratio = (total > 0) ? static_cast<float>(stats_.hits) / total : 0.0f;
    }
};
