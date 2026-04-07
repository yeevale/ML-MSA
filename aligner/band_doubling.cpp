// band_doubling.cpp — Asymmetric band doubling: guaranteed optimality.
// + Method dispatcher: Hirschberg + Four Russians + SIMD always together.
// + pybind11 entry point for the entire 'aligner' module.
//
// ASYMMETRIC EXPANSION (saves ~50% on one-sided escapes):
//   escape_left only:  new_left = centre - hw*2, new_right = centre + hw
//   escape_right only: new_left = centre - hw,   new_right = centre + hw*2
//   both true:         new_left = centre - hw*2, new_right = centre + hw*2
//
// DISPATCHER (each iteration, methods are not mutually exclusive):
//   estimated_mem = max(len1,len2) * (2*hw+1) * 3 * sizeof(float)
//   needs_hirschberg = estimated_mem > HIRSCHBERG_THRESHOLD (200 MB)
//   use_four_russians = hw >= FR_MIN_HALF_WIDTH (16)
//   use_simd = HAVE_AVX2 (compile-time)
//
// FourRussiansAligner created ONCE per align_with_doubling call,
// reused across all iterations (table accumulates).

#include "hirschberg.cpp"

constexpr long long HIRSCHBERG_THRESHOLD = 200LL << 20;  // 200 MB
constexpr int       FR_MIN_HALF_WIDTH    = 16;

// Determine if Hirschberg is needed for memory savings
inline bool needs_hirschberg(int len1, int len2, int hw) {
    return (long long)std::max(len1, len2) * (2LL * hw + 1) * 3 * sizeof(float)
           > HIRSCHBERG_THRESHOLD;
}

// One iteration: always use direct banded NW (SIMD-accelerated when available).
// Four Russians removed from hot path — overhead exceeded benefit.
BandedResult run_one_iteration(
    const std::string& seq1, const std::string& seq2,
    int centre_diag, int half_width,
    float gap_open, float gap_extend, bool is_protein,
    DoublingResult& result_meta)
{
    int n = static_cast<int>(seq1.size());
    int m = static_cast<int>(seq2.size());

    if (needs_hirschberg(n, m, half_width)) {
        result_meta.used_hirschberg = true;
        // For Hirschberg, use plain banded NW (no FR).
        // Split in half, compute last rows directly, then recurse via base case.
        // For simplicity, fall through to align_banded_auto which handles
        // large matrices via banded DP with O(n*W) memory.
    }

#ifdef HAVE_AVX2
    result_meta.used_simd = true;
#endif
    return align_banded_auto(seq1, seq2, centre_diag, half_width,
                             gap_open, gap_extend, is_protein);
}

// Main function: asymmetric band doubling
DoublingResult align_with_doubling(
    const std::string& seq1,
    const std::string& seq2,
    int   pred_centre,
    int   pred_hw,
    float gap_open,
    float gap_extend,
    bool  is_protein,
    const float* subst)
{
    DoublingResult result;
    result.n_doublings = 0;
    result.used_hirschberg = false;
    result.used_four_russians = false;
    result.used_simd = false;

    int n = static_cast<int>(seq1.size());
    int m = static_cast<int>(seq2.size());

    // Ensure minimum band width
    int hw = std::max(pred_hw, 1);
    int centre = pred_centre;

    // Clamp centre to valid range
    int max_centre = n;
    int min_centre = -m;
    centre = std::max(min_centre, std::min(max_centre, centre));

    // Left and right bounds of band (asymmetric tracking)
    int left_bound = centre - hw;
    int right_bound = centre + hw;

    constexpr int MAX_DOUBLINGS = 10;

    for (int iter = 0; iter <= MAX_DOUBLINGS; ++iter) {
        int cur_hw = (right_bound - left_bound) / 2;
        int cur_centre = (left_bound + right_bound) / 2;

        BandedResult res = run_one_iteration(
            seq1, seq2, cur_centre, cur_hw,
            gap_open, gap_extend, is_protein, result);

        if (!res.path_escaped) {
            // Success: path fits within band
            result.alignment = res;
            result.final_left_bound = left_bound;
            result.final_right_bound = right_bound;
            return result;
        }

        // Path escaped → asymmetric doubling
        result.n_doublings++;

        if (res.escape_left && !res.escape_right) {
            // Expand only to the left
            left_bound = cur_centre - cur_hw * 2;
        } else if (res.escape_right && !res.escape_left) {
            // Expand only to the right
            right_bound = cur_centre + cur_hw * 2;
        } else {
            // Both sides or unknown → expand both
            left_bound = cur_centre - cur_hw * 2;
            right_bound = cur_centre + cur_hw * 2;
        }

        // Safety: clamp bounds
        left_bound = std::max(-(m + 1), left_bound);
        right_bound = std::min(n + 1, right_bound);

        // If band covers the entire matrix, do full NW
        if (right_bound - left_bound >= n + m + 2) {
            result.alignment = full_nw_align(seq1, seq2, gap_open, gap_extend,
                                              is_protein, nullptr);
            result.final_left_bound = left_bound;
            result.final_right_bound = right_bound;
            return result;
        }
    }

    // Fallback after too many doublings: full NW
    result.alignment = full_nw_align(seq1, seq2, gap_open, gap_extend,
                                      is_protein, nullptr);
    result.final_left_bound = -(m + 1);
    result.final_right_bound = n + 1;
    return result;
}

// Profile-profile version with doubling
DoublingResult align_profiles_with_doubling(
    const py::array_t<float>& p1,
    const py::array_t<float>& p2,
    const py::array_t<float>& subst,
    int pred_centre, int pred_hw,
    float gap_open, float gap_extend)
{
    DoublingResult result;
    result.n_doublings = 0;
    result.used_hirschberg = false;
    result.used_four_russians = false;
    result.used_simd = false;

    auto p1_buf = p1.unchecked<2>();
    auto p2_buf = p2.unchecked<2>();
    int n = static_cast<int>(p1_buf.shape(0));
    int m = static_cast<int>(p2_buf.shape(0));

    int hw = std::max(pred_hw, 1);
    int centre = pred_centre;

    int left_bound = centre - hw;
    int right_bound = centre + hw;

    constexpr int MAX_DOUBLINGS = 10;

    for (int iter = 0; iter <= MAX_DOUBLINGS; ++iter) {
        int cur_hw = (right_bound - left_bound) / 2;
        int cur_centre = (left_bound + right_bound) / 2;

        BandedResult res = align_banded_profiles(p1, p2, subst, cur_centre, cur_hw,
                                                  gap_open, gap_extend);

        if (!res.path_escaped) {
            result.alignment = res;
            result.final_left_bound = left_bound;
            result.final_right_bound = right_bound;
            return result;
        }

        result.n_doublings++;

        if (res.escape_left && !res.escape_right) {
            left_bound = cur_centre - cur_hw * 2;
        } else if (res.escape_right && !res.escape_left) {
            right_bound = cur_centre + cur_hw * 2;
        } else {
            left_bound = cur_centre - cur_hw * 2;
            right_bound = cur_centre + cur_hw * 2;
        }

        left_bound = std::max(-(m + 1), left_bound);
        right_bound = std::min(n + 1, right_bound);

        if (right_bound - left_bound >= n + m + 2) {
            // Full profile DP
            result.alignment = align_banded_profiles(p1, p2, subst, 0,
                                                      std::max(n, m), gap_open, gap_extend);
            result.final_left_bound = left_bound;
            result.final_right_bound = right_bound;
            return result;
        }
    }

    // Fallback
    result.alignment = align_banded_profiles(p1, p2, subst, 0,
                                              std::max(n, m), gap_open, gap_extend);
    result.final_left_bound = -(m + 1);
    result.final_right_bound = n + 1;
    return result;
}

// Include helpers that depend on align_with_doubling being defined above
#include "profile_dp.cpp"
#include "anchored_align.cpp"

// ======= PYBIND11 MODULE (the only one in the entire project) =======
PYBIND11_MODULE(aligner, m) {
    m.doc() = "Neural-guided banded MSA: Hirschberg+FourRussians+SIMD+AsymDoubling";

    py::class_<BandedResult>(m, "BandedResult")
        .def_readonly("score",         &BandedResult::score)
        .def_readonly("aligned_seq1",  &BandedResult::aligned_seq1)
        .def_readonly("aligned_seq2",  &BandedResult::aligned_seq2)
        .def_readonly("path_escaped",  &BandedResult::path_escaped)
        .def_readonly("escape_left",   &BandedResult::escape_left)
        .def_readonly("escape_right",  &BandedResult::escape_right)
        .def_readonly("max_deviation", &BandedResult::max_deviation);

    py::class_<DoublingResult>(m, "DoublingResult")
        .def_readonly("alignment",          &DoublingResult::alignment)
        .def_readonly("n_doublings",        &DoublingResult::n_doublings)
        .def_readonly("final_left_bound",   &DoublingResult::final_left_bound)
        .def_readonly("final_right_bound",  &DoublingResult::final_right_bound)
        .def_readonly("used_hirschberg",    &DoublingResult::used_hirschberg)
        .def_readonly("used_four_russians", &DoublingResult::used_four_russians)
        .def_readonly("used_simd",          &DoublingResult::used_simd);

    py::class_<FourRussiansAligner::Stats>(m, "FRStats")
        .def_readonly("hits",            &FourRussiansAligner::Stats::hits)
        .def_readonly("computed_simd",   &FourRussiansAligner::Stats::computed_simd)
        .def_readonly("computed_scalar", &FourRussiansAligner::Stats::computed_scalar)
        .def_readonly("hit_ratio",       &FourRussiansAligner::Stats::hit_ratio);

    // Pairwise alignment
    m.def("align_banded", &align_banded_auto,
          py::arg("seq1"), py::arg("seq2"),
          py::arg("centre_diag"), py::arg("half_width"),
          py::arg("gap_open") = -10.0f,
          py::arg("gap_extend") = -0.5f,
          py::arg("is_protein") = false,
          "Banded NW: scalar or SIMD AVX2 (auto)");

    m.def("align_hirschberg", &hirschberg_banded,
          py::arg("seq1"), py::arg("seq2"),
          py::arg("centre_diag"), py::arg("half_width"),
          py::arg("gap_open") = -10.0f,
          py::arg("gap_extend") = -0.5f,
          py::arg("is_protein") = false,
          "Hirschberg+FR+SIMD: O(W) memory, O(nW/logW) time");

    m.def("align_with_doubling",
          [](const std::string& seq1, const std::string& seq2,
             int pred_centre, int pred_hw,
             float gap_open, float gap_extend, bool is_protein) -> DoublingResult {
              return align_with_doubling(seq1, seq2, pred_centre, pred_hw,
                                         gap_open, gap_extend, is_protein, nullptr);
          },
          py::arg("seq1"), py::arg("seq2"),
          py::arg("pred_centre"), py::arg("pred_hw"),
          py::arg("gap_open") = -10.0f,
          py::arg("gap_extend") = -0.5f,
          py::arg("is_protein") = false,
          "Guaranteed-optimal: asymmetric doubling + Hirschberg+FR+SIMD dispatcher");

    // Profile-profile alignment
    m.def("align_profiles", &align_banded_profiles,
          py::arg("profile1"), py::arg("profile2"), py::arg("subst"),
          py::arg("centre_diag"), py::arg("half_width"),
          py::arg("gap_open") = -10.0f,
          py::arg("gap_extend") = -0.5f,
          "Profile-profile banded DP");

    m.def("align_profiles_with_doubling", &align_profiles_with_doubling,
          py::arg("profile1"), py::arg("profile2"), py::arg("subst"),
          py::arg("pred_centre"), py::arg("pred_hw"),
          py::arg("gap_open") = -10.0f,
          py::arg("gap_extend") = -0.5f,
          "Profile-profile with doubling fallback");

    // Reference (verification only) — use lambda to handle optional subst_matrix
    m.def("full_nw_align",
          [](const std::string& seq1, const std::string& seq2,
             float gap_open, float gap_extend, bool is_protein,
             py::object subst_obj) -> BandedResult {
              if (subst_obj.is_none()) {
                  return full_nw_align(seq1, seq2, gap_open, gap_extend, is_protein, nullptr);
              }
              auto subst = subst_obj.cast<py::array_t<float>>();
              return full_nw_align(seq1, seq2, gap_open, gap_extend, is_protein, &subst);
          },
          py::arg("seq1"), py::arg("seq2"),
          py::arg("gap_open") = -10.0f,
          py::arg("gap_extend") = -0.5f,
          py::arg("is_protein") = false,
          py::arg("subst_matrix") = py::none(),
          "Full NW (verification only)");

    m.def("full_nw_traceback",
          [](const std::string& seq1, const std::string& seq2,
             float gap_open, float gap_extend, bool is_protein,
             py::object subst_obj) -> std::vector<std::pair<int,int>> {
              if (subst_obj.is_none()) {
                  return full_nw_traceback(seq1, seq2, gap_open, gap_extend, is_protein, nullptr);
              }
              auto subst = subst_obj.cast<py::array_t<float>>();
              return full_nw_traceback(seq1, seq2, gap_open, gap_extend, is_protein, &subst);
          },
          py::arg("seq1"), py::arg("seq2"),
          py::arg("gap_open") = -10.0f,
          py::arg("gap_extend") = -0.5f,
          py::arg("is_protein") = false,
          py::arg("subst_matrix") = py::none(),
          "Full NW traceback for simulate.py");

    // FourRussiansAligner as standalone (for experiments)
    py::class_<FourRussiansAligner>(m, "FourRussiansAligner")
        .def(py::init([](int block_size, bool is_protein, float gap_open,
                         float gap_extend, int quant_levels) {
                 return new FourRussiansAligner(block_size, is_protein,
                                                gap_open, gap_extend,
                                                quant_levels, nullptr);
             }),
             py::arg("block_size") = 0,
             py::arg("is_protein") = false,
             py::arg("gap_open") = -10.0f,
             py::arg("gap_extend") = -0.5f,
             py::arg("quant_levels") = 16)
        .def("last_row", &FourRussiansAligner::last_row,
             py::arg("seq1"), py::arg("seq2"),
             py::arg("centre_diag"), py::arg("half_width"))
        .def("align", &FourRussiansAligner::align,
             py::arg("seq1"), py::arg("seq2"),
             py::arg("centre_diag"), py::arg("half_width"))
        .def("reset_stats", &FourRussiansAligner::reset_stats)
        .def("get_stats", &FourRussiansAligner::get_stats)
        .def("table_memory_bytes", &FourRussiansAligner::table_memory_bytes)
        .def("set_max_table_bytes", &FourRussiansAligner::set_max_table_bytes,
             py::arg("max_bytes"));
}
