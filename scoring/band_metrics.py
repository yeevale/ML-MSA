# scoring/band_metrics.py — Metrics for neural network band prediction quality.
# band_recall@1x — main metric: fraction of pairs where path fits in band.
# width_efficiency > 1 = overestimate (safe), < 1 = underestimate (triggers doubling).

import numpy as np


def band_recall(true_path: list[tuple[int, int]],
                centre_diag: int,
                half_width: int) -> float:
    """Fraction of path points inside band.
    For each (i,j): diag = i-j; inside if abs(diag - centre_diag) <= half_width."""
    if not true_path:
        return 1.0
    inside = sum(1 for i, j in true_path
                 if abs((i - j) - centre_diag) <= half_width)
    return inside / len(true_path)


def width_efficiency(pred_hw: int, true_hw: int) -> float:
    """pred_hw / true_hw. 1.0=perfect, <1=dangerous (triggers doubling)."""
    return pred_hw / max(true_hw, 1)


def mean_doublings(doubling_results: list) -> float:
    """Mean n_doublings across a list of DoublingResult objects."""
    if not doubling_results:
        return 0.0
    return float(np.mean([r.n_doublings for r in doubling_results]))


def band_recall_at(pred_hws: np.ndarray,
                   true_hws: np.ndarray,
                   multiplier: float) -> float:
    """Fraction of pairs where true_hw <= pred_hw * multiplier."""
    return float((true_hws <= pred_hws * multiplier).mean())


def band_metrics_summary(pred_centres: np.ndarray,
                         pred_hws: np.ndarray,
                         true_centres: np.ndarray,
                         true_hws: np.ndarray) -> dict[str, float]:
    """Compute full suite of band prediction metrics."""
    mae_centre = float(np.abs(pred_centres - true_centres).mean())
    mae_hw = float(np.abs(pred_hws - true_hws).mean())

    efficiencies = pred_hws / np.maximum(true_hws, 1).astype(np.float64)
    underest_frac = float((pred_hws < true_hws).mean())

    return {
        "mae_centre": mae_centre,
        "mae_half_width": mae_hw,
        "band_recall@1.0x": band_recall_at(pred_hws, true_hws, 1.0),
        "band_recall@1.5x": band_recall_at(pred_hws, true_hws, 1.5),
        "band_recall@2.0x": band_recall_at(pred_hws, true_hws, 2.0),
        "width_efficiency_mean": float(efficiencies.mean()),
        "width_efficiency_std": float(efficiencies.std()),
        "underestimate_fraction": underest_frac,
    }


if __name__ == "__main__":
    # Smoke test
    path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5)]
    r = band_recall(path, centre_diag=0, half_width=1)
    print(f"band_recall: {r:.4f}")
    assert r == 1.0  # all within ±1 of diagonal 0

    r2 = band_recall(path, centre_diag=0, half_width=0)
    print(f"band_recall(hw=0): {r2:.4f}")
    assert r2 < 1.0  # (4,5) has diag=-1

    e = width_efficiency(10, 8)
    print(f"width_efficiency(10,8): {e:.4f}")
    assert e > 1.0

    pred_hw = np.array([10, 20, 5, 15])
    true_hw = np.array([8, 25, 5, 10])
    r1x = band_recall_at(pred_hw, true_hw, 1.0)
    print(f"recall@1x: {r1x:.4f}")

    summary = band_metrics_summary(
        pred_centres=np.array([0, 1, -1, 2]),
        pred_hws=pred_hw,
        true_centres=np.array([0, 0, 0, 1]),
        true_hws=true_hw,
    )
    print(f"Summary: {summary}")

    print("Smoke test passed!")
