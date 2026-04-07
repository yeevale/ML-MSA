#!/usr/bin/env python3
"""
Generate FINAL_REPORT.md from experiment results.
Run after experiments/run_all.py:
    python results/interpret_results.py \
        --results_dir results \
        --output results/FINAL_REPORT.md
"""

import argparse
import json
import os
from datetime import datetime

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def fmt(v, decimals=3):
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)


def generate_report(results_dir: str, output: str):
    exp_dir = os.path.join(results_dir, "experiments")
    train_dir = os.path.join(results_dir, "training")

    lines = []
    lines.append("# MSA Neural Band Prediction - Final Results Report")
    lines.append(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ---------- Experiment status ----------
    lines.append("## Experiment Status")
    summary = load_json(os.path.join(exp_dir, "summary.json"))
    if summary:
        lines.append("")
        lines.append("| Experiment | Status | Time (s) |")
        lines.append("|---|---|---|")
        for name, res in summary.items():
            status = res.get("_status", "?")
            elapsed = res.get("_elapsed_s", 0)
            mark = "OK" if status == "OK" else "ERROR"
            lines.append(f"| {name} | {mark} | {elapsed:.1f} |")
    lines.append("")

    # ---------- Training ----------
    lines.append("## 1. Neural Network Training Results")
    training_log = os.path.join(train_dir, "training_history.json")
    if os.path.exists(training_log):
        history = load_json(training_log)
        if history:
            lines.append("")
            lines.append("### Best Epoch Metrics")
            best = history.get("best_epoch_metrics", {})
            lines.append(f"- **band_recall@1x:** {fmt(best.get('band_recall@1.0x'))}")
            lines.append(f"- **band_recall@1.5x:** {fmt(best.get('band_recall@1.5x'))}")
            lines.append(f"- **band_recall@2x:** {fmt(best.get('band_recall@2.0x'))}")
            lines.append(f"- **MAE centre_diag:** {fmt(best.get('mae_centre'))}")
            lines.append(f"- **width_ratio:** {fmt(best.get('width_ratio'))}")
            lines.append(f"- **Best epoch:** {best.get('epoch', 'N/A')}")
            lines.append("")

            epochs = history.get("epochs", [])
            if epochs:
                lines.append("### Training History (all epochs)")
                lines.append("")
                lines.append("| Epoch | Loss | Recall@1x | Recall@1.5x | MAE_centre | Width_ratio |")
                lines.append("|---|---|---|---|---|---|")
                for e in epochs:
                    lines.append(
                        f"| {e.get('epoch','')} "
                        f"| {fmt(e.get('loss'), 4)} "
                        f"| {fmt(e.get('band_recall@1.0x'))} "
                        f"| {fmt(e.get('band_recall@1.5x'))} "
                        f"| {fmt(e.get('mae_centre'))} "
                        f"| {fmt(e.get('width_ratio'))} |"
                    )
    else:
        lines.append("")
        lines.append("*Training history not found*")
    lines.append("")

    # ---------- Experiment 1: band accuracy ----------
    lines.append("## 2. Band Prediction Accuracy")
    band_acc = load_json(os.path.join(exp_dir, "band_prediction_accuracy.json"))
    if band_acc and band_acc.get("_status") == "OK":
        lines.append("")
        lines.append(f"- **Overall recall@1x:** {fmt(band_acc.get('overall_recall_1x'))} "
                     f"(no doubling needed)")
        lines.append(f"- **Overall recall@1.5x:** {fmt(band_acc.get('overall_recall_1_5x'))}")
        lines.append(f"- **Overall recall@2x:** {fmt(band_acc.get('overall_recall_2x'))}")
        lines.append(f"- **MAE centre_diag:** {fmt(band_acc.get('overall_mae_centre'))}")
        lines.append(f"- **Width ratio:** {fmt(band_acc.get('overall_width_ratio'))} "
                     f"(>1 = overestimate, safe)")
        lines.append("")
        lines.append("### By Divergence Group")
        lines.append("")
        lines.append("| Group | Recall@1x | Recall@1.5x | Recall@2x | MAE centre |")
        lines.append("|---|---|---|---|---|")
        by_group = band_acc.get("by_divergence_group", {})
        for group in ["low", "medium", "high"]:
            r1  = fmt(by_group.get("recall_1x", {}).get(group))
            r15 = fmt(by_group.get("recall_1_5x", {}).get(group))
            r2  = fmt(by_group.get("recall_2x", {}).get(group))
            mae = fmt(by_group.get("mae_centre", {}).get(group))
            lines.append(f"| {group} | {r1} | {r15} | {r2} | {mae} |")
    else:
        lines.append("")
        lines.append("*Experiment skipped or failed*")
    lines.append("")

    # ---------- Experiment 2: speedup ----------
    lines.append("## 3. Pairwise Alignment Speedup")
    speedup = load_json(os.path.join(exp_dir, "pairwise_speedup.json"))
    if speedup and speedup.get("_status") == "OK":
        lines.append("")
        lines.append(f"- **Mean speedup vs Full NW:** {fmt(speedup.get('mean_speedup_vs_full'), 1)}x")
        lines.append(f"- **Max speedup:** {fmt(speedup.get('max_speedup_vs_full'), 1)}x")
        lines.append(f"- **Mean speedup vs Fixed W=30:** "
                     f"{fmt(speedup.get('mean_speedup_vs_fixed30'), 2)}x")
        lines.append("")
        lines.append("### Speedup by Sequence Length (vs Full NW)")
        lines.append("")
        lines.append("| Length | Mean speedup |")
        lines.append("|---|---|")
        for length, sp in speedup.get("summary", {}).items():
            lines.append(f"| {length} bp | {sp}x |")

        # Detailed table from CSV
        csv_path = os.path.join(exp_dir, "pairwise_speedup.csv")
        if os.path.exists(csv_path) and HAS_PANDAS:
            df = pd.read_csv(csv_path)
            lines.append("")
            lines.append("### Detailed Table (time in ms)")
            lines.append("")
            lines.append("| Length | Divergence | Full NW | Fixed W=30 | Fixed W=100 | Neural | Speedup |")
            lines.append("|---|---|---|---|---|---|---|")
            for _, row in df.iterrows():
                lines.append(
                    f"| {int(row['length'])} "
                    f"| {row['divergence']:.0%} "
                    f"| {row['t_full_ms']:.1f} "
                    f"| {row['t_fixed30_ms']:.1f} "
                    f"| {row['t_fixed100_ms']:.1f} "
                    f"| {row['t_neural_ms']:.1f} "
                    f"| {row['speedup_vs_full']:.1f}x |"
                )
    else:
        lines.append("")
        lines.append("*Experiment skipped or failed*")
    lines.append("")

    # ---------- Experiment 3: ablation ----------
    lines.append("## 4. Ablation Study")
    ablation = load_json(os.path.join(exp_dir, "ablation_study.json"))
    if ablation and ablation.get("_status") == "OK":
        lines.append("")
        lines.append("| Method | Time (ms) | Speedup vs Full NW | Doublings |")
        lines.append("|---|---|---|---|")
        for cfg in ablation.get("configs", []):
            doublings = f"{cfg.get('mean_doublings', 0):.2f}" if "mean_doublings" in cfg else "-"
            lines.append(
                f"| {cfg['name']} "
                f"| {cfg['time_ms']:.2f} "
                f"| {cfg['speedup']:.1f}x "
                f"| {doublings} |"
            )
        lines.append("")
        lines.append(f"- **Neural mean doublings:** "
                     f"{fmt(ablation.get('neural_mean_doublings'), 3)}")
        lines.append(f"- **Zero-doubling rate:** "
                     f"{fmt(ablation.get('neural_zero_doubling_rate'))} "
                     f"(network predicted correctly on first try)")
    else:
        lines.append("")
        lines.append("*Experiment skipped or failed*")
    lines.append("")

    # ---------- Experiment 4: Four Russians ----------
    lines.append("## 5. Four Russians Lookup Table Accumulation")
    fr = load_json(os.path.join(exp_dir, "fr_hit_ratio.json"))
    if fr and fr.get("_status") == "OK":
        lines.append("")
        lines.append(f"- **Initial hit_ratio:** {fmt(fr.get('initial_hit_ratio'))}")
        lines.append(f"- **Final hit_ratio:** {fmt(fr.get('final_hit_ratio'))}")
        lines.append(f"- **Table size:** {fr.get('final_table_kb', 0)} KB")
        lines.append("")
        csv_path = os.path.join(exp_dir, "fr_hit_ratio.csv")
        if os.path.exists(csv_path) and HAS_PANDAS:
            df = pd.read_csv(csv_path)
            lines.append("| Pairs processed | Hit ratio | Table (KB) |")
            lines.append("|---|---|---|")
            for _, row in df.iterrows():
                lines.append(
                    f"| {int(row['n_pairs'])} "
                    f"| {row['hit_ratio']:.1%} "
                    f"| {int(row['table_kb'])} |"
                )
    else:
        lines.append("")
        lines.append("*Experiment skipped or failed*")
    lines.append("")

    # ---------- Experiment 5: MSA quality ----------
    lines.append("## 6. MSA Quality")
    msa_q = load_json(os.path.join(exp_dir, "msa_quality.json"))
    if msa_q and msa_q.get("_status") == "OK":
        lines.append("")
        lines.append(f"- Groups tested: {msa_q.get('n_groups', 0)}")
        lines.append("")
        lines.append("### Summary Table (all methods)")
        lines.append("")
        lines.append("| Method | SP-score | TC-score | Time (s) | Memory (MB) |")
        lines.append("|---|---|---|---|---|")
        summary_data = msa_q.get("summary", {})
        method_order = ["ClustalW", "MAFFT", "MUSCLE",
                        "Fixed_W30", "Fixed_W100",
                        "Neural_band", "Neural_+_refine"]
        sp_means = summary_data.get("SP_mean", {})
        tc_means = summary_data.get("TC_mean", {})
        t_means  = summary_data.get("Time_mean", {})
        m_means  = summary_data.get("Mem_MB_mean", {})
        for method in method_order:
            if method in sp_means:
                lines.append(
                    f"| {method} "
                    f"| {fmt(sp_means.get(method))} "
                    f"| {fmt(tc_means.get(method))} "
                    f"| {fmt(t_means.get(method), 2)} "
                    f"| {fmt(m_means.get(method), 1)} |"
                )
    else:
        lines.append("")
        lines.append("*Experiment skipped (no model)*")
    lines.append("")

    # ---------- Experiment 6: scaling ----------
    lines.append("## 7. Scaling by Number of Sequences N")
    scaling = load_json(os.path.join(exp_dir, "scaling_by_n.json"))
    if scaling and scaling.get("_status") == "OK":
        lines.append("")
        lines.append("| N | MAFFT (s) | Neural (s) | Speedup |")
        lines.append("|---|---|---|---|")
        for row in scaling.get("rows", []):
            t_mafft = fmt(row.get("t_mafft_s"), 2) if row.get("t_mafft_s") else "N/A"
            t_neural = fmt(row.get("t_neural_s"), 2)
            speedup_val = f"{row['speedup']}x" if row.get("speedup") else "N/A"
            lines.append(
                f"| {row.get('n_seqs')} "
                f"| {t_mafft} "
                f"| {t_neural} "
                f"| {speedup_val} |"
            )
    else:
        lines.append("")
        lines.append("*Experiment skipped or failed*")
    lines.append("")

    # ---------- Tests ----------
    lines.append("## 8. Test Results")
    test_report = load_json(os.path.join(results_dir, "tests/full_test_report.json"))
    if test_report:
        summary_t = test_report.get("summary", {})
        lines.append("")
        lines.append(f"- **Total tests:** {summary_t.get('total', 'N/A')}")
        lines.append(f"- **Passed:** {summary_t.get('passed', 'N/A')}")
        lines.append(f"- **Failed:** {summary_t.get('failed', 0)}")
        lines.append(f"- **Skipped:** {summary_t.get('skipped', 0)}")
        lines.append(f"- **Duration:** {fmt(summary_t.get('duration'), 1)}s")
    else:
        lines.append("")
        lines.append("*Test report not found*")
    lines.append("")

    # ---------- System info ----------
    lines.append("## 9. System Information")
    sysinfo = load_json(os.path.join(results_dir, "system_info.json"))
    if sysinfo:
        lines.append("")
        for k, v in sysinfo.items():
            lines.append(f"- **{k}:** {v}")
    else:
        lines.append("")
        lines.append("*System info not found*")
    lines.append("")

    lines.append("---")
    lines.append("*End of report.*")

    # Write file
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output", default="results/FINAL_REPORT.md")
    args = parser.parse_args()
    generate_report(args.results_dir, args.output)
