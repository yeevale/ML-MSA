#!/usr/bin/env python3
"""
Unified entry point for all experiments.
Run after training:
    python experiments/run_all.py \
        --checkpoint checkpoints/best_model.pt \
        --balibase_dir data/raw/balibase/DATASET-BALiBASE \
        --results_dir results/experiments \
        --device cuda
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_experiment(name: str, fn, results_dir: str) -> dict:
    """Run a single experiment with error handling."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - t0
        result["_status"] = "OK"
        result["_elapsed_s"] = round(elapsed, 2)
        print(f"  DONE in {elapsed:.1f}s")

        # Save result as JSON
        out_path = os.path.join(results_dir, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Saved: {out_path}")
        return result

    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  ERROR: {e}")
        traceback.print_exc()
        result = {"_status": "ERROR", "_error": str(e), "_elapsed_s": round(elapsed, 2)}
        out_path = os.path.join(results_dir, f"{name}_ERROR.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        return result


def exp_band_prediction_accuracy(predictor, results_dir: str) -> dict:
    """
    Experiment 1: Neural band prediction accuracy.

    For each validation pair:
    - predict (centre_diag, half_width)
    - compare with true_half_width from parquet
    - compute band_recall@1x, @1.5x, @2x

    Groups by divergence level: low/medium/high.
    """
    df = pd.read_parquet("data/processed/val.parquet")
    results = []

    for div_group, div_lo, div_hi in [
        ("low",    0.0,  0.10),
        ("medium", 0.10, 0.25),
        ("high",   0.25, 0.50),
    ]:
        subset = df[(df["divergence"] >= div_lo) & (df["divergence"] < div_hi)]
        n_sample = min(500, len(subset))
        if n_sample == 0:
            continue
        subset = subset.sample(n_sample, random_state=42)

        for _, row in subset.iterrows():
            seq1, seq2 = row["seq1"], row["seq2"]
            true_hw = int(row["true_half_width"])
            true_centre = int(row["centre_diag"])
            div = float(row["divergence"])

            pred_centre, pred_hw = predictor.predict_single(seq1, seq2, "dna")
            results.append({
                "div_group":    div_group,
                "divergence":   div,
                "true_hw":      true_hw,
                "pred_hw":      pred_hw,
                "true_centre":  true_centre,
                "pred_centre":  pred_centre,
                "recall_1x":    int(true_hw <= pred_hw * 1.0),
                "recall_1_5x":  int(true_hw <= pred_hw * 1.5),
                "recall_2x":    int(true_hw <= pred_hw * 2.0),
                "centre_err":   abs(pred_centre - true_centre),
                "width_ratio":  pred_hw / max(true_hw, 1),
            })

    res_df = pd.DataFrame(results)
    detail_csv = os.path.join(results_dir, "band_prediction_detail.csv")
    res_df.to_csv(detail_csv, index=False)

    # Aggregate by group
    summary = {}
    if len(res_df) > 0:
        agg = res_df.groupby("div_group").agg(
            recall_1x=("recall_1x", "mean"),
            recall_1_5x=("recall_1_5x", "mean"),
            recall_2x=("recall_2x", "mean"),
            mae_centre=("centre_err", "mean"),
            width_ratio=("width_ratio", "mean"),
            n_samples=("div_group", "count"),
        ).round(4)
        summary = agg.to_dict()

    return {
        "by_divergence_group": summary,
        "overall_recall_1x":   float(res_df["recall_1x"].mean()) if len(res_df) else 0,
        "overall_recall_1_5x": float(res_df["recall_1_5x"].mean()) if len(res_df) else 0,
        "overall_recall_2x":   float(res_df["recall_2x"].mean()) if len(res_df) else 0,
        "overall_mae_centre":  float(res_df["centre_err"].mean()) if len(res_df) else 0,
        "overall_width_ratio": float(res_df["width_ratio"].mean()) if len(res_df) else 0,
        "n_total":             len(res_df),
        "detail_csv":          detail_csv,
    }


def exp_pairwise_speedup(predictor, results_dir: str) -> dict:
    """
    Experiment 2: Pairwise alignment speedup.

    Grid: 5 lengths x 4 divergence levels x 4 methods.
    Methods: Full NW | Fixed W=30 | Fixed W=100 | Neural band.
    Each point averaged over 5 runs.
    """
    import aligner

    DNA = "ACGT"

    def gen_pair(length, divergence, seed):
        rng = np.random.default_rng(seed)
        seq1 = "".join(rng.choice(list(DNA), length))
        seq2 = list(seq1)
        n_mut = int(length * divergence)
        pos = rng.choice(length, min(n_mut, length), replace=False)
        for p in pos:
            seq2[p] = rng.choice([c for c in DNA if c != seq2[p]])
        n_indel = max(1, int(n_mut * 0.3))
        ins_pos = sorted(rng.choice(len(seq2), min(n_indel // 2, len(seq2)), replace=False), reverse=True)
        for p in ins_pos:
            seq2.insert(p, rng.choice(list(DNA)))
        del_count = min(n_indel // 2, len(seq2) - 1)
        if del_count > 0:
            del_pos = sorted(rng.choice(len(seq2), del_count, replace=False), reverse=True)
            for p in del_pos:
                if len(seq2) > 1:
                    seq2.pop(p)
        return seq1, "".join(seq2)

    def timeit(fn, n=5):
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return float(np.median(times))

    lengths = [300, 500, 1000, 2000, 5000]
    divs = [0.05, 0.10, 0.20, 0.30]
    rows = []

    for length in lengths:
        for div in divs:
            seq1, seq2 = gen_pair(length, div, seed=42)

            t_full     = timeit(lambda: aligner.full_nw_align(seq1, seq2))
            t_fixed30  = timeit(lambda: aligner.align_with_doubling(seq1, seq2, 0, 30))
            t_fixed100 = timeit(lambda: aligner.align_with_doubling(seq1, seq2, 0, 100))

            pred_centre, pred_hw = predictor.predict_single(seq1, seq2, "dna")
            t_neural   = timeit(lambda: aligner.align_with_doubling(seq1, seq2, pred_centre, pred_hw))

            speedup_full = t_full / max(t_neural, 1e-9)
            speedup_f30  = t_fixed30 / max(t_neural, 1e-9)

            rows.append({
                "length":             length,
                "divergence":         div,
                "t_full_ms":          round(t_full * 1000, 2),
                "t_fixed30_ms":       round(t_fixed30 * 1000, 2),
                "t_fixed100_ms":      round(t_fixed100 * 1000, 2),
                "t_neural_ms":        round(t_neural * 1000, 2),
                "speedup_vs_full":    round(speedup_full, 1),
                "speedup_vs_fixed30": round(speedup_f30, 2),
                "pred_hw":            pred_hw,
            })
            print(f"  len={length:5d}, div={div:.0%}: "
                  f"full={t_full*1000:.1f}ms, neural={t_neural*1000:.1f}ms, "
                  f"speedup={speedup_full:.1f}x")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, "pairwise_speedup.csv")
    df.to_csv(csv_path, index=False)

    return {
        "mean_speedup_vs_full":    float(df["speedup_vs_full"].mean()),
        "max_speedup_vs_full":     float(df["speedup_vs_full"].max()),
        "mean_speedup_vs_fixed30": float(df["speedup_vs_fixed30"].mean()),
        "results_csv":             csv_path,
        "summary":                 df.groupby("length")["speedup_vs_full"].mean().round(1).to_dict(),
    }


def exp_ablation_study(predictor, results_dir: str) -> dict:
    """
    Experiment 3: Ablation — contribution of each component.

    Methods (added one by one):
    1. Full NW (baseline)
    2. Fixed band W=50
    3. Fixed band W=100
    4. Neural band (our method)

    Metrics: time, n_doublings, band_recall.
    """
    import aligner

    DNA = "ACGT"
    rng = np.random.default_rng(0)
    # Generate 50 pairs, length=1000, div=15%
    pairs = []
    for i in range(50):
        seq1 = "".join(rng.choice(list(DNA), 1000))
        seq2 = list(seq1)
        for p in rng.choice(1000, 150, replace=False):
            seq2[p] = rng.choice([c for c in DNA if c != seq2[p]])
        pairs.append((seq1, "".join(seq2)))

    def timeit_all(fn):
        times = []
        for s1, s2 in pairs:
            t0 = time.perf_counter()
            fn(s1, s2)
            times.append(time.perf_counter() - t0)
        return float(np.mean(times)) * 1000  # ms

    t_full     = timeit_all(lambda s1, s2: aligner.full_nw_align(s1, s2))
    t_fixed50  = timeit_all(lambda s1, s2: aligner.align_with_doubling(s1, s2, 0, 50))
    t_fixed100 = timeit_all(lambda s1, s2: aligner.align_with_doubling(s1, s2, 0, 100))

    neural_times, neural_doublings = [], []
    for s1, s2 in pairs:
        c, hw = predictor.predict_single(s1, s2, "dna")
        t0 = time.perf_counter()
        r = aligner.align_with_doubling(s1, s2, c, hw)
        neural_times.append((time.perf_counter() - t0) * 1000)
        neural_doublings.append(r.n_doublings)
    t_neural = float(np.mean(neural_times))

    configs = [
        {"name": "Full NW",     "time_ms": round(t_full, 2),     "speedup": 1.0},
        {"name": "Fixed W=50",  "time_ms": round(t_fixed50, 2),  "speedup": round(t_full / max(t_fixed50, 1e-9), 1)},
        {"name": "Fixed W=100", "time_ms": round(t_fixed100, 2), "speedup": round(t_full / max(t_fixed100, 1e-9), 1)},
        {"name": "Neural band", "time_ms": round(t_neural, 2),   "speedup": round(t_full / max(t_neural, 1e-9), 1),
         "mean_doublings": round(float(np.mean(neural_doublings)), 2)},
    ]

    # Save as CSV too
    cfg_df = pd.DataFrame(configs)
    cfg_df.to_csv(os.path.join(results_dir, "ablation_study.csv"), index=False)

    for c in configs:
        doublings_str = f", doublings={c.get('mean_doublings', '-')}" if "mean_doublings" in c else ""
        print(f"  {c['name']:15s}: {c['time_ms']:7.2f}ms  speedup={c['speedup']:.1f}x{doublings_str}")

    return {
        "configs": configs,
        "neural_mean_doublings": round(float(np.mean(neural_doublings)), 3),
        "neural_zero_doubling_rate": round(float(np.mean([d == 0 for d in neural_doublings])), 3),
    }


def exp_fr_hit_ratio(results_dir: str) -> dict:
    """
    Experiment 4: Four Russians lookup table accumulation.
    Shows how hit_ratio grows with each new pair.
    """
    import aligner

    fr = aligner.FourRussiansAligner(0, False, -10.0, -0.5, 16)
    rng = np.random.default_rng(42)
    history = []

    for i in range(300):
        seq1 = "".join(rng.choice(list("ACGT"), 500))
        seq2 = "".join(rng.choice(list("ACGT"), 500))
        fr.last_row(seq1, seq2, 0, 50)

        if i % 10 == 9:
            stats = fr.get_stats()
            history.append({
                "n_pairs":   i + 1,
                "hit_ratio": round(stats.hit_ratio, 4),
                "table_kb":  fr.table_memory_bytes() // 1024,
            })
            print(f"  After {i+1:4d} pairs: hit_ratio={stats.hit_ratio:.1%}, "
                  f"table={fr.table_memory_bytes()//1024}KB")

    df = pd.DataFrame(history)
    csv_path = os.path.join(results_dir, "fr_hit_ratio.csv")
    df.to_csv(csv_path, index=False)

    return {
        "initial_hit_ratio": history[0]["hit_ratio"] if history else 0,
        "final_hit_ratio":   history[-1]["hit_ratio"] if history else 0,
        "final_table_kb":    history[-1]["table_kb"] if history else 0,
        "history_csv":       csv_path,
    }


def exp_msa_quality(predictor, balibase_groups: list, results_dir: str) -> dict:
    """
    Experiment 5: MSA quality on BAliBASE.

    Methods:
    1. MAFFT
    2. MUSCLE
    3. ClustalW
    4. Fixed band W=30 (direct aligner, no progressive MSA — ablation)
    5. Fixed band W=100 (direct aligner — ablation)
    6. Neural band (our method)
    7. Neural band + iterative refinement

    Metrics: SP-score, TC-score, time, memory.
    """
    import tracemalloc
    from msa.progressive_msa import progressive_msa
    from msa.iterative_refine import iterative_refine
    from baselines.classical import run_mafft, run_muscle, run_clustalw
    from scoring.metrics import sp_score, tc_score

    def measure(fn, seqs, ids, ref):
        """Measure SP, TC, time and memory for an MSA method."""
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            msa_result = fn(seqs, ids)
            elapsed = time.perf_counter() - t0
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Validate predicted MSA
            if not msa_result or not all(isinstance(s, str) for s in msa_result):
                return {"sp": 0, "tc": 0, "time_s": round(elapsed, 3),
                        "mem_mb": round(peak_mem / 1e6, 2), "ok": False,
                        "error": "MSA result is empty or not strings"}

            # Validate and score
            ref_valid = (ref is not None
                         and len(ref) > 0
                         and all(isinstance(s, str) for s in ref)
                         and all(len(s) == len(ref[0]) for s in ref)
                         and any('-' in s for s in ref))  # must have gaps
            if ref_valid and len(ref) == len(msa_result):
                sp = sp_score(msa_result, ref)
                tc = tc_score(msa_result, ref)
            else:
                sp = -1.0
                tc = -1.0
            return {"sp": round(sp, 4), "tc": round(tc, 4),
                    "time_s": round(elapsed, 3), "mem_mb": round(peak_mem / 1e6, 2), "ok": True}
        except Exception as e:
            tracemalloc.stop()
            return {"sp": 0, "tc": 0, "time_s": 999, "mem_mb": 0,
                    "ok": False, "error": str(e)}

    # Limit to 30 groups for speed
    groups = balibase_groups[:30]

    # Detect seq_type: BAliBASE is mostly protein
    def _detect_seq_type(seqs: list[str]) -> str:
        sample = "".join(s[:100] for s in seqs[:5]).upper()
        non_dna = sum(1 for c in sample if c not in "ACGTNU-")
        return "protein" if non_dna > len(sample) * 0.1 else "dna"

    # For fixed-band experiments, use aligner directly with pairwise alignment
    # (this is an ablation baseline — not full progressive MSA)
    import aligner

    def fixed_band_pairwise(seqs, ids, hw=30):
        """Simple pairwise-based MSA with fixed band (ablation only for pairs)."""
        if len(seqs) <= 1:
            return seqs
        st = _detect_seq_type(seqs)
        class FixedPredictor:
            def predict_single(self, s1, s2, seq_type="dna"):
                return (0, hw)
            def predict_batch(self, pairs, seq_type="dna"):
                return [(0, hw) for _ in pairs]
        return progressive_msa(seqs, ids, FixedPredictor(), seq_type=st)

    # --- DIAGNOSTIC RUN ---
    first_group = groups[0]
    import subprocess, tempfile, os as _os
    from Bio import SeqIO, AlignIO
    from io import StringIO

    # Run MAFFT directly
    _diag_seqs = first_group["sequences"]
    _diag_ref  = first_group["reference"]
    _diag_ids  = first_group["seq_ids"]

    # Write input FASTA
    _fasta_in = "\n".join(f">{_diag_ids[i]}\n{_diag_seqs[i]}" for i in range(len(_diag_seqs)))
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta',
                                     delete=False) as _f:
        _f.write(_fasta_in)
        _tmp_in = _f.name

    # Run MAFFT
    _result = subprocess.run(
        ["mafft", "--auto", "--quiet", _tmp_in],
        capture_output=True, text=True
    )
    _mafft_output = _result.stdout

    # Parse MAFFT output
    _records = list(SeqIO.parse(StringIO(_mafft_output), "fasta"))
    _predicted = [str(r.seq).upper() for r in _records]
    _pred_ids  = [r.id for r in _records]

    print("="*60)
    print(f"GROUP: {first_group['group_id']}")
    print(f"N sequences: {len(_diag_seqs)}")
    print(f"Reference sequences: {len(_diag_ref)}")
    print(f"Predicted sequences: {len(_predicted)}")
    print(f"Ref seq IDs:  {first_group['seq_ids'][:3]}")
    print(f"Pred seq IDs: {_pred_ids[:3]}")
    print(f"Ref lengths (unique):  {sorted(set(len(s) for s in _diag_ref))}")
    print(f"Pred lengths (unique): {sorted(set(len(s) for s in _predicted))}")
    print(f"Ref[0][:80]:  {_diag_ref[0][:80]}")
    print(f"Pred[0][:80]: {_predicted[0][:80]}")
    print(f"Ref gaps:  {[s.count('-') for s in _diag_ref]}")
    print(f"Pred gaps: {[s.count('-') for s in _predicted]}")
    print("="*60)
    _os.unlink(_tmp_in)
    # --- END DIAGNOSTIC ---

    methods = {}

    # Classical baselines (may fail if tools not installed)
    try:
        from baselines.classical import run_mafft
        methods["MAFFT"] = lambda s, ids: run_mafft(s, ids)
    except Exception:
        pass
    try:
        from baselines.classical import run_muscle
        methods["MUSCLE"] = lambda s, ids: run_muscle(s, ids)
    except Exception:
        pass
    try:
        from baselines.classical import run_clustalw
        methods["ClustalW"] = lambda s, ids: run_clustalw(s, ids)
    except Exception:
        pass

    # Fixed band ablations
    methods["Fixed_W30"]  = lambda s, ids: fixed_band_pairwise(s, ids, hw=30)
    methods["Fixed_W100"] = lambda s, ids: fixed_band_pairwise(s, ids, hw=100)

    # Neural methods (auto-detect seq_type per group)
    def neural_band(s, ids):
        st = _detect_seq_type(s)
        return progressive_msa(s, ids, predictor, seq_type=st)

    def neural_refine(s, ids):
        st = _detect_seq_type(s)
        msa = progressive_msa(s, ids, predictor, seq_type=st)
        return iterative_refine(msa, s, predictor)

    methods["Neural_band"]     = neural_band
    methods["Neural_+_refine"] = neural_refine

    all_rows = []
    for method_name, method_fn in methods.items():
        print(f"\n  Running {method_name}...")
        for g in groups:
            r = measure(method_fn, g["sequences"], g["seq_ids"], g["reference"])
            r["method"] = method_name
            r["group_id"] = g["group_id"]
            r["ref_class"] = g.get("ref_class", "")
            r["n_seqs"] = len(g["sequences"])
            all_rows.append(r)
            if r["ok"]:
                print(f"    {g['group_id']}: SP={r['sp']:.3f}, TC={r['tc']:.3f}, "
                      f"t={r['time_s']:.2f}s")
            else:
                print(f"    {g['group_id']}: ERROR: {r.get('error', 'unknown')}")

    df = pd.DataFrame(all_rows)
    detail_csv = os.path.join(results_dir, "msa_quality_detail.csv")
    df.to_csv(detail_csv, index=False)

    # Summary table
    summary_data = {}
    if len(df[df["ok"]]) > 0:
        summary = df[df["ok"]].groupby("method").agg(
            SP_mean=("sp", "mean"),
            TC_mean=("tc", "mean"),
            Time_mean=("time_s", "mean"),
            Mem_MB_mean=("mem_mb", "mean"),
        ).round(4)
        summary_csv = os.path.join(results_dir, "msa_quality_summary.csv")
        summary.to_csv(summary_csv)
        summary_data = summary.to_dict()
        print("\n  Summary:")
        print(summary.to_string())

    return {
        "summary": summary_data,
        "n_groups": len(groups),
        "detail_csv": detail_csv,
    }


def exp_scaling_by_n(predictor, results_dir: str) -> dict:
    """
    Experiment 6: Scaling by number of sequences N.
    Compare MAFFT vs Neural band at N = 10, 20, 50, 100.
    """
    from msa.progressive_msa import progressive_msa
    from baselines.classical import run_mafft

    DNA = "ACGT"
    rng = np.random.default_rng(0)

    def gen_seqs(n, length=300, div=0.15):
        base = "".join(rng.choice(list(DNA), length))
        seqs = []
        for _ in range(n):
            s = list(base)
            n_mut = int(length * div)
            for p in rng.choice(length, min(n_mut, length), replace=False):
                s[p] = rng.choice([c for c in DNA if c != s[p]])
            seqs.append("".join(s))
        return seqs

    rows = []
    for n in [10, 20, 50, 100]:
        seqs = gen_seqs(n)
        ids = [f"seq{i}" for i in range(n)]

        # MAFFT
        t_mafft = None
        try:
            t0 = time.perf_counter()
            run_mafft(seqs, ids)
            t_mafft = time.perf_counter() - t0
        except Exception as e:
            print(f"  MAFFT failed for N={n}: {e}")

        # Neural
        t0 = time.perf_counter()
        progressive_msa(seqs, ids, predictor)
        t_neural = time.perf_counter() - t0

        speedup = round(t_mafft / t_neural, 2) if t_mafft else None
        rows.append({
            "n_seqs":     n,
            "t_mafft_s":  round(t_mafft, 2) if t_mafft else None,
            "t_neural_s": round(t_neural, 2),
            "speedup":    speedup,
        })
        mafft_str = f"{t_mafft:.2f}s" if t_mafft else "N/A"
        print(f"  N={n:4d}: MAFFT={mafft_str}, Neural={t_neural:.2f}s")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, "scaling_by_n.csv")
    df.to_csv(csv_path, index=False)

    return {
        "rows": rows,
        "csv": csv_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Run all MSA experiments")
    parser.add_argument("--checkpoint",    default="checkpoints/best_model.pt")
    parser.add_argument("--balibase_dir",  default=None,
                        help="Path to DATASET-BALiBASE directory")
    parser.add_argument("--results_dir",   default="results/experiments")
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--skip",          nargs="*", default=[],
                        help="Experiments to skip, e.g. --skip msa_quality scaling")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    predictor = None
    model_available = False
    try:
        from model.evaluate import BandPredictorInference
        predictor = BandPredictorInference(args.checkpoint, device=args.device)
        print("Model loaded OK")
        model_available = True
    except Exception as e:
        print(f"WARNING: Could not load model: {e}")
        print("Experiments requiring neural network will be skipped")

    # Load BAliBASE data
    balibase_groups = []
    balibase_available = False
    if args.balibase_dir and os.path.isdir(args.balibase_dir):
        try:
            from data.loaders import BAliBASELoader
            loader = BAliBASELoader(args.balibase_dir)
            all_groups = loader.load_all()
            _, _, balibase_groups = loader.train_val_test_split()
            if not balibase_groups:
                balibase_groups = all_groups[:30]
            print(f"BAliBASE test: {len(balibase_groups)} groups")
            balibase_available = True
        except Exception as e:
            print(f"WARNING: Could not load BAliBASE: {e}")
    else:
        print("BAliBASE directory not provided or not found - MSA quality experiment skipped")

    # Run all experiments
    all_results = {}

    # Experiment 1: Band prediction accuracy (needs model + val.parquet)
    if "band_accuracy" not in args.skip and model_available:
        if os.path.exists("data/processed/val.parquet"):
            all_results["band_prediction_accuracy"] = run_experiment(
                "band_prediction_accuracy",
                lambda: exp_band_prediction_accuracy(predictor, args.results_dir),
                args.results_dir
            )
        else:
            print("\n  SKIP band_prediction_accuracy: val.parquet not found")

    # Experiment 2: Pairwise speedup (needs model)
    if "speedup" not in args.skip and model_available:
        all_results["pairwise_speedup"] = run_experiment(
            "pairwise_speedup",
            lambda: exp_pairwise_speedup(predictor, args.results_dir),
            args.results_dir
        )

    # Experiment 3: Ablation study (needs model)
    if "ablation" not in args.skip and model_available:
        all_results["ablation_study"] = run_experiment(
            "ablation_study",
            lambda: exp_ablation_study(predictor, args.results_dir),
            args.results_dir
        )

    # Experiment 4: Four Russians hit ratio (no model needed)
    if "fr" not in args.skip:
        all_results["fr_hit_ratio"] = run_experiment(
            "fr_hit_ratio",
            lambda: exp_fr_hit_ratio(args.results_dir),
            args.results_dir
        )

    # Experiment 5: MSA quality on BAliBASE (needs model + BAliBASE)
    if "msa_quality" not in args.skip and model_available and balibase_available:
        all_results["msa_quality"] = run_experiment(
            "msa_quality",
            lambda: exp_msa_quality(predictor, balibase_groups, args.results_dir),
            args.results_dir
        )

    # Experiment 6: Scaling by N (needs model)
    if "scaling" not in args.skip and model_available:
        all_results["scaling_by_n"] = run_experiment(
            "scaling_by_n",
            lambda: exp_scaling_by_n(predictor, args.results_dir),
            args.results_dir
        )

    # Save summary JSON
    summary_path = os.path.join(args.results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS DONE")
    print(f"Summary: {summary_path}")
    print(f"{'='*60}")

    # Status for each experiment
    for name, result in all_results.items():
        status = result.get("_status", "?")
        elapsed = result.get("_elapsed_s", 0)
        mark = "OK" if status == "OK" else "FAIL"
        print(f"  [{mark:4s}] {name:35s} {elapsed:.1f}s")


if __name__ == "__main__":
    main()
