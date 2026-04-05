#!/bin/bash
# =============================================================================
# vastai_run.sh — Training + experiments runner
# Run AFTER vastai_setup.sh:
#   bash vastai_run.sh 2>&1 | tee logs/full_run.log
# =============================================================================

set -eo pipefail
echo "=========================================="
echo "  MSA Band Prediction — Full Run"
echo "  Started: $(date)"
echo "=========================================="

# --------------------------------------------------------------------------
# STEP 0: Save system info
# --------------------------------------------------------------------------
echo ""
echo "=== STEP 0: Saving system info ==="
python save_system_info.py 2>&1 | tee logs/system_info.log

# --------------------------------------------------------------------------
# STEP 1: Data Generation
# --------------------------------------------------------------------------
echo ""
echo "=== STEP 1: Data Generation ==="

# Check current dataset size
CURRENT_TRAIN=$(python -c "
import pandas as pd, os
if os.path.exists('data/processed/train.parquet'):
    print(len(pd.read_parquet('data/processed/train.parquet')))
else:
    print(0)
")
echo "Current train size: $CURRENT_TRAIN"

TRAIN_PARQUET="data/processed/train.parquet"

# If less than 400k — generate more
if [ "$CURRENT_TRAIN" -lt "400000" ]; then
    echo "Generating additional DNA training data (target: 500k total)..."
    NEEDED=$((500000 - CURRENT_TRAIN))
    python -m data.simulate \
        --n_samples $NEEDED \
        --seq_type dna \
        --output data/processed/train_extra.parquet \
        --n_workers $(nproc) \
        --seed 123 \
        2>&1 | tee logs/data_gen_dna.log

    # Combine DNA into single file, remove extras to avoid glob collisions
    python -c "
import pandas as pd, os
dfs = []
for f in ['data/processed/train.parquet', 'data/processed/train_extra.parquet']:
    if os.path.exists(f):
        dfs.append(pd.read_parquet(f))
combined = pd.concat(dfs, ignore_index=True)
combined.to_parquet('data/processed/train_full.parquet', index=False)
print(f'Combined train: {len(combined)} samples')
# Remove partial files so glob('train_*.parquet') only picks up train_full
for f in ['data/processed/train_extra.parquet']:
    if os.path.exists(f):
        os.remove(f)
        print(f'Removed {f}')
"
    TRAIN_PARQUET="data/processed/train_full.parquet"
else
    echo "Train data sufficient ($CURRENT_TRAIN samples)"
fi

# Generate protein data and MERGE into main train file
if [ ! -f "data/processed/train_protein.parquet" ]; then
    echo "Generating protein training data (200k)..."
    python -m data.simulate \
        --n_samples 200000 \
        --seq_type protein \
        --output data/processed/train_protein.parquet \
        --n_workers $(nproc) \
        --seed 456 \
        2>&1 | tee logs/data_gen_protein.log
fi

# Merge protein into the main training file
python -c "
import pandas as pd, os
train_path = '$TRAIN_PARQUET'
protein_path = 'data/processed/train_protein.parquet'
if os.path.exists(train_path) and os.path.exists(protein_path):
    df_main = pd.read_parquet(train_path)
    df_prot = pd.read_parquet(protein_path)
    combined = pd.concat([df_main, df_prot], ignore_index=True)
    combined.to_parquet('data/processed/train_combined.parquet', index=False)
    print(f'Combined DNA+protein train: {len(combined)} samples')
    # Clean up individual files, keep only train_combined
    for f in [protein_path]:
        if os.path.exists(f) and f != 'data/processed/train_combined.parquet':
            os.remove(f)
            print(f'Removed {f}')
else:
    print('Skipping protein merge (files not found)')
"
TRAIN_PARQUET="data/processed/train_combined.parquet"

# Generate validation data if missing
if [ ! -f "data/processed/val.parquet" ]; then
    echo "Generating validation data (10k DNA + 2k protein)..."
    python -m data.simulate \
        --n_samples 10000 \
        --seq_type dna \
        --output data/processed/val_dna.parquet \
        --n_workers $(nproc) \
        --seed 789 \
        2>&1 | tee logs/data_gen_val.log

    python -m data.simulate \
        --n_samples 2000 \
        --seq_type protein \
        --output data/processed/val_protein.parquet \
        --n_workers $(nproc) \
        --seed 790 \
        2>&1 | tee -a logs/data_gen_val.log

    python -c "
import pandas as pd, os
dfs = []
for f in ['data/processed/val_dna.parquet', 'data/processed/val_protein.parquet']:
    if os.path.exists(f):
        dfs.append(pd.read_parquet(f))
combined = pd.concat(dfs, ignore_index=True)
combined.to_parquet('data/processed/val.parquet', index=False)
print(f'Validation set: {len(combined)} samples')
for f in ['data/processed/val_dna.parquet', 'data/processed/val_protein.parquet']:
    if os.path.exists(f):
        os.remove(f)
"
fi

echo "Data generation complete."

# --------------------------------------------------------------------------
# STEP 2: BAliBASE Preparation (if available)
# --------------------------------------------------------------------------
echo ""
echo "=== STEP 2: BAliBASE Preparation ==="

BALIBASE_DIR="data/raw/balibase/DATASET-BALiBASE"
BALIBASE_AVAILABLE=0
if [ -d "$BALIBASE_DIR" ]; then
    echo "Converting BAliBASE to splits..."
    python -c "
from data.loaders import BAliBASELoader
import json
import pandas as pd

loader = BAliBASELoader('$BALIBASE_DIR')
groups = loader.load_all()
print(f'Loaded {len(groups)} BAliBASE groups')

train_g, val_g, test_g = loader.train_val_test_split()
print(f'Train: {len(train_g)}, Val: {len(val_g)}, Test: {len(test_g)}')

for split_name, split_data in [('train', train_g), ('val', val_g), ('test', test_g)]:
    # Save JSON (for reference)
    with open(f'data/balibase_{split_name}.json', 'w') as f:
        json.dump(split_data, f, default=str)
    print(f'Saved balibase_{split_name}.json')

# Also save train split as parquet for model.train Stage 2
rows = []
for g in train_g:
    seqs = g['sequences']
    ids = g.get('seq_ids', [f'seq{i}' for i in range(len(seqs))])
    # Create all pairwise combinations
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            s1, s2 = seqs[i], seqs[j]
            n = max(len(s1), len(s2))
            # Simple divergence estimate
            min_len = min(len(s1), len(s2))
            mismatches = sum(1 for a, b in zip(s1[:min_len], s2[:min_len]) if a != b)
            div = mismatches / max(min_len, 1)
            # seq_type: detect protein (has non-ACGT chars)
            charset = set(s1.upper() + s2.upper())
            is_protein = bool(charset - set('ACGTNU-'))
            rows.append({
                'seq1': s1, 'seq2': s2,
                'centre_diag': 0,
                'true_half_width': max(10, int(n * 0.15)),
                'divergence': round(div, 4),
                'seq_type': 'protein' if is_protein else 'dna',
            })
if rows:
    df = pd.DataFrame(rows)
    df.to_parquet('data/balibase_train.parquet', index=False)
    print(f'Saved balibase_train.parquet ({len(df)} pairs)')
else:
    print('WARNING: No BAliBASE pairs generated')
" 2>&1 | tee logs/balibase_prep.log
    BALIBASE_AVAILABLE=1
else
    echo "BAliBASE not found - MSA quality experiments will be skipped"
fi

# --------------------------------------------------------------------------
# STEP 3: Neural Network Training (Stage 1 — Synthetic)
# --------------------------------------------------------------------------
echo ""
echo "=== STEP 3: Neural Network Training (Stage 1 - Synthetic) ==="
echo "Started: $(date)"

# Clear stale cache and old checkpoints from previous (broken) runs
echo "Clearing stale cache and old checkpoints..."
rm -rf data/cache/train data/cache/val checkpoints/*.pt

python -m model.train \
    --data_dir data/processed \
    --train_parquet $TRAIN_PARQUET \
    --cache_dir data/cache \
    --checkpoint_dir checkpoints \
    --results_dir results/training \
    --epochs_pretrain 20 \
    --epochs_finetune 0 \
    --batch_size 512 \
    --num_workers 4 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --patience 5 \
    --device cuda \
    2>&1 | tee logs/training_stage1.log

echo "Stage 1 complete: $(date)"

# --------------------------------------------------------------------------
# STEP 4: Fine-tuning on BAliBASE (Stage 2 — if available)
# --------------------------------------------------------------------------
if [ "$BALIBASE_AVAILABLE" -eq "1" ]; then
    echo ""
    echo "=== STEP 4: Fine-tuning on BAliBASE (Stage 2) ==="
    echo "Started: $(date)"

    python -m model.train \
        --data_dir data/processed \
        --train_parquet $TRAIN_PARQUET \
        --balibase_parquet data/balibase_train.parquet \
        --cache_dir data/cache \
        --checkpoint_dir checkpoints \
        --results_dir results/training \
        --resume checkpoints/best_model.pt \
        --epochs_pretrain 0 \
        --epochs_finetune 10 \
        --lr 1e-4 \
        --batch_size 128 \
        --num_workers 4 \
        --patience 5 \
        --device cuda \
        2>&1 | tee logs/training_stage2.log

    echo "Stage 2 complete: $(date)"
else
    echo ""
    echo "=== STEP 4: Skipped (no BAliBASE) ==="
fi

# --------------------------------------------------------------------------
# STEP 5: Run All Experiments
# --------------------------------------------------------------------------
echo ""
echo "=== STEP 5: Running All Experiments ==="
echo "Started: $(date)"

BALIBASE_TEST_ARG=""
if [ "$BALIBASE_AVAILABLE" -eq "1" ]; then
    BALIBASE_TEST_ARG="--balibase_dir $BALIBASE_DIR"
fi

python experiments/run_all.py \
    --checkpoint checkpoints/best_model.pt \
    $BALIBASE_TEST_ARG \
    --results_dir results/experiments \
    --device cuda \
    2>&1 | tee logs/experiments.log

echo "Experiments complete: $(date)"

# --------------------------------------------------------------------------
# STEP 6: Full Test Suite
# --------------------------------------------------------------------------
echo ""
echo "=== STEP 6: Full Test Suite ==="

pip install --quiet pytest-json-report 2>/dev/null || true

python -m pytest tests/ \
    -v --tb=short \
    --json-report --json-report-file=results/tests/full_test_report.json \
    2>&1 | tee logs/test_full.log

# Fallback: if pytest-json-report is not available, generate basic report
if [ ! -f "results/tests/full_test_report.json" ]; then
    python -c "
import json, subprocess, re
output = open('logs/test_full.log').read()
# Parse basic summary from pytest output
passed = len(re.findall(r'PASSED', output))
failed = len(re.findall(r'FAILED', output))
skipped = len(re.findall(r'SKIPPED', output))
total = passed + failed + skipped
report = {
    'summary': {
        'total': total,
        'passed': passed,
        'failed': failed,
        'skipped': skipped,
        'duration': 0,
    }
}
with open('results/tests/full_test_report.json', 'w') as f:
    json.dump(report, f, indent=2)
print(f'Generated test report: {total} total, {passed} passed, {failed} failed, {skipped} skipped')
"
fi

# --------------------------------------------------------------------------
# STEP 7: Generate Final Report
# --------------------------------------------------------------------------
echo ""
echo "=== STEP 7: Generating Final Report ==="

python results/interpret_results.py \
    --results_dir results \
    --output results/FINAL_REPORT.md \
    2>&1 | tee logs/report.log

echo ""
echo "=========================================="
echo "  ALL DONE: $(date)"
echo "  Results in: results/"
echo "  Model: checkpoints/best_model.pt"
echo "  Report: results/FINAL_REPORT.md"
echo "=========================================="
echo ""
echo "Next step: bash vastai_download.sh"
