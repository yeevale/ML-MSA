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

echo "Data generation complete."

# --------------------------------------------------------------------------
# STEP 2: Neural Network Training
# --------------------------------------------------------------------------
echo ""
echo "=== STEP 2: Neural Network Training ==="
echo "Started: $(date)"

# Clear old checkpoints from previous runs (keep feature cache!)
echo "Clearing old checkpoints..."
rm -rf checkpoints/*.pt

# Precompute features in parallel (uses all CPU cores, ~10min one-time)
echo "Precomputing train features..."
python -c "
from model.train import precompute_features
import os
precompute_features('$TRAIN_PARQUET', 'data/cache/train', n_workers=min(16, os.cpu_count() or 4))
" 2>&1 | tee logs/precompute_train.log

echo "Precomputing val features..."
python -c "
import glob, os
from model.train import precompute_features
val_files = sorted(glob.glob('data/processed/val*.parquet'))
if val_files:
    precompute_features(val_files[0], 'data/cache/val', n_workers=min(16, os.cpu_count() or 4))
" 2>&1 | tee logs/precompute_val.log

python -m model.train \
    --data_dir data/processed \
    --train_parquet $TRAIN_PARQUET \
    --cache_dir data/cache \
    --checkpoint_dir checkpoints \
    --results_dir results/training \
    --epochs_pretrain 5 \
    --batch_size 512 \
    --num_workers 16 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --patience 10 \
    --device cuda \
    2>&1 | tee logs/training.log

echo "Training complete: $(date)"

# --------------------------------------------------------------------------
# STEP 3: Run All Experiments
# --------------------------------------------------------------------------
echo ""
echo "=== STEP 3: Running All Experiments ==="
echo "Started: $(date)"

python experiments/run_all.py \
    --checkpoint checkpoints/best_model.pt \
    --results_dir results/experiments \
    --device cuda \
    2>&1 | tee logs/experiments.log

echo "Experiments complete: $(date)"

# --------------------------------------------------------------------------
# STEP 4: Full Test Suite
# --------------------------------------------------------------------------
echo ""
echo "=== STEP 4: Full Test Suite ==="

python -m pytest tests/ \
    -v --tb=short \
    2>&1 | tee logs/test_full.log

# --------------------------------------------------------------------------
# STEP 5: Generate Final Report
# --------------------------------------------------------------------------
echo ""
echo "=== STEP 5: Generating Final Report ==="

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
