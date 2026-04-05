#!/bin/bash
# =============================================================================
# vastai_run.sh — Training + experiments runner
# Run AFTER vastai_setup.sh:
#   bash vastai_run.sh 2>&1 | tee logs/full_run.log
# =============================================================================

set -e
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

    # Combine with existing
    python -c "
import pandas as pd, os
dfs = []
for f in ['data/processed/train.parquet', 'data/processed/train_extra.parquet']:
    if os.path.exists(f):
        dfs.append(pd.read_parquet(f))
combined = pd.concat(dfs, ignore_index=True)
combined.to_parquet('data/processed/train_full.parquet', index=False)
print(f'Combined train: {len(combined)} samples')
"
    TRAIN_PARQUET="data/processed/train_full.parquet"
else
    echo "Train data sufficient ($CURRENT_TRAIN samples)"
fi

# Generate protein data if not present
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

loader = BAliBASELoader('$BALIBASE_DIR')
groups = loader.load_all()
print(f'Loaded {len(groups)} BAliBASE groups')

train_g, val_g, test_g = loader.train_val_test_split()
print(f'Train: {len(train_g)}, Val: {len(val_g)}, Test: {len(test_g)}')

for split_name, split_data in [('train', train_g), ('val', val_g), ('test', test_g)]:
    with open(f'data/balibase_{split_name}.json', 'w') as f:
        json.dump(split_data, f, default=str)
    print(f'Saved balibase_{split_name}.json')
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

python -m model.train \
    --data_dir data/processed \
    --cache_dir data/cache \
    --checkpoint_dir checkpoints \
    --epochs_pretrain 20 \
    --epochs_finetune 0 \
    --batch_size 256 \
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
        --balibase_parquet data/balibase_train.json \
        --cache_dir data/cache \
        --checkpoint_dir checkpoints \
        --epochs_pretrain 0 \
        --epochs_finetune 10 \
        --lr 1e-4 \
        --batch_size 128 \
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

python -m pytest tests/ \
    -v --tb=short \
    2>&1 | tee logs/test_full.log

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
