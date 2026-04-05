#!/bin/bash
# =============================================================================
# vastai_download.sh — Download results from vast.ai to local machine
# Run LOCALLY (not on server):
#   bash vastai_download.sh <SSH_HOST> <SSH_PORT>
# Example:
#   bash vastai_download.sh ssh6.vast.ai 12345
# =============================================================================

SSH_HOST=${1:-"ssh6.vast.ai"}
SSH_PORT=${2:-"22"}
REMOTE_DIR="/root/DIPLOM"
LOCAL_DIR="./vastai_results"

echo "Downloading results from $SSH_HOST:$SSH_PORT..."
mkdir -p $LOCAL_DIR

# Download trained model (most important)
echo "[1/5] Downloading trained model..."
scp -P $SSH_PORT \
    root@$SSH_HOST:$REMOTE_DIR/checkpoints/best_model.pt \
    $LOCAL_DIR/best_model.pt

# Download all experiment results
echo "[2/5] Downloading experiment results..."
scp -P $SSH_PORT -r \
    root@$SSH_HOST:$REMOTE_DIR/results/ \
    $LOCAL_DIR/results/

# Download training logs
echo "[3/5] Downloading training logs..."
scp -P $SSH_PORT -r \
    root@$SSH_HOST:$REMOTE_DIR/logs/ \
    $LOCAL_DIR/logs/

# Download final report
echo "[4/5] Downloading final report..."
scp -P $SSH_PORT \
    root@$SSH_HOST:$REMOTE_DIR/results/FINAL_REPORT.md \
    $LOCAL_DIR/FINAL_REPORT.md

# Download plots
echo "[5/5] Downloading plots..."
scp -P $SSH_PORT -r \
    root@$SSH_HOST:$REMOTE_DIR/results/plots/ \
    $LOCAL_DIR/plots/ 2>/dev/null || echo "  No plots directory found (skipped)"

echo ""
echo "Download complete! Files in: $LOCAL_DIR"
echo ""
echo "Key files:"
echo "  Model:   $LOCAL_DIR/best_model.pt"
echo "  Report:  $LOCAL_DIR/FINAL_REPORT.md"
echo "  Results: $LOCAL_DIR/results/"
