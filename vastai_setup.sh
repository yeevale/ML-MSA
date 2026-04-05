#!/bin/bash
# =============================================================================
# vastai_setup.sh — Full vast.ai server setup from scratch
# Run ONCE after connecting to server:
#   bash vastai_setup.sh
# =============================================================================

set -e  # stop on any error
echo "=========================================="
echo "  MSA Band Prediction — vast.ai Setup"
echo "=========================================="

# --------------------------------------------------------------------------
# 1. System dependencies
# --------------------------------------------------------------------------
echo "[1/8] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    build-essential \
    cmake \
    gcc \
    g++ \
    git \
    wget \
    curl \
    mafft \
    muscle \
    clustalw \
    htop \
    tree

# --------------------------------------------------------------------------
# 2. Python dependencies
# --------------------------------------------------------------------------
echo "[2/8] Installing Python packages..."
pip install --quiet --upgrade pip

# PyTorch with CUDA (auto-detect version)
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2 | tr -d '.')
echo "Detected CUDA: $CUDA_VERSION"

if [ "$CUDA_VERSION" -ge "121" ]; then
    pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
elif [ "$CUDA_VERSION" -ge "118" ]; then
    pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    pip install --quiet torch torchvision
fi

# Other dependencies
pip install --quiet \
    pandas \
    pyarrow \
    scipy \
    numpy \
    pybind11 \
    tqdm \
    wandb \
    biopython \
    scikit-learn \
    matplotlib \
    seaborn \
    pytest \
    pytest-benchmark \
    tabulate

echo "Python packages installed."

# --------------------------------------------------------------------------
# 3. Build C++ aligner module
# --------------------------------------------------------------------------
echo "[3/8] Building C++ aligner module..."

# Check AVX2 support
if grep -q "avx2" /proc/cpuinfo; then
    echo "AVX2 supported - building with SIMD optimization"
    AVX2_FLAG="-DHAVE_AVX2=ON"
else
    echo "WARNING: AVX2 not supported - building without SIMD"
    AVX2_FLAG="-DHAVE_AVX2=OFF"
fi

# Build via setup.py
python setup.py build_ext --inplace 2>&1 | tail -5

# Verify module built successfully
python -c "import aligner; print('aligner module OK')" || {
    echo "ERROR: aligner module failed to build via setup.py!"
    echo "Trying cmake build..."
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release $AVX2_FLAG
    make -j$(nproc)
    cd ..
    cp build/aligner*.so . 2>/dev/null || true
    python -c "import aligner; print('aligner module OK (cmake)')"
}

# --------------------------------------------------------------------------
# 4. Create results directory structure
# --------------------------------------------------------------------------
echo "[4/8] Creating results directory structure..."
mkdir -p results/training
mkdir -p results/experiments
mkdir -p results/plots
mkdir -p results/tests
mkdir -p checkpoints
mkdir -p data/cache/train
mkdir -p data/cache/val
mkdir -p logs

# --------------------------------------------------------------------------
# 5. Check data availability
# --------------------------------------------------------------------------
echo "[5/8] Checking data..."

# Synthetic data
if [ -f "data/processed/train.parquet" ]; then
    TRAIN_SIZE=$(python -c "import pandas as pd; df=pd.read_parquet('data/processed/train.parquet'); print(len(df))")
    echo "  Synthetic train: $TRAIN_SIZE samples"
else
    echo "  WARNING: train.parquet not found! Will generate..."
fi

if [ -f "data/processed/val.parquet" ]; then
    VAL_SIZE=$(python -c "import pandas as pd; df=pd.read_parquet('data/processed/val.parquet'); print(len(df))")
    echo "  Synthetic val: $VAL_SIZE samples"
fi

# BAliBASE
if [ -d "data/raw/balibase/DATASET-BALiBASE" ]; then
    BALI_ALIGNED=$(ls data/raw/balibase/DATASET-BALiBASE/Aligned\ sequences/*.xml 2>/dev/null | wc -l)
    BALI_UNALIGNED=$(ls data/raw/balibase/DATASET-BALiBASE/Unaligned\ sequences/*.tfa 2>/dev/null | wc -l)
    echo "  BAliBASE: $BALI_ALIGNED aligned, $BALI_UNALIGNED unaligned groups"
else
    echo "  WARNING: BAliBASE not found at data/raw/balibase/DATASET-BALiBASE"
    echo "  MSA quality tests will be skipped"
fi

# --------------------------------------------------------------------------
# 6. Quick import checks
# --------------------------------------------------------------------------
echo "[6/8] Checking Python imports..."
python -c "
import torch
import numpy as np
import pandas as pd
import scipy
import Bio
import aligner
from features.profile_features import make_input
from model.band_predictor import BandPredictor
print('All imports OK')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# --------------------------------------------------------------------------
# 7. Run correctness tests (fast, no GPU needed)
# --------------------------------------------------------------------------
echo "[7/8] Running correctness tests..."
python -m pytest tests/test_correctness.py tests/test_four_russians.py \
    -v --tb=short \
    2>&1 | tee logs/test_correctness.log

echo "[8/8] Setup complete!"
echo ""
echo "Next step: bash vastai_run.sh"
echo "=========================================="
