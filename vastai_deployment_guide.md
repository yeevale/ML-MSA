# vast.ai Deployment Guide — MSA Neural Band Prediction
> **Для Copilot:** этот файл описывает ВСЁ что нужно сделать для деплоя на vast.ai.
> Сгенерируй все файлы описанные здесь. Читай каждый раздел полностью перед генерацией.

---

## [КОНТЕКСТ ДЛЯ COPILOT]

Проект: нейросеть предсказывает band для banded Needleman-Wunsch выравнивания (MSA).
Структура проекта описана в `code_structure_report.md`.

**Что нужно сделать:**
1. Создать скрипт `vastai_setup.sh` — полная настройка сервера с нуля
2. Создать скрипт `vastai_run.sh` — запуск обучения и всех экспериментов
3. Создать скрипт `vastai_download.sh` — скачивание результатов на локальную машину
4. Создать `experiments/run_all.py` — единая точка запуска всех экспериментов
5. Обновить `data/loaders.py` — загрузка BAliBASE из новой структуры папок
6. Создать `results/interpret_results.py` — форматирование результатов для анализа

**Датасеты:**
- Синтетика: `data/processed/train.parquet` (50k пар, уже готово) + догенерировать до 500k
- BAliBASE: `data/raw/balibase/DATASET-BALiBASE/` со структурой:
  - `Aligned sequences/*.xml` — эталонные выравнивания
  - `Unaligned sequences/*.tfa` — исходные последовательности

---

## Файл 1: `vastai_setup.sh`

```bash
#!/bin/bash
# =============================================================================
# vastai_setup.sh — Полная настройка сервера vast.ai с нуля
# Запускать ОДИН РАЗ после подключения к серверу:
#   bash vastai_setup.sh
# =============================================================================

set -e  # остановить при любой ошибке
echo "=========================================="
echo "  MSA Band Prediction — vast.ai Setup"
echo "=========================================="

# --------------------------------------------------------------------------
# 1. Системные зависимости
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
# 2. Python зависимости
# --------------------------------------------------------------------------
echo "[2/8] Installing Python packages..."
pip install --quiet --upgrade pip

# PyTorch с CUDA (определяем версию автоматически)
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2 | tr -d '.')
echo "Detected CUDA: $CUDA_VERSION"

if [ "$CUDA_VERSION" -ge "121" ]; then
    pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
elif [ "$CUDA_VERSION" -ge "118" ]; then
    pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    pip install --quiet torch torchvision
fi

# Остальные зависимости
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
# 3. Сборка C++ модуля
# --------------------------------------------------------------------------
echo "[3/8] Building C++ aligner module..."

# Проверить что AVX2 поддерживается
if grep -q "avx2" /proc/cpuinfo; then
    echo "AVX2 supported - building with SIMD optimization"
    AVX2_FLAG="-DHAVE_AVX2=ON"
else
    echo "WARNING: AVX2 not supported - building without SIMD"
    AVX2_FLAG="-DHAVE_AVX2=OFF"
fi

# Сборка через setup.py (проще чем cmake на vast.ai)
python setup.py build_ext --inplace 2>&1 | tail -5

# Проверить что модуль собрался
python -c "import aligner; print('aligner module OK')" || {
    echo "ERROR: aligner module failed to build!"
    echo "Trying cmake build..."
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release $AVX2_FLAG
    make -j$(nproc)
    cd ..
    cp build/aligner*.so . 2>/dev/null || true
    python -c "import aligner; print('aligner module OK (cmake)')"
}

# --------------------------------------------------------------------------
# 4. Создать структуру папок для результатов
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
# 5. Проверить наличие данных
# --------------------------------------------------------------------------
echo "[5/8] Checking data..."

# Синтетические данные
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
# 6. Быстрая проверка импортов
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
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# --------------------------------------------------------------------------
# 7. Запустить тесты корректности (быстрые, без GPU)
# --------------------------------------------------------------------------
echo "[7/8] Running correctness tests..."
python -m pytest tests/test_correctness.py tests/test_four_russians.py \
    -v --tb=short \
    --json-report --json-report-file=results/tests/correctness_tests.json \
    2>&1 | tee logs/test_correctness.log

echo "[8/8] Setup complete!"
echo ""
echo "Next step: bash vastai_run.sh"
echo "=========================================="
```

---

## Файл 2: `vastai_run.sh`

```bash
#!/bin/bash
# =============================================================================
# vastai_run.sh — Запуск обучения и всех экспериментов
# Запускать ПОСЛЕ vastai_setup.sh:
#   bash vastai_run.sh 2>&1 | tee logs/full_run.log
# =============================================================================

set -e
echo "=========================================="
echo "  MSA Band Prediction — Full Run"
echo "  Started: $(date)"
echo "=========================================="

# --------------------------------------------------------------------------
# ШАГ 1: Генерация синтетических данных
# --------------------------------------------------------------------------
echo ""
echo "=== STEP 1: Data Generation ==="

# Проверить текущий размер датасета
CURRENT_TRAIN=$(python -c "
import pandas as pd, os
if os.path.exists('data/processed/train.parquet'):
    print(len(pd.read_parquet('data/processed/train.parquet')))
else:
    print(0)
")
echo "Current train size: $CURRENT_TRAIN"

# Если меньше 400k — догенерировать
if [ "$CURRENT_TRAIN" -lt "400000" ]; then
    echo "Generating additional DNA training data (target: 500k total)..."
    NEEDED=$((500000 - CURRENT_TRAIN))
    python -m data.simulate \
        --n_samples $NEEDED \
        --seq_type dna \
        --max_length 2000 \
        --output data/processed/train_extra.parquet \
        --n_workers $(nproc) \
        2>&1 | tee logs/data_gen_dna.log

    # Объединить с существующим
    python -c "
import pandas as pd
dfs = []
import os
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
    TRAIN_PARQUET="data/processed/train.parquet"
fi

# Генерация белковых данных
if [ ! -f "data/processed/train_protein.parquet" ]; then
    echo "Generating protein training data (200k)..."
    python -m data.simulate \
        --n_samples 200000 \
        --seq_type protein \
        --max_length 500 \
        --output data/processed/train_protein.parquet \
        --n_workers $(nproc) \
        2>&1 | tee logs/data_gen_protein.log
fi

echo "Data generation complete."

# --------------------------------------------------------------------------
# ШАГ 2: Конвертация BAliBASE в parquet (если доступен)
# --------------------------------------------------------------------------
echo ""
echo "=== STEP 2: BAliBASE Preparation ==="

BALIBASE_DIR="data/raw/balibase/DATASET-BALiBASE"
if [ -d "$BALIBASE_DIR" ]; then
    echo "Converting BAliBASE to parquet..."
    python -c "
from data.loaders import BAliBASELoader
import pandas as pd

loader = BAliBASELoader('$BALIBASE_DIR')
groups = loader.load_all()
print(f'Loaded {len(groups)} BAliBASE groups')

train_g, val_g, test_g = loader.train_val_test_split()
print(f'Train: {len(train_g)}, Val: {len(val_g)}, Test: {len(test_g)}')

# Сохранить сплиты как JSON для экспериментов
import json
for split_name, split_data in [('train', train_g), ('val', val_g), ('test', test_g)]:
    with open(f'data/balibase_{split_name}.json', 'w') as f:
        json.dump(split_data, f)
    print(f'Saved balibase_{split_name}.json')
" 2>&1 | tee logs/balibase_prep.log
    BALIBASE_AVAILABLE=1
else
    echo "BAliBASE not found - MSA quality experiments will be skipped"
    BALIBASE_AVAILABLE=0
fi

# --------------------------------------------------------------------------
# ШАГ 3: Обучение нейросети (Stage 1 — синтетика)
# --------------------------------------------------------------------------
echo ""
echo "=== STEP 3: Neural Network Training (Stage 1 - Synthetic) ==="
echo "Started: $(date)"

python -m model.train \
    --data_dir data/processed \
    --train_parquet $TRAIN_PARQUET \
    --val_parquet data/processed/val.parquet \
    --cache_dir data/cache \
    --checkpoint_dir checkpoints \
    --epochs_pretrain 20 \
    --epochs_finetune 0 \
    --batch_size 256 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --patience 5 \
    --device cuda \
    --results_dir results/training \
    2>&1 | tee logs/training_stage1.log

echo "Stage 1 complete: $(date)"

# --------------------------------------------------------------------------
# ШАГ 4: Дообучение на BAliBASE (Stage 2 — если доступен)
# --------------------------------------------------------------------------
if [ "$BALIBASE_AVAILABLE" -eq "1" ]; then
    echo ""
    echo "=== STEP 4: Fine-tuning on BAliBASE (Stage 2) ==="
    echo "Started: $(date)"

    python -m model.train \
        --data_dir data/processed \
        --train_parquet $TRAIN_PARQUET \
        --val_parquet data/processed/val.parquet \
        --balibase_train data/balibase_train.json \
        --balibase_val data/balibase_val.json \
        --cache_dir data/cache \
        --checkpoint_dir checkpoints \
        --epochs_pretrain 0 \
        --epochs_finetune 10 \
        --lr 1e-4 \
        --batch_size 128 \
        --patience 5 \
        --device cuda \
        --resume checkpoints/best_model.pt \
        --results_dir results/training \
        2>&1 | tee logs/training_stage2.log

    echo "Stage 2 complete: $(date)"
else
    echo ""
    echo "=== STEP 4: Skipped (no BAliBASE) ==="
fi

# --------------------------------------------------------------------------
# ШАГ 5: Полный набор экспериментов
# --------------------------------------------------------------------------
echo ""
echo "=== STEP 5: Running All Experiments ==="
echo "Started: $(date)"

python experiments/run_all.py \
    --checkpoint checkpoints/best_model.pt \
    --balibase_test data/balibase_test.json \
    --results_dir results/experiments \
    --device cuda \
    2>&1 | tee logs/experiments.log

echo "Experiments complete: $(date)"

# --------------------------------------------------------------------------
# ШАГ 6: Финальные тесты
# --------------------------------------------------------------------------
echo ""
echo "=== STEP 6: Full Test Suite ==="

python -m pytest tests/ \
    -v --tb=short \
    --checkpoint checkpoints/best_model.pt \
    --balibase_dir "$BALIBASE_DIR" \
    --json-report --json-report-file=results/tests/full_test_report.json \
    2>&1 | tee logs/test_full.log

# --------------------------------------------------------------------------
# ШАГ 7: Генерация итогового отчёта
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
```

---

## Файл 3: `vastai_download.sh`

```bash
#!/bin/bash
# =============================================================================
# vastai_download.sh — Скачать результаты с vast.ai на локальную машину
# Запускать ЛОКАЛЬНО (не на сервере):
#   bash vastai_download.sh <SSH_HOST> <SSH_PORT>
# Пример:
#   bash vastai_download.sh ssh6.vast.ai 12345
# =============================================================================

SSH_HOST=${1:-"ssh6.vast.ai"}
SSH_PORT=${2:-"22"}
REMOTE_DIR="/root/DIPLOM"   # путь на сервере
LOCAL_DIR="./vastai_results" # папка куда скачивать

echo "Downloading results from $SSH_HOST:$SSH_PORT..."
mkdir -p $LOCAL_DIR

# Скачать обученную модель (самое важное)
echo "[1/5] Downloading trained model..."
scp -P $SSH_PORT \
    root@$SSH_HOST:$REMOTE_DIR/checkpoints/best_model.pt \
    $LOCAL_DIR/best_model.pt

# Скачать все результаты экспериментов
echo "[2/5] Downloading experiment results..."
scp -P $SSH_PORT -r \
    root@$SSH_HOST:$REMOTE_DIR/results/ \
    $LOCAL_DIR/results/

# Скачать логи обучения
echo "[3/5] Downloading training logs..."
scp -P $SSH_PORT -r \
    root@$SSH_HOST:$REMOTE_DIR/logs/ \
    $LOCAL_DIR/logs/

# Скачать итоговый отчёт
echo "[4/5] Downloading final report..."
scp -P $SSH_PORT \
    root@$SSH_HOST:$REMOTE_DIR/results/FINAL_REPORT.md \
    $LOCAL_DIR/FINAL_REPORT.md

# Скачать графики
echo "[5/5] Downloading plots..."
scp -P $SSH_PORT -r \
    root@$SSH_HOST:$REMOTE_DIR/results/plots/ \
    $LOCAL_DIR/plots/

echo ""
echo "Download complete! Files in: $LOCAL_DIR"
echo ""
echo "Key files:"
echo "  Model:   $LOCAL_DIR/best_model.pt"
echo "  Report:  $LOCAL_DIR/FINAL_REPORT.md"
echo "  Results: $LOCAL_DIR/results/"
```

---

## Файл 4: `experiments/run_all.py`

```python
#!/usr/bin/env python3
"""
Единая точка запуска всех экспериментов.
Запускать после обучения модели:
    python experiments/run_all.py \
        --checkpoint checkpoints/best_model.pt \
        --balibase_test data/balibase_test.json \
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

# Добавить корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_experiment(name: str, fn, results_dir: str) -> dict:
    """Запустить один эксперимент с обработкой ошибок."""
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

        # Сохранить результат
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


def exp_band_prediction_accuracy(predictor, test_pairs: list) -> dict:
    """
    Эксперимент 1: точность предсказания band нейросетью.

    Для каждой тестовой пары:
    - предсказываем (centre_diag, half_width)
    - сравниваем с true_half_width из парквет файла
    - считаем band_recall@1x, @1.5x, @2x

    Группируем по уровню дивергенции: low/medium/high.
    """
    import aligner
    import pandas as pd

    df = pd.read_parquet("data/processed/val.parquet")
    # Взять по 500 примеров из каждой группы
    results = []
    for div_group, div_range in [
        ("low",    (0.0,  0.10)),
        ("medium", (0.10, 0.25)),
        ("high",   (0.25, 0.50)),
    ]:
        subset = df[
            (df["divergence"] >= div_range[0]) &
            (df["divergence"] <  div_range[1])
        ].sample(min(500, len(df)), random_state=42)

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

    # Сохранить детальные результаты
    res_df.to_csv("results/experiments/band_prediction_detail.csv", index=False)

    # Агрегировать по группам
    summary = res_df.groupby("div_group").agg(
        recall_1x=("recall_1x", "mean"),
        recall_1_5x=("recall_1_5x", "mean"),
        recall_2x=("recall_2x", "mean"),
        mae_centre=("centre_err", "mean"),
        width_ratio=("width_ratio", "mean"),
        n_samples=("div_group", "count"),
    ).round(4).to_dict()

    return {
        "by_divergence_group": summary,
        "overall_recall_1x":   float(res_df["recall_1x"].mean()),
        "overall_recall_1_5x": float(res_df["recall_1_5x"].mean()),
        "overall_recall_2x":   float(res_df["recall_2x"].mean()),
        "overall_mae_centre":  float(res_df["centre_err"].mean()),
        "overall_width_ratio": float(res_df["width_ratio"].mean()),
        "n_total":             len(res_df),
    }


def exp_pairwise_speedup(predictor) -> dict:
    """
    Эксперимент 2: speedup попарного выравнивания.

    Сетка: 5 длин × 4 уровня дивергенции × 4 метода.
    Методы: Full NW | Fixed W=30 | Fixed W=100 | Neural band.
    Каждая точка усредняется по 10 запускам.
    """
    import aligner
    import time
    import numpy as np

    DNA = "ACGT"

    def gen_pair(length, divergence, seed):
        rng = np.random.default_rng(seed)
        seq1 = "".join(rng.choice(list(DNA), length))
        seq2 = list(seq1)
        n_mut = int(length * divergence)
        pos = rng.choice(length, n_mut, replace=False)
        for p in pos:
            seq2[p] = rng.choice([c for c in DNA if c != seq2[p]])
        n_indel = max(1, int(n_mut * 0.3))
        ins_pos = sorted(rng.choice(len(seq2), n_indel // 2, replace=False), reverse=True)
        for p in ins_pos:
            seq2.insert(p, rng.choice(list(DNA)))
        del_pos = sorted(rng.choice(len(seq2), n_indel // 2, replace=False), reverse=True)
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

    configs = [
        (300,   0.05),
        (500,   0.10),
        (1000,  0.15),
        (2000,  0.20),
        (5000,  0.10),
    ]
    divs = [0.05, 0.10, 0.20, 0.30]
    rows = []

    for length in [300, 500, 1000, 2000, 5000]:
        for div in divs:
            seq1, seq2 = gen_pair(length, div, seed=42)
            true_hw = int(length * div * 1.5) + 5

            t_full    = timeit(lambda: aligner.full_nw_align(seq1, seq2))
            t_fixed30 = timeit(lambda: aligner.align_with_doubling(seq1, seq2, 0, 30))
            t_fixed100= timeit(lambda: aligner.align_with_doubling(seq1, seq2, 0, 100))

            pred_centre, pred_hw = predictor.predict_single(seq1, seq2, "dna")
            t_neural  = timeit(lambda: aligner.align_with_doubling(seq1, seq2, pred_centre, pred_hw))

            rows.append({
                "length":            length,
                "divergence":        div,
                "t_full_ms":         round(t_full * 1000, 2),
                "t_fixed30_ms":      round(t_fixed30 * 1000, 2),
                "t_fixed100_ms":     round(t_fixed100 * 1000, 2),
                "t_neural_ms":       round(t_neural * 1000, 2),
                "speedup_vs_full":   round(t_full / t_neural, 1),
                "speedup_vs_fixed30":round(t_fixed30 / t_neural, 2),
                "pred_hw":           pred_hw,
            })
            print(f"  len={length:5d}, div={div:.0%}: "
                  f"full={t_full*1000:.1f}ms, neural={t_neural*1000:.1f}ms, "
                  f"speedup={t_full/t_neural:.1f}x")

    df = pd.DataFrame(rows)
    df.to_csv("results/experiments/pairwise_speedup.csv", index=False)

    return {
        "mean_speedup_vs_full":    float(df["speedup_vs_full"].mean()),
        "max_speedup_vs_full":     float(df["speedup_vs_full"].max()),
        "mean_speedup_vs_fixed30": float(df["speedup_vs_fixed30"].mean()),
        "results_csv": "results/experiments/pairwise_speedup.csv",
        "summary": df.groupby("length")["speedup_vs_full"].mean().round(1).to_dict(),
    }


def exp_ablation_study(predictor) -> dict:
    """
    Эксперимент 3: ablation — вклад каждого компонента.

    Методы (добавляем по одному):
    1. Full NW (baseline)
    2. Fixed band W=50
    3. Fixed band W=50 + SIMD (автоматически внутри aligner)
    4. Fixed band W=50 + Four Russians (автоматически)
    5. Neural band (наш метод)

    Метрики: время, n_doublings, band_recall.
    """
    import aligner
    import time
    import numpy as np

    DNA = "ACGT"
    rng = np.random.default_rng(0)
    # Генерировать 50 пар длиной 1000, div=15%
    pairs = []
    for i in range(50):
        seq1 = "".join(rng.choice(list(DNA), 1000))
        seq2 = list(seq1)
        for p in rng.choice(1000, 150, replace=False):
            seq2[p] = rng.choice([c for c in DNA if c != seq2[p]])
        pairs.append((seq1, "".join(seq2)))

    def timeit_all(fn):
        times = [time.perf_counter()]
        for s1, s2 in pairs:
            fn(s1, s2)
            times.append(time.perf_counter())
        return float(np.mean(np.diff(times))) * 1000  # ms

    t_full     = timeit_all(lambda s1,s2: aligner.full_nw_align(s1, s2))
    t_fixed50  = timeit_all(lambda s1,s2: aligner.align_with_doubling(s1, s2, 0, 50))
    t_fixed100 = timeit_all(lambda s1,s2: aligner.align_with_doubling(s1, s2, 0, 100))

    neural_times, neural_doublings = [], []
    for s1, s2 in pairs:
        c, hw = predictor.predict_single(s1, s2, "dna")
        t0 = time.perf_counter()
        r = aligner.align_with_doubling(s1, s2, c, hw)
        neural_times.append((time.perf_counter() - t0) * 1000)
        neural_doublings.append(r.n_doublings)
    t_neural = float(np.mean(neural_times))

    result = {
        "configs": [
            {"name": "Full NW",      "time_ms": round(t_full, 2),    "speedup": 1.0},
            {"name": "Fixed W=50",   "time_ms": round(t_fixed50, 2), "speedup": round(t_full/t_fixed50, 1)},
            {"name": "Fixed W=100",  "time_ms": round(t_fixed100, 2),"speedup": round(t_full/t_fixed100, 1)},
            {"name": "Neural band",  "time_ms": round(t_neural, 2),  "speedup": round(t_full/t_neural, 1),
             "mean_doublings": round(float(np.mean(neural_doublings)), 2)},
        ],
        "neural_mean_doublings": round(float(np.mean(neural_doublings)), 3),
        "neural_zero_doubling_rate": round(float(np.mean([d==0 for d in neural_doublings])), 3),
    }

    for c in result["configs"]:
        print(f"  {c['name']:15s}: {c['time_ms']:7.2f}ms  speedup={c['speedup']:.1f}x")

    return result


def exp_fr_hit_ratio() -> dict:
    """
    Эксперимент 4: накопление Four Russians lookup table.
    Показывает как растёт hit_ratio с каждой новой парой.
    """
    import aligner
    import numpy as np

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
    df.to_csv("results/experiments/fr_hit_ratio.csv", index=False)

    return {
        "initial_hit_ratio": history[0]["hit_ratio"],
        "final_hit_ratio":   history[-1]["hit_ratio"],
        "final_table_kb":    history[-1]["table_kb"],
        "history_csv": "results/experiments/fr_hit_ratio.csv",
    }


def exp_msa_quality(predictor, balibase_test: list) -> dict:
    """
    Эксперимент 5: качество MSA на BAliBASE (финальная таблица).

    Методы:
    1. ClustalW
    2. MAFFT
    3. MUSCLE
    4. Fixed band W=30 (ablation)
    5. Fixed band W=100 (ablation)
    6. Neural band (наш метод)
    7. Neural band + iterative refinement

    Метрики: SP-score, TC-score, время, память.
    """
    import subprocess
    import tracemalloc
    from msa.progressive_msa import progressive_msa
    from msa.iterative_refine import iterative_refine
    from baselines.classical import run_mafft, run_muscle, run_clustalw
    from scoring.metrics import sp_score, tc_score

    def measure(fn, seqs, ids, ref):
        """Замерить SP, TC, время и память."""
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            msa_result = fn(seqs, ids)
            elapsed = time.perf_counter() - t0
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            sp = sp_score(msa_result, ref)
            tc = tc_score(msa_result, ref)
            return {"sp": sp, "tc": tc, "time_s": elapsed, "mem_mb": peak_mem / 1e6, "ok": True}
        except Exception as e:
            tracemalloc.stop()
            return {"sp": 0, "tc": 0, "time_s": 999, "mem_mb": 0, "ok": False, "error": str(e)}

    # Ограничить до 30 групп для скорости
    groups = balibase_test[:30]

    def fixed_msa(seqs, ids, hw=30):
        """MSA с фиксированным band — для ablation."""
        from msa.guide_tree import pairwise_distance_matrix, build_guide_tree
        import aligner
        # Простой прогрессивный MSA с фиксированным band
        # (использует align_with_doubling с centre=0, hw=hw для всех пар)
        return progressive_msa(seqs, ids, predictor=None, fixed_hw=hw)

    methods = {
        "ClustalW":        lambda s, ids: run_clustalw(s, ids),
        "MAFFT":           lambda s, ids: run_mafft(s, ids),
        "MUSCLE":          lambda s, ids: run_muscle(s, ids),
        "Fixed_W30":       lambda s, ids: fixed_msa(s, ids, hw=30),
        "Fixed_W100":      lambda s, ids: fixed_msa(s, ids, hw=100),
        "Neural_band":     lambda s, ids: progressive_msa(s, ids, predictor),
        "Neural_+_refine": lambda s, ids: iterative_refine(
            progressive_msa(s, ids, predictor), s, predictor),
    }

    all_rows = []
    for method_name, method_fn in methods.items():
        print(f"\n  Running {method_name}...")
        for g in groups:
            r = measure(method_fn, g["sequences"], g["seq_ids"], g["reference"])
            r["method"] = method_name
            r["group_id"] = g["group_id"]
            r["ref_class"] = g["ref_class"]
            r["n_seqs"] = len(g["sequences"])
            all_rows.append(r)
            if r["ok"]:
                print(f"    {g['group_id']}: SP={r['sp']:.3f}, TC={r['tc']:.3f}, "
                      f"t={r['time_s']:.2f}s")

    df = pd.DataFrame(all_rows)
    df.to_csv("results/experiments/msa_quality_detail.csv", index=False)

    # Сводная таблица
    summary = df[df["ok"]].groupby("method").agg(
        SP_mean=("sp", "mean"),
        TC_mean=("tc", "mean"),
        Time_mean=("time_s", "mean"),
        Mem_MB_mean=("mem_mb", "mean"),
    ).round(4)
    summary.to_csv("results/experiments/msa_quality_summary.csv")
    print("\n  Summary:")
    print(summary.to_string())

    return {
        "summary": summary.to_dict(),
        "n_groups": len(groups),
        "detail_csv": "results/experiments/msa_quality_detail.csv",
        "summary_csv": "results/experiments/msa_quality_summary.csv",
    }


def exp_scaling_by_n(predictor) -> dict:
    """
    Эксперимент 6: масштабирование по числу последовательностей N.
    Сравниваем MAFFT vs Neural band при N = 10, 20, 50, 100.
    """
    import numpy as np
    from msa.progressive_msa import progressive_msa
    from baselines.classical import run_mafft

    DNA = "ACGT"
    rng = np.random.default_rng(0)

    def gen_seqs(n, length=300, div=0.15):
        base = "".join(rng.choice(list(DNA), length))
        seqs = []
        for _ in range(n):
            s = list(base)
            for p in rng.choice(length, int(length*div), replace=False):
                s[p] = rng.choice([c for c in DNA if c != s[p]])
            seqs.append("".join(s))
        return seqs

    rows = []
    for n in [10, 20, 50, 100]:
        seqs = gen_seqs(n)
        ids  = [f"seq{i}" for i in range(n)]

        t0 = time.perf_counter()
        try:
            run_mafft(seqs, ids)
            t_mafft = time.perf_counter() - t0
        except:
            t_mafft = None

        t0 = time.perf_counter()
        progressive_msa(seqs, ids, predictor)
        t_neural = time.perf_counter() - t0

        rows.append({
            "n_seqs":    n,
            "t_mafft_s":  round(t_mafft, 2) if t_mafft else None,
            "t_neural_s": round(t_neural, 2),
            "speedup":    round(t_mafft / t_neural, 2) if t_mafft else None,
        })
        print(f"  N={n:4d}: MAFFT={t_mafft:.2f}s, Neural={t_neural:.2f}s")

    df = pd.DataFrame(rows)
    df.to_csv("results/experiments/scaling_by_n.csv", index=False)

    return {
        "rows": rows,
        "csv": "results/experiments/scaling_by_n.csv",
    }


def main():
    parser = argparse.ArgumentParser(description="Run all MSA experiments")
    parser.add_argument("--checkpoint",    default="checkpoints/best_model.pt")
    parser.add_argument("--balibase_test", default="data/balibase_test.json")
    parser.add_argument("--results_dir",   default="results/experiments")
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--skip",          nargs="*", default=[],
                        help="Experiments to skip, e.g. --skip msa_quality scaling")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Загрузить модель
    print(f"\nLoading model from {args.checkpoint}...")
    try:
        from model.evaluate import BandPredictorInference
        predictor = BandPredictorInference(args.checkpoint, device=args.device)
        print("Model loaded OK")
        model_available = True
    except Exception as e:
        print(f"WARNING: Could not load model: {e}")
        print("Experiments requiring neural network will be skipped")
        predictor = None
        model_available = False

    # Загрузить BAliBASE тест
    balibase_test = []
    if os.path.exists(args.balibase_test):
        with open(args.balibase_test) as f:
            balibase_test = json.load(f)
        print(f"BAliBASE test: {len(balibase_test)} groups")
        balibase_available = True
    else:
        print("WARNING: BAliBASE test data not found")
        balibase_available = False

    # Запустить все эксперименты
    all_results = {}

    # Эксперимент 1: точность band prediction (нужна модель)
    if "band_accuracy" not in args.skip and model_available:
        all_results["band_prediction_accuracy"] = run_experiment(
            "band_prediction_accuracy",
            lambda: exp_band_prediction_accuracy(predictor, []),
            args.results_dir
        )

    # Эксперимент 2: speedup попарного выравнивания
    if "speedup" not in args.skip:
        all_results["pairwise_speedup"] = run_experiment(
            "pairwise_speedup",
            lambda: exp_pairwise_speedup(predictor) if model_available
                    else exp_pairwise_speedup_no_neural(),
            args.results_dir
        )

    # Эксперимент 3: ablation
    if "ablation" not in args.skip and model_available:
        all_results["ablation"] = run_experiment(
            "ablation_study",
            lambda: exp_ablation_study(predictor),
            args.results_dir
        )

    # Эксперимент 4: Four Russians hit ratio
    if "fr" not in args.skip:
        all_results["fr_hit_ratio"] = run_experiment(
            "fr_hit_ratio",
            lambda: exp_fr_hit_ratio(),
            args.results_dir
        )

    # Эксперимент 5: качество MSA на BAliBASE
    if "msa_quality" not in args.skip and balibase_available and model_available:
        all_results["msa_quality"] = run_experiment(
            "msa_quality",
            lambda: exp_msa_quality(predictor, balibase_test),
            args.results_dir
        )

    # Эксперимент 6: масштабирование по N
    if "scaling" not in args.skip and model_available:
        all_results["scaling"] = run_experiment(
            "scaling_by_n",
            lambda: exp_scaling_by_n(predictor),
            args.results_dir
        )

    # Сохранить сводный JSON
    summary_path = os.path.join(args.results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS DONE")
    print(f"Summary: {summary_path}")
    print(f"{'='*60}")

    # Статус по каждому
    for name, result in all_results.items():
        status = result.get("_status", "?")
        elapsed = result.get("_elapsed_s", 0)
        emoji = "✓" if status == "OK" else "✗"
        print(f"  {emoji} {name:35s} {status:6s} {elapsed:.1f}s")


if __name__ == "__main__":
    main()
```

---

## Файл 5: `results/interpret_results.py`

```python
#!/usr/bin/env python3
"""
Генерирует итоговый отчёт FINAL_REPORT.md из результатов экспериментов.
Запускать после experiments/run_all.py:
    python results/interpret_results.py \
        --results_dir results \
        --output results/FINAL_REPORT.md

Этот файл отправляется Клоду для анализа и интерпретации.
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_json(path: str) -> dict:
    """Загрузить JSON или вернуть пустой dict."""
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return {}


def fmt(v, decimals=3):
    """Форматировать число."""
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{decimals}f}"
    except:
        return str(v)


def generate_report(results_dir: str, output: str):
    """Сгенерировать FINAL_REPORT.md."""
    exp_dir = os.path.join(results_dir, "experiments")
    train_dir = os.path.join(results_dir, "training")

    lines = []
    lines.append("# MSA Neural Band Prediction — Final Results Report")
    lines.append(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> **Отправить этот файл Клоду для интерпретации результатов.**")
    lines.append("")

    # ---------- Статус экспериментов ----------
    lines.append("## Статус экспериментов")
    summary = load_json(os.path.join(exp_dir, "summary.json"))
    if summary:
        lines.append("")
        lines.append("| Эксперимент | Статус | Время (с) |")
        lines.append("|---|---|---|")
        for name, res in summary.items():
            status = res.get("_status", "?")
            elapsed = res.get("_elapsed_s", 0)
            emoji = "✅" if status == "OK" else "❌"
            lines.append(f"| {name} | {emoji} {status} | {elapsed:.1f} |")
    lines.append("")

    # ---------- Обучение ----------
    lines.append("## 1. Результаты обучения нейросети")
    training_log = os.path.join(train_dir, "training_history.json")
    if os.path.exists(training_log):
        history = load_json(training_log)
        if history:
            lines.append("")
            lines.append("### Финальные метрики (лучшая эпоха)")
            best = history.get("best_epoch_metrics", {})
            lines.append(f"- **band_recall@1x:** {fmt(best.get('band_recall@1.0x'))}")
            lines.append(f"- **band_recall@1.5x:** {fmt(best.get('band_recall@1.5x'))}")
            lines.append(f"- **band_recall@2x:** {fmt(best.get('band_recall@2.0x'))}")
            lines.append(f"- **MAE centre_diag:** {fmt(best.get('mae_centre'))}")
            lines.append(f"- **width_ratio:** {fmt(best.get('width_ratio'))}")
            lines.append(f"- **Best epoch:** {best.get('epoch', 'N/A')}")
            lines.append("")

            # Таблица по эпохам
            epochs = history.get("epochs", [])
            if epochs:
                lines.append("### История обучения (все эпохи)")
                lines.append("")
                lines.append("| Epoch | Loss | Recall@1x | Recall@1.5x | MAE_centre | Width_ratio |")
                lines.append("|---|---|---|---|---|---|")
                for e in epochs:
                    lines.append(
                        f"| {e.get('epoch','')} "
                        f"| {fmt(e.get('loss'),4)} "
                        f"| {fmt(e.get('band_recall@1.0x'))} "
                        f"| {fmt(e.get('band_recall@1.5x'))} "
                        f"| {fmt(e.get('mae_centre'))} "
                        f"| {fmt(e.get('width_ratio'))} |"
                    )
    lines.append("")

    # ---------- Эксперимент 1: точность band ----------
    lines.append("## 2. Точность предсказания band нейросетью")
    band_acc = load_json(os.path.join(exp_dir, "band_prediction_accuracy.json"))
    if band_acc and band_acc.get("_status") == "OK":
        lines.append("")
        lines.append(f"- **Общий recall@1x:** {fmt(band_acc.get('overall_recall_1x'))} "
                     f"(без doubling)")
        lines.append(f"- **Общий recall@1.5x:** {fmt(band_acc.get('overall_recall_1_5x'))}")
        lines.append(f"- **Общий recall@2x:** {fmt(band_acc.get('overall_recall_2x'))}")
        lines.append(f"- **MAE centre_diag:** {fmt(band_acc.get('overall_mae_centre'))}")
        lines.append(f"- **Width ratio:** {fmt(band_acc.get('overall_width_ratio'))} "
                     f"(>1 = переоцениваем, безопасно)")
        lines.append("")
        lines.append("### По группам дивергенции")
        lines.append("")
        lines.append("| Группа | Recall@1x | Recall@1.5x | Recall@2x | MAE centre |")
        lines.append("|---|---|---|---|---|")
        by_group = band_acc.get("by_divergence_group", {})
        for group in ["low", "medium", "high"]:
            r1  = fmt(by_group.get("recall_1x", {}).get(group))
            r15 = fmt(by_group.get("recall_1_5x", {}).get(group))
            r2  = fmt(by_group.get("recall_2x", {}).get(group))
            mae = fmt(by_group.get("mae_centre", {}).get(group))
            lines.append(f"| {group} | {r1} | {r15} | {r2} | {mae} |")
    lines.append("")

    # ---------- Эксперимент 2: speedup ----------
    lines.append("## 3. Speedup попарного выравнивания")
    speedup = load_json(os.path.join(exp_dir, "pairwise_speedup.json"))
    if speedup and speedup.get("_status") == "OK":
        lines.append("")
        lines.append(f"- **Средний speedup vs Full NW:** {fmt(speedup.get('mean_speedup_vs_full'), 1)}x")
        lines.append(f"- **Максимальный speedup:** {fmt(speedup.get('max_speedup_vs_full'), 1)}x")
        lines.append(f"- **Средний speedup vs Fixed W=30:** "
                     f"{fmt(speedup.get('mean_speedup_vs_fixed30'), 2)}x")
        lines.append("")
        lines.append("### Speedup по длине последовательности (vs Full NW)")
        lines.append("")
        lines.append("| Длина | Средний speedup |")
        lines.append("|---|---|")
        for length, sp in speedup.get("summary", {}).items():
            lines.append(f"| {length} bp | {sp}x |")

        # Детальная таблица из CSV
        csv_path = os.path.join(exp_dir, "pairwise_speedup.csv")
        if os.path.exists(csv_path) and HAS_PANDAS:
            df = pd.read_csv(csv_path)
            lines.append("")
            lines.append("### Детальная таблица (время в мс)")
            lines.append("")
            lines.append("| Длина | Дивергенция | Full NW | Fixed W=30 | Fixed W=100 | Neural | Speedup |")
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
    lines.append("")

    # ---------- Эксперимент 3: ablation ----------
    lines.append("## 4. Ablation Study — вклад каждого компонента")
    ablation = load_json(os.path.join(exp_dir, "ablation_study.json"))
    if ablation and ablation.get("_status") == "OK":
        lines.append("")
        lines.append("| Метод | Время (мс) | Speedup vs Full NW | Doublings |")
        lines.append("|---|---|---|---|")
        for cfg in ablation.get("configs", []):
            doublings = f"{cfg.get('mean_doublings', 0):.2f}" if "mean_doublings" in cfg else "—"
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
                     f"(нейросеть угадала с первого раза)")
    lines.append("")

    # ---------- Эксперимент 4: Four Russians ----------
    lines.append("## 5. Four Russians — накопление lookup table")
    fr = load_json(os.path.join(exp_dir, "fr_hit_ratio.json"))
    if fr and fr.get("_status") == "OK":
        lines.append("")
        lines.append(f"- **Начальный hit_ratio:** {fmt(fr.get('initial_hit_ratio'))}")
        lines.append(f"- **Финальный hit_ratio:** {fmt(fr.get('final_hit_ratio'))}")
        lines.append(f"- **Размер таблицы:** {fr.get('final_table_kb', 0)} KB")
        lines.append("")
        csv_path = os.path.join(exp_dir, "fr_hit_ratio.csv")
        if os.path.exists(csv_path) and HAS_PANDAS:
            df = pd.read_csv(csv_path)
            lines.append("| Пар обработано | Hit ratio | Таблица (KB) |")
            lines.append("|---|---|---|")
            for _, row in df.iterrows():
                lines.append(
                    f"| {int(row['n_pairs'])} "
                    f"| {row['hit_ratio']:.1%} "
                    f"| {int(row['table_kb'])} |"
                )
    lines.append("")

    # ---------- Эксперимент 5: качество MSA ----------
    lines.append("## 6. Качество MSA на BAliBASE")
    msa_q = load_json(os.path.join(exp_dir, "msa_quality.json"))
    if msa_q and msa_q.get("_status") == "OK":
        lines.append("")
        lines.append(f"- Протестировано групп: {msa_q.get('n_groups', 0)}")
        lines.append("")
        lines.append("### Сводная таблица (все методы)")
        lines.append("")
        lines.append("| Метод | SP-score | TC-score | Время (с) | Память (МБ) |")
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

        csv_path = os.path.join(exp_dir, "msa_quality_summary.csv")
        if os.path.exists(csv_path) and HAS_PANDAS:
            lines.append("")
            lines.append(f"*Детальные результаты: `{csv_path}`*")
    else:
        lines.append("")
        lines.append("*Эксперимент пропущен (нет BAliBASE или модели)*")
    lines.append("")

    # ---------- Эксперимент 6: масштабирование ----------
    lines.append("## 7. Масштабирование по числу последовательностей N")
    scaling = load_json(os.path.join(exp_dir, "scaling_by_n.json"))
    if scaling and scaling.get("_status") == "OK":
        lines.append("")
        lines.append("| N | MAFFT (с) | Neural (с) | Speedup |")
        lines.append("|---|---|---|---|")
        for row in scaling.get("rows", []):
            t_mafft = fmt(row.get("t_mafft_s"), 2) if row.get("t_mafft_s") else "N/A"
            t_neural = fmt(row.get("t_neural_s"), 2)
            speedup = fmt(row.get("speedup"), 2) if row.get("speedup") else "N/A"
            lines.append(
                f"| {row.get('n_seqs')} "
                f"| {t_mafft} "
                f"| {t_neural} "
                f"| {speedup}x |"
            )
    lines.append("")

    # ---------- Тесты ----------
    lines.append("## 8. Результаты тестов")
    test_report = load_json(os.path.join(results_dir, "tests/full_test_report.json"))
    if test_report:
        summary_t = test_report.get("summary", {})
        lines.append("")
        lines.append(f"- **Всего тестов:** {summary_t.get('total', 'N/A')}")
        lines.append(f"- **Прошли:** {summary_t.get('passed', 'N/A')}")
        lines.append(f"- **Провалились:** {summary_t.get('failed', 0)}")
        lines.append(f"- **Пропущены:** {summary_t.get('skipped', 0)}")
        lines.append(f"- **Время:** {fmt(summary_t.get('duration'), 1)}с")
    lines.append("")

    # ---------- Системная информация ----------
    lines.append("## 9. Информация о системе")
    sysinfo_path = os.path.join(results_dir, "system_info.json")
    if os.path.exists(sysinfo_path):
        sysinfo = load_json(sysinfo_path)
        lines.append("")
        for k, v in sysinfo.items():
            lines.append(f"- **{k}:** {v}")
    lines.append("")

    lines.append("---")
    lines.append("*Отправить этот файл Клоду: он интерпретирует результаты и даст выводы для диссертации.*")

    # Записать файл
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output", default="results/FINAL_REPORT.md")
    args = parser.parse_args()
    generate_report(args.results_dir, args.output)
```

---

## Файл 6: `save_system_info.py`

```python
#!/usr/bin/env python3
"""
Сохраняет информацию о системе перед запуском экспериментов.
Запускается автоматически из vastai_run.sh.
"""
import json, subprocess, sys, os

def get_system_info() -> dict:
    info = {}

    # Python
    info["Python"] = sys.version.split()[0]

    # PyTorch + CUDA
    try:
        import torch
        info["PyTorch"] = torch.__version__
        info["CUDA_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["GPU"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["VRAM_GB"] = f"{props.total_memory / 1e9:.1f}"
            info["CUDA_version"] = torch.version.cuda
    except:
        pass

    # CPU
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    info["CPU"] = line.split(":")[1].strip()
                    break
        info["CPU_cores"] = str(os.cpu_count())
        info["AVX2"] = "yes" if "avx2" in open("/proc/cpuinfo").read() else "no"
    except:
        pass

    # RAM
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    kb = int(line.split()[1])
                    info["RAM_GB"] = f"{kb / 1e6:.1f}"
                    break
    except:
        pass

    # aligner module
    try:
        import aligner
        info["aligner_module"] = "OK"
    except Exception as e:
        info["aligner_module"] = f"ERROR: {e}"

    return info

if __name__ == "__main__":
    info = get_system_info()
    os.makedirs("results", exist_ok=True)
    with open("results/system_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("System info saved to results/system_info.json")
    for k, v in info.items():
        print(f"  {k}: {v}")
```

---

## [ЗАПРОС ДЛЯ COPILOT — СКОПИРОВАТЬ ЦЕЛИКОМ]

```
Read this file (vastai_deployment_guide.md) completely before generating any code.

Generate ALL files described in this guide. In this exact order:

1. vastai_setup.sh          — server setup script
2. vastai_run.sh            — training + experiments runner
3. vastai_download.sh       — download results locally
4. experiments/run_all.py   — all experiments unified entry point
5. results/interpret_results.py — generate FINAL_REPORT.md
6. save_system_info.py      — save GPU/CPU info before run

RULES:
- Implement every function fully — no placeholders
- All results must be saved as both JSON and CSV
- Every experiment must catch exceptions and save error status
  (never crash the full pipeline due to one failed experiment)
- All output paths must be relative to project root
- Shell scripts must use `set -e` and log everything to logs/
- Python scripts must work with or without BAliBASE
  (skip gracefully if data not available)
- results/FINAL_REPORT.md must be human-readable markdown
  that can be sent to Claude for interpretation

After generating, also update vastai_run.sh to call save_system_info.py
at the very beginning (before any other steps).

Important paths:
- BAliBASE: data/raw/balibase/DATASET-BALiBASE/
  - Aligned sequences/*.xml
  - Unaligned sequences/*.tfa
- Synthetic data: data/processed/train.parquet (50k, already exists)
- Model checkpoint: checkpoints/best_model.pt
- All results: results/experiments/*.json and *.csv
- Final report: results/FINAL_REPORT.md
```
