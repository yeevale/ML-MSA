# MSA Neural Band Prediction — Copilot Generation Guide
> **Target models:** Claude Opus / Claude Sonnet via GitHub Copilot
> **Language:** Python 3.11 + C++17 (mixed codebase)
> **Purpose:** этот файл — единственный источник правды для генерации всего кода проекта.
>              Читай его целиком перед тем как генерировать любой файл.

---

## [COPILOT SYSTEM CONTEXT — READ FIRST, DO NOT SKIP]

Ты генерируешь код для магистерской диссертации по биоинформатике.
Тема: **нейросеть предсказывает диагональную полосу (band) для ускорения
множественного выравнивания последовательностей (MSA)**.

### Что такое MSA и banded NW — краткий ликбез

**Задача MSA:** выровнять N биологических последовательностей (ДНК/белки) друг с другом,
вставив гэпы `-` так чтобы гомологичные позиции совпали по столбцам.

**Needleman-Wunsch (NW):** классический алгоритм глобального попарного выравнивания.
Заполняет матрицу DP размером (len1+1) × (len2+1). Сложность O(n²) по времени и памяти.

**Banded NW:** вместо полной матрицы вычисляет только диагональную полосу шириной 2W+1
вокруг предсказанного центра. Сложность O(n·W). Гарантия: если оптимальный путь попал
в band → результат точный. Если нет → band doubling (удваиваем и пересчитываем).

**Наша идея (научная новизна):** нейросеть смотрит на пару последовательностей
и предсказывает КУДА пойдёт оптимальный путь (centre_diag) и НАСКОЛЬКО ШИРОКО
он отклонится (half_width). Это и есть band. Без нашего предсказания band
либо фиксированный (часто промахивается), либо не используется вовсе.

**Ключевой инвариант:** band doubling ВСЕГДА гарантирует правильный ответ.
Нейросеть влияет только на скорость, но не на корректность.

### Типы данных и их представления

```
# Последовательность ДНК: строка из "ACGTN"
seq: str = "ACGTACGT..."

# Последовательность белка: строка из 20 аминокислот
seq: str = "MKTAYIAKQRQISFVKSHFSRQ..."

# Профиль MSA: матрица частот символов
# shape = (alignment_length, alphabet_size)
# alphabet_size = 5 для ДНК (A,C,G,T,-) или 21 для белков (20 aa + -)
# profile[i, a] = доля символа a в позиции i, сумма по строке = 1.0
profile: np.ndarray  # shape (L, A), dtype=float32

# Band параметры
centre_diag: int   # смещение пути от главной диагонали (i-j для точки (i,j))
half_width: int    # максимальное отклонение пути от centre_diag
# Band содержит ячейки (i,j) где abs((i-j) - centre_diag) <= half_width

# Результат выравнивания из C++
BandedResult.score: float
BandedResult.aligned_seq1: str   # с гэпами '-'
BandedResult.aligned_seq2: str   # с гэпами '-'
BandedResult.path_escaped: bool  # True → нужен band doubling
BandedResult.escape_left: bool   # путь ушёл влево за band
BandedResult.escape_right: bool  # путь ушёл вправо за band
BandedResult.max_deviation: int  # максимальное отклонение по traceback
```

### Архитектура системы (весь пайплайн)

```
ВХОД: N последовательностей (str или .fasta файл)
  │
  ├─[если max(len) > 5000]─► ЯКОРНЫЙ РЕЖИМ (anchors.py)
  │                            минимайзеры → LIS-цепочка → блоки
  │
  ▼
1. GUIDE TREE (guide_tree.py)
   k-mer Jaccard дистанции → NJ/UPGMA дерево
   Параллельно: joblib.Parallel(n_jobs=-1)
   Возвращает: TreeNode (бинарное дерево, листья = индексы последовательностей)
  │
  ▼
2. ПРОГРЕССИВНОЕ MSA (progressive_msa.py)
   Post-order обход дерева (снизу вверх), N-1 шагов
   На каждом шаге:
     a. make_input(obj1, obj2)  → (matrix 1×64×64, scalars ~70)
        obj = str (лист) или np.ndarray профиль (внутренний узел)
     b. BandPredictorInference.predict_batch(pairs)  [GPU]
        → (centre_diag, half_width)
     c. aligner.align_with_doubling(seq1, seq2, centre, hw)  [C++]
        ВНУТРИ: Hirschberg + Four Russians + SIMD AVX2 + асимм. doubling
     d. build_profile(aligned_seqs)  → новый профиль
     e. del child_profiles; gc.collect()  [ленивая память]
  │
  ▼
3. ИТЕРАТИВНОЕ УТОЧНЕНИЕ (iterative_refine.py)
   3 прохода, MUSCLE-style
   Каждый раз: убрать seq_i → profile остатка → выровнять seq_i vs profile
   Принять если SP-score улучшился
  │
  ▼
ВЫХОД: list[str] — N выровненных строк одинаковой длины с гэпами
```

### Полная карта файлов и их зависимостей

```
msa_band_neural/
│
├── data/
│   ├── simulate.py       # генератор синтетики → .parquet
│   │   └── uses: full_nw (C++)
│   └── loaders.py        # FASTA file loaders
│
├── features/
│   ├── kmer.py           # kmer_features(seq1, seq2) → np.ndarray(70,)
│   ├── dotplot.py        # dotplot_tensor(seq1, seq2) → np.ndarray(1,64,64)
│   ├── profile_features.py  # make_input(obj1, obj2) → унифицированный интерфейс
│   │   └── uses: kmer.py, dotplot.py
│   └── anchors.py        # find_anchors, chain_anchors, split_by_anchors
│
├── model/
│   ├── band_predictor.py # BandPredictor(nn.Module) + loss functions
│   ├── train.py          # BandDataset + тренировочный цикл
│   └── evaluate.py       # BandPredictorInference (батчевый GPU инференс)
│       └── uses: band_predictor.py, profile_features.py
│
├── aligner/              # C++17, компилируется в Python модуль 'aligner'
│   ├── full_nw.cpp       # полный NW + traceback (эталон, только для верификации)
│   ├── banded_nw.cpp     # базовый banded NW + BandedResult struct
│   ├── simd_banded_nw.cpp # AVX2 антидиагональная версия
│   ├── hirschberg.cpp    # divide-and-conquer, O(W) память, использует FR внутри
│   ├── four_russians.cpp # FourRussiansAligner класс, t×t lookup table + SIMD
│   ├── band_doubling.cpp # align_with_doubling + асимметричное расширение + pybind11
│   ├── profile_dp.cpp    # profile-profile версии всех функций
│   └── anchored_align.cpp # выравнивание блоков для длинных последовательностей
│
├── msa/
│   ├── guide_tree.py     # TreeNode, pairwise_distance_matrix, build_guide_tree
│   ├── progressive_msa.py # progressive_msa() — главный пайплайн
│   │   └── uses: guide_tree.py, profile_features.py, evaluate.py, aligner
│   ├── profile_align.py  # вспомогательные функции profile-profile DP
│   └── iterative_refine.py # iterative_refine()
│
├── baselines/
│   └── classical.py      # run_mafft, run_muscle, run_clustalw (subprocess)
│
├── scoring/
│   ├── metrics.py        # sp_score, tc_score, sp_score_internal, benchmark
│   └── band_metrics.py   # band_recall, width_efficiency, mean_doublings
│
├── experiments/
│   ├── run_pairwise.py   # бенчмарк попарного выравнивания
│   ├── run_msa.py        # бенчмарк полного MSA
│   ├── ablation.py       # neural vs fixed band (W=30, W=100)
│   └── compare.py        # финальная сводная таблица
│
├── CMakeLists.txt
└── requirements.txt
```

### Глобальные константы и типы (используй везде)

```python
# Алфавиты
DNA_ALPHABET     = "ACGT"
DNA_GAP_ALPHABET = "ACGT-"   # для профилей
PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
PROTEIN_GAP_ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"

# Размеры алфавитов
DNA_ALPHA_SIZE     = 4   # без гэпа (для последовательностей)
DNA_PROF_SIZE      = 5   # с гэпом (для профилей)
PROTEIN_ALPHA_SIZE = 20
PROTEIN_PROF_SIZE  = 21

# Нейросеть
SCALAR_DIM   = 70    # размер скалярного вектора признаков
MATRIX_SIZE  = 64    # размер dot-plot тензора (64×64)
CNN_OUT_DIM  = 256   # выход CNN ветки
MLP_OUT_DIM  = 64    # выход MLP ветки
EMB_DIM      = 8     # seq_type embedding
TOTAL_DIM    = CNN_OUT_DIM + MLP_OUT_DIM + EMB_DIM  # = 328

# Обучение
SEQ_TYPE_DNA     = 0
SEQ_TYPE_PROTEIN = 1

# C++ выравнивание
DEFAULT_GAP_OPEN   = -10.0
DEFAULT_GAP_EXTEND = -0.5
MARGIN             = 3     # запас при вычислении true_half_width
MAX_DIRECT_LEN     = 5000  # порог для якорного режима
FR_MIN_HALF_WIDTH  = 16    # минимум для Four Russians
HIRSCHBERG_MEMORY_THRESHOLD = 200 * 1024 * 1024  # 200 МБ

# Дивергенция
DIV_LOW_MAX    = 0.10
DIV_MEDIUM_MAX = 0.25
DIV_HIGH_MAX   = 0.50
```

### Как работают C++ модули из Python

```python
import aligner  # единственный C++ модуль, содержит всё

# Попарное выравнивание (автоматически выбирает Hirschberg + FR + SIMD)
result: aligner.DoublingResult = aligner.align_with_doubling(
    seq1, seq2,
    pred_centre=centre_diag,
    pred_hw=half_width,
    gap_open=DEFAULT_GAP_OPEN,
    gap_extend=DEFAULT_GAP_EXTEND,
    is_protein=False
)
print(result.alignment.aligned_seq1)  # str с гэпами
print(result.n_doublings)             # 0 = нейросеть угадала точно

# Profile-profile выравнивание
result = aligner.align_profiles_with_doubling(
    profile1,   # np.ndarray (L1, A)
    profile2,   # np.ndarray (L2, A)
    subst,      # np.ndarray (A, A)
    pred_centre, pred_hw
)

# FourRussiansAligner (для прямого использования)
fr = aligner.FourRussiansAligner(block_size=0, is_protein=False,
                                   gap_open=-10.0, gap_extend=-0.5,
                                   quant_levels=16)
fr.last_row(seq1, seq2, centre_diag, half_width)  # для Hirschberg
stats = fr.get_stats()  # stats.hit_ratio — должно быть > 0.9
```

### Ключевые паттерны которые используются везде

```python
# Паттерн 1: make_input — единый вход для нейросети
from features.profile_features import make_input
matrix, scalars = make_input(obj1, obj2, seq_type="dna")
# obj = str (последовательность) или np.ndarray профиль

# Паттерн 2: батчевый инференс нейросети
predictor = BandPredictorInference(checkpoint_path, device="cuda")
predictions: list[tuple[int,int]] = predictor.predict_batch(
    [(obj1_a, obj2_a), (obj1_b, obj2_b)],  # весь уровень дерева сразу
    seq_type="dna"
)

# Паттерн 3: ленивые профили
node_objects: dict[int, str | np.ndarray] = {}  # seq_idx → str или профиль
# После объединения:
node_objects[parent_idx] = new_profile
del node_objects[left_idx], node_objects[right_idx]
import gc; gc.collect()

# Паттерн 4: build_profile
profile = build_profile(aligned_seqs, seq_type="dna")
# aligned_seqs: list[str] — строки одинаковой длины с гэпами
# Возвращает np.ndarray (alignment_length, alphabet_size), dtype=float32
```

---

## Раздел 1: Генерация данных

### `data/simulate.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# Этот модуль генерирует обучающие данные для нейросети.
# Нейросеть учится предсказывать band по паре последовательностей.
# Для обучения нужно знать ТОЧНЫЙ оптимальный путь NW → вычисляем через full_nw.
# Синтетика нужна потому что нужно знать точный путь выравнивания.
#
# ВЫХОД: .parquet файлы с колонками:
#   seq1 (str), seq2 (str), centre_diag (int), true_half_width (int),
#   divergence (float), seq_type (str)
#
# ВАЖНО: стратифицировать по уровню дивергенции равномерно (33% на каждую группу):
#   low    = divergence ∈ [0.01, 0.10]
#   medium = divergence ∈ [0.10, 0.25]
#   high   = divergence ∈ [0.25, 0.50]
#
# ПАРАЛЛЕЛИЗАЦИЯ: multiprocessing.Pool, каждый worker независим через seed

import numpy as np
from dataclasses import dataclass, asdict
from multiprocessing import Pool
import pyarrow as pa
import pyarrow.parquet as pq
import sys
sys.path.append("../aligner")
import full_nw  # C++ pybind11: full_nw.align(seq1, seq2) → path: list[(i,j)]

PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
DNA_ALPHABET     = "ACGT"
MARGIN           = 3

@dataclass
class AlignmentSample:
    seq1:            str
    seq2:            str
    centre_diag:     int    # median(i-j) по traceback пути
    true_half_width: int    # max(abs(d - centre)) + MARGIN
    divergence:      float  # edit_distance / max(len1, len2)
    seq_type:        str    # "dna" или "protein"

def mutate_sequence(seq: str, p_sub: float, p_ins: float, p_del: float,
                    alphabet: str, rng: np.random.Generator) -> str:
    """Мутировать seq: deletion → substitution (посимвольно), затем insertion pass.
    Использовать rng для воспроизводимости. Возвращать новую строку."""
    ...

def compute_band_params(path: list[tuple[int, int]]) -> tuple[int, int]:
    """По traceback пути вычислить (centre_diag, true_half_width).
    diagonals = [i - j for i, j in path]
    centre_diag = int(np.median(diagonals))
    true_half_width = max(abs(d - centre_diag) for d in diagonals) + MARGIN"""
    ...

def simulate_one(args: tuple) -> dict | None:
    """Сгенерировать одну пару для multiprocessing.
    args = (length: int, p_sub: float, p_ins: float, p_del: float,
            seq_type: str, seed: int)
    Возвращает dict(asdict(AlignmentSample)) или None при ошибке."""
    ...

def sample_mutation_params(divergence_group: str,
                            rng: np.random.Generator) -> tuple[float, float, float]:
    """Сэмплировать (p_sub, p_ins, p_del) для заданной группы дивергенции.
    low:    p_sub ∈ [0.01, 0.08], p_ins ∈ [0.0, 0.02], p_del ∈ [0.0, 0.02]
    medium: p_sub ∈ [0.08, 0.20], p_ins ∈ [0.01, 0.05], p_del ∈ [0.01, 0.05]
    high:   p_sub ∈ [0.20, 0.40], p_ins ∈ [0.03, 0.10], p_del ∈ [0.03, 0.10]"""
    ...

def generate_dataset(n_samples: int, output_path: str,
                     seq_type: str = "dna", n_workers: int = 8, seed: int = 42):
    """Сгенерировать n_samples пар и сохранить в parquet.
    По n_samples // 3 на каждую группу дивергенции.
    Длины: Uniform(50, 2000) для ДНК, Uniform(30, 500) для белков.
    Записывать батчами по 10_000 строк (экономия RAM).
    Показывать прогресс через tqdm."""
    ...
```

### `data/loaders.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# FASTA file loaders — вспомогательные функции для чтения FASTA файлов.

from pathlib import Path

def load_fasta(path: str) -> list[tuple[str, str]]:
    """Парсить .fasta/.tfa.
    Возвращает list[(header: str, sequence_without_gaps: str)]."""
    ...

def load_fasta_with_gaps(path: str) -> list[tuple[str, str]]:
    """Парсить .fasta/.tfa, сохраняя гэпы.
    Возвращает list[(header: str, sequence_with_gaps: str)]."""
    ...
```

---

## Раздел 2: Признаки для нейросети

### `features/kmer.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# Скалярные признаки для нейросети (режим 1: две последовательности).
# НЕ требуют выравнивания — вычисляются быстро, дают глобальную оценку дивергенции.
# Нейросеть использует их для предсказания ШИРИНЫ band (half_width).
# Все признаки нормализовать в [0, 1] или z-score.

import numpy as np
from collections import Counter
from typing import Literal

SeqType = Literal["dna", "protein"]

def kmer_freq(seq: str, k: int) -> np.ndarray:
    """Вектор нормализованных частот всех k-меров.
    Для ДНК k=4: вектор длиной 256. Для белков k=3: 8000.
    k-меры содержащие 'N' или 'X' игнорировать.
    Нормализовать: делить на sum (→ частоты)."""
    ...

def minimizers(seq: str, w: int, k: int) -> set[str]:
    """Стандартный алгоритм минимайзеров (Roberts et al. 2004).
    Для каждого окна размером w: выбрать лексикографически минимальный k-мер.
    Возвращает set уникальных минимайзеров."""
    ...

def kmer_features(seq1: str, seq2: str, seq_type: SeqType = "dna") -> np.ndarray:
    """Собрать ~70 скалярных признаков для пары последовательностей.

    Список признаков:
      # Базовые (4 штуки)
      len1, len2, len1/len2, abs(len1-len2)/max(len1,len2)

      # GC-контент для ДНК или состав аминокислот для белков (2 штуки)
      GC_content(seq1), GC_content(seq2)  # для ДНК
      charged_fraction(seq1), charged_fraction(seq2)  # для белков (DEKRH)

      # K-mer статистики для k=3 И k=4 (ДНК) или k=2 И k=3 (белки):
      # по 3 признака на каждое k = 6 штук
      Jaccard(kmer_set1, kmer_set2)       # |A∩B|/|A∪B|
      cosine(kmer_freq1, kmer_freq2)      # dot(f1,f2)/(norm(f1)*norm(f2))
      l1_dist(kmer_freq1, kmer_freq2)     # sum(|f1-f2|) / 2 (нормировано)

      # Минимайзеры w=5, k=8 (3 штуки)
      shared_minimizer_fraction           # |min1∩min2|/|min1∪min2|
      len(minimizers(seq1))/len(seq1)     # плотность минимайзеров
      len(minimizers(seq2))/len(seq2)

      # Энтропия k-mer распределений (2 штуки)
      entropy(kmer_freq(seq1, k=4))       # Shannon entropy
      entropy(kmer_freq(seq2, k=4))

    Итого: 4 + 2 + 6 + 3 + 2 = 17 базовых.
    Дополнить нулями до SCALAR_DIM=70 если нужно (для совместимости с профилями).
    dtype=float32."""
    ...
```

### `features/dotplot.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# Dot-plot тензор — визуальное представление где совпадают k-меры двух последовательностей.
# "След" совпадений идёт вдоль диагонали → CNN видит куда пойдёт путь выравнивания.
# Смещение следа от главной диагонали → centre_diag.
# Ширина следа → half_width.
# Форма выхода ВСЕГДА (1, 64, 64) float32 — независимо от длин последовательностей.

import numpy as np
from scipy.ndimage import zoom

def dotplot_tensor(seq1: str, seq2: str,
                   target_size: int = 64, k: int = 4) -> np.ndarray:
    """Построить сжатый dot-plot тензор.

    Алгоритм:
    1. Хеш-таблица k-меров seq2: kmer_to_pos2 = defaultdict(list)
       для j in range(len(seq2)-k+1): kmer_to_pos2[seq2[j:j+k]].append(j)
    2. Бинарная матрица dot: shape (len1-k+1, len2-k+1), dtype=float32
       для i in range(len1-k+1):
           kmer = seq1[i:i+k]
           для j in kmer_to_pos2.get(kmer, []):
               dot[i, j] = 1.0
    3. Сжать до (target_size, target_size):
       zoom_factors = (target_size / dot.shape[0], target_size / dot.shape[1])
       dot_small = zoom(dot, zoom_factors, order=1)  # билинейная интерполяция
    4. Нормализовать в [0, 1] через min-max (если max > 0)
    5. Вернуть shape (1, target_size, target_size) dtype=float32

    k: 4 для ДНК, 3 для белков (меньше случайных совпадений при малом алфавите).
    Если len(seq) < k: вернуть нулевой тензор shape (1, 64, 64)."""
    ...
```

### `features/profile_features.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# Этот файл — ЦЕНТРАЛЬНЫЙ для нейросети.
# Он реализует make_input() — унифицированный интерфейс,
# который работает для ОБОИХ режимов:
#   Режим 1: obj = str (листья guide tree)
#   Режим 2: obj = np.ndarray профиль (внутренние узлы)
# Нейросеть ОДНА — архитектура не меняется.
# Выходные тензоры всегда одной формы: (1,64,64) + вектор (SCALAR_DIM,).
#
# ПРОФИЛЬ: np.ndarray shape (L, A) где:
#   L = длина выравнивания (число колонок)
#   A = DNA_PROF_SIZE=5 или PROTEIN_PROF_SIZE=21
#   profile[i, a] = частота символа a в позиции i, sum(profile[i]) = 1.0

import numpy as np
from scipy.ndimage import zoom
from features.kmer import kmer_features, dotplot_tensor

# Загрузить BLOSUM62 из файла при импорте
# Файл blosum62.txt: строки "#comment" или "A  C  D ..." (20×20 матрица)
BLOSUM62: np.ndarray | None = None  # shape (20, 20), dtype=float32
_BLOSUM62_PATH = "data/blosum62.txt"

DNA_SUBST = np.array([
    [ 1., -1., -1., -1.],  # A
    [-1.,  1., -1., -1.],  # C
    [-1., -1.,  1., -1.],  # G
    [-1., -1., -1.,  1.],  # T
], dtype=np.float32)

def load_blosum62(path: str = _BLOSUM62_PATH) -> np.ndarray:
    """Загрузить BLOSUM62 матрицу из текстового файла.
    Стандартный формат: строки с '#' — комментарии,
    первая нестрочная строка — заголовок (порядок аминокислот),
    остальные — строки матрицы."""
    ...

def column_entropy(col: np.ndarray) -> float:
    """Shannon entropy одной колонки профиля.
    entropy = -sum(p * log2(p + 1e-9) for p in col)
    0 = полностью консервативная, log2(A) = случайная."""
    ...

def profile_scalar_features(profile1: np.ndarray, profile2: np.ndarray,
                              seq_type: str = "dna") -> np.ndarray:
    """Скалярные признаки для двух профилей, shape (SCALAR_DIM,) dtype=float32.

    Признаки:
      L1, L2, L1/L2, abs(L1-L2)/max(L1,L2)          — 4 штуки
      mean_entropy(p1), mean_entropy(p2)              — 2 штуки
      std_entropy(p1), std_entropy(p2)                — 2 штуки
      gap_fraction(p1), gap_fraction(p2)              — 2 штуки
        (gap_fraction = доля колонок где p[last_idx] > 0.5)
      mean_profile_sim, max_profile_sim               — 2 штуки
        (сэмплировать 100 пар (i,j), считать
         sum_ab p1[i,a]*p2[j,b]*subst[a,b])

    Дополнить нулями до SCALAR_DIM=70.
    Нормализовать все в [0, 1]."""
    ...

def profile_similarity_matrix(profile1: np.ndarray, profile2: np.ndarray,
                               subst: np.ndarray,
                               target_size: int = 64) -> np.ndarray:
    """Матрица сходств профилей, shape (1, 64, 64) dtype=float32.

    sim = np.einsum('ia,jb,ab->ij', profile1, profile2, subst)
    → shape (L1, L2)
    Нормализовать в [0, 1] через min-max.
    Сжать до (target_size, target_size) через scipy.ndimage.zoom(order=1).
    Вернуть shape (1, target_size, target_size)."""
    ...

def make_input(obj1: str | np.ndarray,
               obj2: str | np.ndarray,
               seq_type: str = "dna") -> tuple[np.ndarray, np.ndarray]:
    """ГЛАВНАЯ ФУНКЦИЯ — единый интерфейс для нейросети.

    Автоматически определяет режим по типу obj:
      isinstance(obj1, str) → Режим 1 (последовательности)
        matrix  = dotplot_tensor(obj1, obj2, k=4 if dna else 3)
        scalars = kmer_features(obj1, obj2, seq_type)
      isinstance(obj1, np.ndarray) → Режим 2 (профили)
        subst   = DNA_SUBST if seq_type=='dna' else BLOSUM62
        matrix  = profile_similarity_matrix(obj1, obj2, subst)
        scalars = profile_scalar_features(obj1, obj2, seq_type)

    Возвращает:
      matrix:  np.ndarray shape (1, 64, 64) dtype=float32
      scalars: np.ndarray shape (SCALAR_DIM,) dtype=float32"""
    ...
```

### `features/anchors.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# Якорный подход для последовательностей длиннее MAX_DIRECT_LEN=5000.
# Идея: найти точные совпадения (якоря), разбить задачу на блоки между ними.
# Якоря = perfect match → не нужно выравнивать.
# Блоки между якорями = короткие подзадачи → нейросеть + banded NW.
# Используется как pre-processing перед нейросетью.

from dataclasses import dataclass
from features.kmer import minimizers

MAX_DIRECT_LEN = 5000

@dataclass
class Anchor:
    i: int   # позиция начала совпадения в seq1
    j: int   # позиция начала совпадения в seq2
    k: int   # длина совпадающего k-мера

def find_anchors(seq1: str, seq2: str,
                 window: int = 10, k: int = 15) -> list[Anchor]:
    """Найти общие минимайзеры как якоря.
    1. min1 = {kmer: [pos]} из minimizers(seq1, window, k)
    2. min2 = {kmer: [pos]} из minimizers(seq2, window, k)
    3. Пересечение ключей → список потенциальных Anchor(i, j, k)"""
    ...

def chain_anchors(anchors: list[Anchor], max_gap: int = 1000) -> list[Anchor]:
    """Найти монотонную цепочку якорей (LIS по парам (i, j)).
    Два якоря совместимы: a1.i < a2.i AND a1.j < a2.j AND
                          (a2.i - a1.i) < max_gap AND (a2.j - a1.j) < max_gap.
    O(n log n) через patience sorting."""
    ...

def split_by_anchors(seq1: str, seq2: str,
                     chain: list[Anchor]) -> list[tuple[str, str, int, int]]:
    """Разбить пару на блоки между якорями.
    Возвращает list[(block_seq1, block_seq2, offset_i, offset_j)].
    Блоки: подстроки seq1/seq2 между последовательными якорями.
    Сами якоря → perfect match, не включать в блоки."""
    ...
```

---

## Раздел 3: Нейросеть

### `model/band_predictor.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# Нейросеть предсказывает (centre_diag, half_width) для banded NW.
# Два входа: матрица сходств (1,64,64) + скалярный вектор (70,).
# Третий вход: seq_type (0=DNA, 1=protein) через Embedding.
# Выход: [centre_diag, log_half_width] → half_width = exp(log_hw).
#
# КРИТИЧЕСКИ ВАЖНО для loss:
#   Недооценка half_width опасна (вызывает doubling) → штраф ×5.
#   Переоценка half_width безопасна (тратим лишнее время).
#   Поэтому AsymmetricHuber с penalty=5.0 для underestimate.
#
# Дополнение из исследования (PLM embeddings — опционально):
#   Если доступны ProtT5/DNABERT-2 эмбеддинги, добавить их как
#   дополнительный вход через отдельный Linear(embedding_dim, 128).
#   Этот вход опционален — без него модель тоже работает.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

SCALAR_DIM  = 70
MATRIX_SIZE = 64
CNN_OUT_DIM = 256
MLP_OUT_DIM = 64
EMB_DIM     = 8
TOTAL_DIM   = CNN_OUT_DIM + MLP_OUT_DIM + EMB_DIM  # 328

class DotPlotCNN(nn.Module):
    """CNN ветка: (batch, 1, 64, 64) → (batch, 256).

    Архитектура (4 блока Conv-BN-ReLU-Pool):
    Блок 1: Conv2d(1→32, k=3, p=1) → BN → ReLU → MaxPool2d(2) → (32,32,32)
    Блок 2: Conv2d(32→64, k=3, p=1) → BN → ReLU → MaxPool2d(2) → (64,16,16)
    Блок 3: Conv2d(64→128, k=3, p=1) → BN → ReLU → MaxPool2d(2) → (128,8,8)
    Блок 4: Conv2d(128→128, k=3, p=1) → BN → ReLU → AdaptiveAvgPool2d(4) → (128,4,4)
    Flatten → Linear(2048→256) → ReLU → Dropout(0.3)

    AdaptiveAvgPool2d(4) вместо MaxPool2d(2) в последнем блоке
    позволяет принимать тензоры отличные от 64×64 без изменения архитектуры."""
    def __init__(self):
        super().__init__()
        # реализовать как описано выше
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 64, 64) → (batch, 256)
        ...

class ScalarMLP(nn.Module):
    """MLP ветка: (batch, SCALAR_DIM) → (batch, 64).

    Linear(SCALAR_DIM→128) → LayerNorm(128) → ReLU → Dropout(0.2)
    Linear(128→128) → LayerNorm(128) → ReLU → Dropout(0.2)
    Linear(128→64)

    LayerNorm вместо BatchNorm: работает корректно при batch_size=1
    во время инференса (когда нейросеть вызывается для одной пары)."""
    def __init__(self):
        super().__init__()
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, SCALAR_DIM) → (batch, 64)
        ...

class BandPredictor(nn.Module):
    """Главная модель: предсказывает band параметры.

    Входы:
      matrix:   (batch, 1, 64, 64) float32
      scalars:  (batch, SCALAR_DIM) float32
      seq_type: (batch,) int64  {0=DNA, 1=protein}

    Выход: (batch, 2) float32
      [:, 0] = centre_diag    (непрерывное число)
      [:, 1] = log_half_width (логарифм, exp() даёт положительный half_width)

    Архитектура объединения:
      cnn_out = DotPlotCNN(matrix)           → (batch, 256)
      mlp_out = ScalarMLP(scalars)           → (batch, 64)
      type_emb = Embedding(2, 8)(seq_type)   → (batch, 8)
      combined = cat([cnn_out, mlp_out, type_emb]) → (batch, 328)
      head: Linear(328→128) → ReLU → Dropout(0.2)
            Linear(128→32) → ReLU
            Linear(32→2)"""
    def __init__(self):
        super().__init__()
        self.cnn = DotPlotCNN()
        self.mlp = ScalarMLP()
        self.seq_type_emb = nn.Embedding(2, EMB_DIM)
        self.head = nn.Sequential(
            nn.Linear(TOTAL_DIM, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, matrix: torch.Tensor,
                scalars: torch.Tensor,
                seq_type: torch.Tensor) -> torch.Tensor:
        ...

    def predict(self, matrix: np.ndarray, scalars: np.ndarray,
                seq_type: str = "dna", device: str = "cpu") -> tuple[int, int]:
        """Инференс для одной пары (без батча).
        Добавить batch dim, прогнать forward, вернуть:
          centre_diag = int(round(output[0].item()))
          half_width  = max(1, int(round(exp(output[1].item()))))"""
        ...

def asymmetric_huber_loss(pred_log_hw: torch.Tensor,
                           true_hw: torch.Tensor,
                           delta: float = 1.0,
                           penalty: float = 5.0) -> torch.Tensor:
    """Асимметричный Huber loss для ширины band.
    true_log_hw = torch.log(true_hw.float() + 1.0)
    err = pred_log_hw - true_log_hw
    base = Huber(err, delta=delta)  # F.huber_loss(pred_log_hw, true_log_hw, ...)
    weight = torch.where(err < 0, torch.full_like(err, penalty), torch.ones_like(err))
    return (base * weight).mean()"""
    ...

def band_loss(pred: torch.Tensor,
              true_centre: torch.Tensor,
              true_hw: torch.Tensor,
              lam: float = 2.0) -> torch.Tensor:
    """Итоговый loss = MSE(centre) + lam * AsymmetricHuber(width).
    pred: (batch, 2) → pred[:,0]=centre, pred[:,1]=log_hw"""
    ...
```

### `model/train.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# Одноэтапное обучение:
#   Этап 1 (20 эпох): синтетические данные (ДНК + белки) из simulate.py
# Признаки предвычислить и кешировать → ускоряет обучение в 5-10x.
# WeightedRandomSampler → балансировка между low/medium/high дивергенцией.
# Метрика выбора лучшей модели: val band_recall@1x (доля без doubling).

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
import pandas as pd
import wandb
from pathlib import Path
from tqdm import tqdm
from model.band_predictor import BandPredictor, band_loss
from features.profile_features import make_input

class BandDataset(Dataset):
    """Датасет для обучения нейросети.

    Загружает .parquet файлы с колонками:
      seq1, seq2, centre_diag, true_half_width, divergence, seq_type

    __getitem__ возвращает dict:
      'matrix':     torch.Tensor (1, 64, 64) float32
      'scalars':    torch.Tensor (SCALAR_DIM,) float32
      'seq_type':   torch.Tensor () int64  {0=DNA, 1=protein}
      'centre':     torch.Tensor () float32
      'true_hw':    torch.Tensor () float32

    Если cache_dir задан: сохранять/загружать предвычисленные .npy признаки.
    Ключ кеша: f'{cache_dir}/{parquet_stem}_{idx}.npz'
    WeightedRandomSampler веса: обратно пропорциональны частоте divergence_group."""
    def __init__(self, parquet_paths: list[str], cache_dir: str | None = None): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict: ...
    def get_sample_weights(self) -> torch.Tensor: ...

def precompute_features(parquet_path: str, cache_dir: str, n_workers: int = 8):
    """Предвычислить все признаки параллельно и сохранить в .npy.
    Использовать multiprocessing.Pool(n_workers).
    Прогресс через tqdm. Пропускать уже существующие файлы."""
    ...

def train_epoch(model: BandPredictor, loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device: str) -> dict[str, float]:
    """Один epoch обучения. Возвращает {'loss': float, 'mae_centre': float}."""
    ...

def evaluate(model: BandPredictor, loader: DataLoader,
             device: str,
             multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)) -> dict[str, float]:
    """Валидация. Возвращает:
      'loss': float
      'band_recall@1x': float  — доля пар где true_hw <= pred_hw * 1.0
      'band_recall@1.5x': float
      'band_recall@2x': float
      'mae_centre': float
      'width_ratio': float  — mean(pred_hw / true_hw)"""
    ...

def train(config: dict):
    """Полный тренировочный цикл.

    config keys:
      data_dir, cache_dir, checkpoint_dir
      epochs_pretrain=20
      batch_size=128, lr=1e-3, weight_decay=1e-4
      lam=2.0, penalty=5.0
      patience=5  (early stopping)
      wandb_project, wandb_run_name
      device='cuda'

    Логировать в wandb каждую эпоху.
    Сохранять чекпоинт каждые 5 эпох + лучший по val band_recall@1x.
    Формат чекпоинта: {'epoch': int, 'model_state': ..., 'config': config}"""
    ...
```

### `model/evaluate.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# Батчевый GPU инференс для использования в MSA пайплайне.
# Нейросеть вызывается N-1 раз за MSA → группируем узлы одного уровня дерева в батч.
# Признаки вычисляются параллельно на CPU (ThreadPoolExecutor),
# затем один batched GPU forward pass.
# torch.compile() для PyTorch 2.x — дополнительное ускорение инференса.

import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from model.band_predictor import BandPredictor
from features.profile_features import make_input

class BandPredictorInference:
    """Батчевый GPU инференс для MSA пайплайна."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """Загрузить модель из чекпоинта, перевести в eval режим.
        Применить torch.compile(model) если PyTorch >= 2.0 и device='cuda'.
        Загружать чекпоинт через map_location=device."""
        ...

    def predict_batch(self,
                      pairs: list[tuple[str | np.ndarray, str | np.ndarray]],
                      seq_type: str = "dna") -> list[tuple[int, int]]:
        """Батчевый инференс для списка пар.

        Алгоритм:
        1. Параллельно вычислить признаки через ThreadPoolExecutor(max_workers=4):
           [(matrix, scalars) for obj1, obj2 in pairs]
        2. Stack в батч:
           matrices = torch.stack([torch.from_numpy(m) for m,s in features])
           scalars  = torch.stack([torch.from_numpy(s) for m,s in features])
           seq_types = torch.full((batch_size,), 0 if dna else 1, dtype=torch.long)
        3. Перенести на device, torch.no_grad(), model.forward()
        4. Декодировать выход:
           centre_diag = int(round(pred[:,0].item()))
           half_width  = max(1, int(round(exp(pred[:,1]).item())))
        5. Вернуть list[(centre_diag, half_width)]"""
        ...

    def predict_single(self,
                       obj1: str | np.ndarray,
                       obj2: str | np.ndarray,
                       seq_type: str = "dna") -> tuple[int, int]:
        """Обёртка для одной пары."""
        return self.predict_batch([(obj1, obj2)], seq_type)[0]
```

---

## Раздел 4: C++ ядро выравнивания

> **Для Copilot:** все C++ файлы компилируются в ЕДИНЫЙ Python модуль `aligner`
> через pybind11. Точка входа pybind11 — только в `band_doubling.cpp`.
> Остальные файлы `#include` друг друга. Порядок include:
> `full_nw.cpp` → `banded_nw.cpp` → `simd_banded_nw.cpp` →
> `four_russians.cpp` → `hirschberg.cpp` → `band_doubling.cpp`

### Общие C++ типы (определены в `banded_nw.cpp`, используются везде)

```cpp
// Результат одного выравнивания
struct BandedResult {
    float       score;          // оптимальный score
    std::string aligned_seq1;   // с гэпами '-'
    std::string aligned_seq2;   // с гэпами '-'
    bool        path_escaped;   // true → нужен band doubling
    bool        escape_left;    // путь ушёл левее band (i-j < centre-hw)
    bool        escape_right;   // путь ушёл правее band (i-j > centre+hw)
    int         max_deviation;  // max abs((i-j)-centre) по traceback
};

// Результат с doubling статистикой
struct DoublingResult {
    BandedResult alignment;
    int n_doublings;        // сколько раз удваивали
    int final_left_bound;   // итоговая левая граница band
    int final_right_bound;  // итоговая правая граница band
    bool used_hirschberg;
    bool used_four_russians;
    bool used_simd;
};

// Кодировка символов
// ДНК:   A=0, C=1, G=2, T=3, N=-1 (игнорировать)
// Белки: стандартная 20-буквенная → 0..19, X=-1
```

### `aligner/full_nw.cpp`

```cpp
// КОНТЕКСТ ДЛЯ COPILOT:
// Эталонный полный Needleman-Wunsch. Используется ТОЛЬКО для:
//   1. Верификации banded_nw во время разработки
//   2. Генерации обучающих данных (traceback пути) в simulate.py
// НЕ используется в продуктовом пайплайне.
//
// Аффинные штрафы (алгоритм Гото-Смита, 3 матрицы M, X, Y):
//   gap_score(k) = gap_open + gap_extend * k
//   gap_open = -10.0f, gap_extend = -0.5f (по умолчанию)
//
// Матрица замен:
//   ДНК: match=+1, mismatch=-1 (простая)
//   Белки: BLOSUM62 (передаётся через py::array_t<float>)

#include <string>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
constexpr float NEG_INF = -1e9f;

// Кодировщики символов
int encode_dna(char c);      // A=0,C=1,G=2,T=3,N=-1
int encode_protein(char c);  // 20 аминокислот → 0..19, X=-1

// Полное NW выравнивание (O(n*m) время и память)
BandedResult full_nw_align(
    const std::string& seq1,
    const std::string& seq2,
    float gap_open   = -10.0f,
    float gap_extend = -0.5f,
    bool  is_protein = false,
    const py::array_t<float>* subst_matrix = nullptr  // nullptr → +1/-1 для ДНК
);

// Traceback: возвращает list[(i,j)] от (0,0) до (len1,len2)
// Нужен для simulate.py чтобы вычислить band параметры
std::vector<std::pair<int,int>> full_nw_traceback(
    const std::string& seq1,
    const std::string& seq2,
    float gap_open   = -10.0f,
    float gap_extend = -0.5f,
    bool  is_protein = false,
    const py::array_t<float>* subst_matrix = nullptr
);
```

### `aligner/banded_nw.cpp`

```cpp
// КОНТЕКСТ ДЛЯ COPILOT:
// Базовый banded Needleman-Wunsch — ядро всей системы.
// Вычисляет только ячейки в band: j ∈ [i-centre-hw, i-centre+hw] ∩ [0,len2].
// Хранит ТОЛЬКО band в памяти: матрица (len1+1) × (2*hw+1), не (len1+1)×(len2+1).
// Три матрицы M, X, Y для аффинных штрафов (алгоритм Гото-Смита).
// Traceback: отслеживает escape_left, escape_right, max_deviation.
//
// Profile-profile версия: score(col_i, col_j) = einsum('a,b,ab->', p1[i], p2[j], subst)

#include "full_nw.cpp"  // для BandedResult struct и encode_*

BandedResult align_banded(
    const std::string& seq1,
    const std::string& seq2,
    int   centre_diag,
    int   half_width,
    float gap_open   = -10.0f,
    float gap_extend = -0.5f,
    bool  is_protein = false,
    const py::array_t<float>* subst_matrix = nullptr
);

// Profile-profile: profile1 shape (L1,A), profile2 shape (L2,A), subst shape (A,A)
BandedResult align_banded_profiles(
    const py::array_t<float>& profile1,
    const py::array_t<float>& profile2,
    const py::array_t<float>& subst,
    int   centre_diag,
    int   half_width,
    float gap_open   = -10.0f,
    float gap_extend = -0.5f
);
```

### `aligner/simd_banded_nw.cpp`

```cpp
// КОНТЕКСТ ДЛЯ COPILOT:
// SIMD AVX2 ускорение banded NW через антидиагональную параллелизацию.
// Антидиагональ d содержит ячейки (i,j): i+j=d, i-j ∈ [centre-hw, centre+hw].
// Все ячейки одной антидиагонали НЕЗАВИСИМЫ → вычислять 8 штук за раз через AVX2.
//
// AVX2 = 256 бит = 8 × float32.
// Операции: _mm256_max_ps (для max), _mm256_add_ps (для суммирования).
//
// Фолбэк: если HAVE_AVX2 не определён → использовать align_banded.
// align_banded_auto: автоматически выбирает AVX2 или скалярную версию.

#include <immintrin.h>
#include "banded_nw.cpp"

bool avx2_supported();  // runtime проверка через cpuid

#ifdef HAVE_AVX2
BandedResult align_banded_avx2(
    const std::string& seq1,
    const std::string& seq2,
    int   centre_diag,
    int   half_width,
    float gap_open   = -10.0f,
    float gap_extend = -0.5f,
    bool  is_protein = false
);
#endif

// Автоматический диспетчер (вызывать из Python и из band_doubling)
BandedResult align_banded_auto(
    const std::string& seq1,
    const std::string& seq2,
    int   centre_diag,
    int   half_width,
    float gap_open   = -10.0f,
    float gap_extend = -0.5f,
    bool  is_protein = false
);
```

### `aligner/four_russians.cpp`

```cpp
// КОНТЕКСТ ДЛЯ COPILOT:
// Метод четырёх русских (Arlazarov, Dinic, Kronrod, Faradzhev, 1970).
// Ускорение banded NW с O(n*W) до O(n*W/t) через lookup table блоков t×t.
//
// КЛЮЧЕВОЕ: FourRussiansAligner создаётся ОДИН РАЗ и переиспользуется:
//   - Между итерациями band_doubling (таблица накапливается)
//   - Внутри Hirschberg (передаётся по ссылке во все рекурсивные вызовы)
//   Это важно: hit_ratio растёт с каждым вызовом, достигая >90%.
//
// Параметр t: t = floor(log2(2*hw+1))
//   ДНК (|Σ|=4), t=4: таблица ~1.6M записей
//   Белки (|Σ|=20): t=min(t,2) — иначе таблица слишком большая
//
// Квантизация границ в B=16 уровней → конечный размер ключа → lookup table
//
// compute_block при промахе кеша → SIMD AVX2 (если HAVE_AVX2)
// set_max_table_bytes(512MB) → LRU eviction при превышении

#include "simd_banded_nw.cpp"
#include <unordered_map>

struct BlockBoundary {
    std::vector<float> bottom_row;  // нижняя граница блока (t+1 значений M,X,Y)
    std::vector<float> right_col;   // правая граница блока
};

class FourRussiansAligner {
public:
    FourRussiansAligner(int    block_size,    // t; если 0 → вычислить автоматически
                         bool   is_protein,
                         float  gap_open,
                         float  gap_extend,
                         int    quant_levels = 16,
                         const float* subst  = nullptr);

    // Для Hirschberg: только последняя строка, O(W) память
    // Накапливает lookup table между вызовами
    std::vector<float> last_row(
        const std::string& seq1,
        const std::string& seq2,
        int centre_diag, int half_width
    );

    // Полное выравнивание с traceback (standalone использование)
    BandedResult align(
        const std::string& seq1,
        const std::string& seq2,
        int centre_diag, int half_width
    );

    void reset_stats();
    struct Stats { int hits; int computed_simd; int computed_scalar; float hit_ratio; };
    Stats get_stats() const;
    size_t table_memory_bytes() const;
    void set_max_table_bytes(size_t max_bytes);

private:
    int t_; bool is_protein_; float go_, ge_; int B_;
    size_t max_bytes_ = 512ULL<<20;
    std::unordered_map<size_t, BlockBoundary> table_;
    Stats stats_{};

    static int compute_t(int half_width, bool is_protein);
    BlockBoundary compute_block_simd(const std::vector<float>& top,
                                      const std::vector<float>& left,
                                      const std::string& s1, const std::string& s2);
    BlockBoundary compute_block_scalar(const std::vector<float>& top,
                                        const std::vector<float>& left,
                                        const std::string& s1, const std::string& s2);
    size_t hash_block(const std::vector<float>& top, const std::vector<float>& left,
                      const std::string& s1, const std::string& s2);
    std::vector<int> quantize(const std::vector<float>& v);
};
```

### `aligner/hirschberg.cpp`

```cpp
// КОНТЕКСТ ДЛЯ COPILOT:
// Алгоритм Хиршберга (1975) поверх banded NW → O(W) память вместо O(n*W).
// Divide-and-conquer: делим seq1 пополам, находим точку разбиения на середине,
// рекурсивно решаем две подзадачи.
//
// КОМБИНАЦИЯ С FOUR RUSSIANS:
//   FourRussiansAligner создаётся в hirschberg_banded() ОДИН РАЗ.
//   Передаётся по ссылке во все рекурсивные вызовы hirschberg_banded_impl().
//   nw_last_row_fr() использует fr_aligner.last_row() → O(n*W/t) на каждый уровень.
//   Таблица накапливается по всей рекурсии → hit_ratio растёт с глубиной.
//
// КОМБИНАЦИЯ С SIMD:
//   Базовый случай (len1 <= BASE_CASE_LEN=64) → align_banded_auto() (AVX2 если доступен).
//   Внутри FR: compute_block_simd при промахе кеша.
//
// ИТОГОВЫЕ СЛОЖНОСТИ: O(W) память + O(n*W/t) время + SIMD константа ~8x.

#include "four_russians.cpp"

constexpr int BASE_CASE_LEN = 64;

// Один проход (forward или backward), только последняя строка, через Four Russians
std::vector<float> nw_last_row_fr(
    const std::string& seq1,
    const std::string& seq2,
    int   centre_diag,
    int   half_width,
    float gap_open,
    float gap_extend,
    bool  is_protein,
    FourRussiansAligner& fr  // по ссылке — накапливает кеш
);

// Найти оптимальную точку разбиения на строке mid, искать только внутри band
int find_split_point(
    const std::vector<float>& fwd,
    const std::vector<float>& bwd,
    int mid, int centre_diag, int half_width, int len2
);

// Рекурсивная реализация (fr передаётся по ссылке через всю рекурсию)
BandedResult hirschberg_banded_impl(
    const std::string& seq1, const std::string& seq2,
    int centre_diag, int half_width,
    float gap_open, float gap_extend, bool is_protein,
    FourRussiansAligner& fr
);

// Публичный интерфейс: создаёт fr_aligner, вызывает _impl
BandedResult hirschberg_banded(
    const std::string& seq1,
    const std::string& seq2,
    int   centre_diag,
    int   half_width,
    float gap_open   = -10.0f,
    float gap_extend = -0.5f,
    bool  is_protein = false
);

// Profile-profile версия
BandedResult hirschberg_banded_profiles(
    const py::array_t<float>& p1,
    const py::array_t<float>& p2,
    const py::array_t<float>& subst,
    int centre_diag, int half_width,
    float gap_open = -10.0f, float gap_extend = -0.5f
);
```

### `aligner/band_doubling.cpp`

```cpp
// КОНТЕКСТ ДЛЯ COPILOT:
// Асимметричный band doubling — гарантия корректности.
// + Диспетчер методов: Hirschberg + Four Russians + SIMD всегда вместе.
// + pybind11 точка входа для всего модуля aligner.
//
// АСИММЕТРИЧНОЕ РАСШИРЕНИЕ (экономия ~50% при односторонних эскейпах):
//   escape_left only:  new_left=centre-hw*2, new_right=centre+hw
//   escape_right only: new_left=centre-hw,   new_right=centre+hw*2
//   оба true:          new_left=centre-hw*2, new_right=centre+hw*2
//   new_centre = (new_left + new_right) / 2
//   new_hw     = (new_right - new_left) / 2
//
// ДИСПЕТЧЕР (при каждой итерации, методы не исключают друг друга):
//   estimated_mem = max(len1,len2) * (2*hw+1) * 3 * sizeof(float)
//   needs_hirschberg = estimated_mem > HIRSCHBERG_THRESHOLD (200 МБ)
//   use_four_russians = hw >= FR_MIN_HALF_WIDTH (16)
//   use_simd = HAVE_AVX2 (compile-time)
//
//   if needs_hirschberg: hirschberg_banded(seq1, seq2, centre, hw)
//                        (FR и SIMD используются внутри автоматически)
//   else if use_four_russians: FourRussiansAligner.align(...)
//   else: align_banded_auto(...)  (SIMD если HAVE_AVX2)
//
// FourRussiansAligner создаётся ОДИН РАЗ на весь вызов align_with_doubling,
// переиспользуется во всех итерациях (таблица накапливается).

#include "hirschberg.cpp"

constexpr long long HIRSCHBERG_THRESHOLD = 200LL << 20;  // 200 МБ
constexpr int       FR_MIN_HALF_WIDTH    = 16;

// Определить нужен ли Hirschberg
inline bool needs_hirschberg(int len1, int len2, int hw) {
    return (long long)std::max(len1,len2) * (2LL*hw+1) * 3 * sizeof(float)
           > HIRSCHBERG_THRESHOLD;
}

// Одна итерация с правильным методом (fr переиспользуется)
BandedResult run_one_iteration(
    const std::string& seq1, const std::string& seq2,
    int centre_diag, int half_width,
    float gap_open, float gap_extend, bool is_protein,
    FourRussiansAligner& fr,
    DoublingResult& result_meta  // записать флаги used_*
);

// Основная функция
DoublingResult align_with_doubling(
    const std::string& seq1,
    const std::string& seq2,
    int   pred_centre,
    int   pred_hw,
    float gap_open   = -10.0f,
    float gap_extend = -0.5f,
    bool  is_protein = false,
    const float* subst = nullptr
);

// Profile-profile версия
DoublingResult align_profiles_with_doubling(
    const py::array_t<float>& p1,
    const py::array_t<float>& p2,
    const py::array_t<float>& subst,
    int pred_centre, int pred_hw,
    float gap_open = -10.0f, float gap_extend = -0.5f
);

// ======= PYBIND11 MODULE (единственный во всём проекте) =======
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
        .def_readonly("hits",           &FourRussiansAligner::Stats::hits)
        .def_readonly("computed_simd",  &FourRussiansAligner::Stats::computed_simd)
        .def_readonly("hit_ratio",      &FourRussiansAligner::Stats::hit_ratio);

    // Попарное выравнивание
    m.def("align_banded",    &align_banded_auto,
          "Banded NW: scalar or SIMD AVX2 (auto)");
    m.def("align_hirschberg",&hirschberg_banded,
          "Hirschberg+FR+SIMD: O(W) memory, O(nW/logW) time");
    m.def("align_with_doubling", &align_with_doubling,
          "Guaranteed-optimal: asymmetric doubling + Hirschberg+FR+SIMD dispatcher");

    // Profile-profile выравнивание
    m.def("align_profiles",              &align_banded_profiles,
          "Profile-profile banded DP");
    m.def("align_profiles_with_doubling",&align_profiles_with_doubling,
          "Profile-profile with doubling fallback");

    // Эталон (только для верификации)
    m.def("full_nw_align",    &full_nw_align,    "Full NW (verification only)");
    m.def("full_nw_traceback",&full_nw_traceback,"Full NW traceback for simulate.py");

    // FourRussiansAligner как standalone (для экспериментов)
    py::class_<FourRussiansAligner>(m, "FourRussiansAligner")
        .def(py::init<int,bool,float,float,int,const float*>())
        .def("last_row",          &FourRussiansAligner::last_row)
        .def("align",             &FourRussiansAligner::align)
        .def("reset_stats",       &FourRussiansAligner::reset_stats)
        .def("get_stats",         &FourRussiansAligner::get_stats)
        .def("table_memory_bytes",&FourRussiansAligner::table_memory_bytes)
        .def("set_max_table_bytes",&FourRussiansAligner::set_max_table_bytes);
}
```

---

## Раздел 5: MSA пайплайн

### `msa/guide_tree.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# Guide tree определяет ПОРЯДОК объединений в прогрессивном MSA.
# Матрица дистанций: k-mer Jaccard, БЕЗ выравнивания (быстро).
# Параллелизация: joblib.Parallel по всем доступным ядрам.
# NJ (Neighbour-Joining) точнее UPGMA для дивергентных последовательностей.
# tree_levels() нужна для батчевого инференса нейросети по уровням дерева.

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from joblib import Parallel, delayed
from dataclasses import dataclass, field
from typing import Optional
from features.kmer import minimizers

@dataclass
class TreeNode:
    """Узел guide tree."""
    left:     Optional['TreeNode'] = None
    right:    Optional['TreeNode'] = None
    seq_idx:  Optional[int] = None      # только для листьев
    distance: float = 0.0
    node_id:  int = -1                  # уникальный ID для node_objects dict

def kmer_jaccard_dist(seq1: str, seq2: str, k: int = 4) -> float:
    """Jaccard дистанция по k-мерам: 1 - |set1 ∩ set2| / |set1 ∪ set2|.
    Быстро через set операции Python."""
    ...

def pairwise_distance_matrix(sequences: list[str],
                              seq_type: str = "dna",
                              n_jobs: int = -1) -> np.ndarray:
    """Симметричная матрица дистанций (N, N), нули на диагонали.
    k=4 для ДНК, k=3 для белков.
    Параллелизация: joblib.Parallel(n_jobs=n_jobs, backend='loky')
    Вычислять только верхний треугольник, затем симметризовать."""
    ...

def build_guide_tree(dist_matrix: np.ndarray,
                     method: str = "nj") -> TreeNode:
    """Построить guide tree.
    method='upgma': scipy linkage(squareform(dist_matrix), method='average')
    method='nj':    BioPython DistanceTreeConstructor с NJTreeConstructor
    Вернуть TreeNode — корень бинарного дерева."""
    ...

def tree_levels(root: TreeNode) -> list[list[TreeNode]]:
    """BFS обход: вернуть список уровней от листьев к корню.
    levels[0] = внутренние узлы у которых оба дочерних — листья
    levels[-1] = [root]
    Используется для батчевого инференса нейросети по уровням."""
    ...

def assign_node_ids(root: TreeNode) -> int:
    """DFS: присвоить уникальные node_id всем узлам. Возвращает max_id+1."""
    ...
```

### `msa/progressive_msa.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# Главный пайплайн MSA.
# Post-order обход guide tree (снизу вверх), N-1 шагов.
# На каждом шаге: нейросеть предсказывает band → C++ выравнивает.
#
# ЛЕНИВЫЕ ПРОФИЛИ (экономия памяти):
#   node_objects: dict[int, str | np.ndarray]
#   После объединения левого и правого → удалить оба, сохранить новый профиль.
#   В пике памяти: O(log N) профилей одновременно.
#
# БАТЧЕВЫЙ ИНФЕРЕНС ПО УРОВНЯМ:
#   tree_levels() возвращает узлы поуровнево.
#   Все пары одного уровня → один батч для нейросети.
#   Это критически важно для GPU эффективности при больших N.
#
# ЯКОРНЫЙ РЕЖИМ:
#   Если max(len(seq1), len(seq2)) > MAX_DIRECT_LEN:
#     anchors → split_by_anchors → выровнять блоки → склеить.

import numpy as np
import gc
from msa.guide_tree import pairwise_distance_matrix, build_guide_tree, tree_levels, TreeNode
from features.profile_features import make_input
from features.anchors import MAX_DIRECT_LEN, find_anchors, chain_anchors, split_by_anchors
from model.evaluate import BandPredictorInference
import aligner

def build_profile(aligned_seqs: list[str], seq_type: str = "dna") -> np.ndarray:
    """Построить профиль из набора выровненных последовательностей.
    aligned_seqs: list[str] — строки ОДИНАКОВОЙ длины (с гэпами)
    Возвращает np.ndarray shape (alignment_length, alphabet_size) dtype=float32.
    alphabet_size = DNA_PROF_SIZE=5 (ACGT-) или PROTEIN_PROF_SIZE=21 (20aa+-)
    profile[i, a] = count(a at column i) / len(aligned_seqs)"""
    ...

def apply_gaps_to_seqs(seqs: list[str], gap_pattern: str) -> list[str]:
    """Применить гэп-паттерн к набору последовательностей.
    gap_pattern: строка из 'M' (match) и '-' (gap).
    Для каждой строки в seqs: вставить '-' там где gap_pattern[k] == '-'.
    Используется для расширения профилей после profile-profile выравнивания."""
    ...

def align_pair_with_anchors(seq1: str, seq2: str,
                             predictor: BandPredictorInference,
                             seq_type: str) -> tuple[str, str]:
    """Выровнять длинную пару через якорный режим.
    1. find_anchors → chain_anchors → split_by_anchors
    2. Для каждого блока: predictor.predict_single + aligner.align_with_doubling
    3. Склеить: якоря (perfect match) + aligned блоки"""
    ...

def progressive_msa(sequences: list[str],
                    seq_ids: list[str],
                    predictor: BandPredictorInference,
                    seq_type: str = "dna",
                    tree_method: str = "nj",
                    n_jobs: int = -1) -> list[str]:
    """Главная функция прогрессивного MSA.

    Алгоритм:
    1. dist_matrix = pairwise_distance_matrix(sequences, seq_type, n_jobs)
    2. tree = build_guide_tree(dist_matrix, tree_method)
    3. assign_node_ids(tree)
    4. node_objects = {leaf.node_id: sequences[leaf.seq_idx] for leaf in leaves}
       seq_groups = {leaf.node_id: [sequences[leaf.seq_idx]] for leaf in leaves}
    5. Для каждого уровня в tree_levels(tree) (снизу вверх):
       a. pairs = [(node_objects[n.left.node_id], node_objects[n.right.node_id])
                   for n in level]
       b. predictions = predictor.predict_batch(pairs, seq_type)
       c. Для каждого узла n, (centre, hw) = predictions[i]:
          - obj1, obj2 = node_objects[n.left.node_id], node_objects[...]
          - if isinstance(obj1, str) and max(len(obj1),len(obj2)) > MAX_DIRECT_LEN:
              a1, a2 = align_pair_with_anchors(obj1, obj2, predictor, seq_type)
            else if isinstance(obj1, str):
              r = aligner.align_with_doubling(obj1, obj2, centre, hw)
              a1, a2 = r.alignment.aligned_seq1, r.alignment.aligned_seq2
            else (профили):
              subst = DNA_SUBST if dna else BLOSUM62
              r = aligner.align_profiles_with_doubling(obj1, obj2, subst, centre, hw)
              # применить паттерн гэпов к seq_groups
          - new_seqs = apply_gaps_to_seqs(seq_groups[left], a1.gap_pattern)
                     + apply_gaps_to_seqs(seq_groups[right], a2.gap_pattern)
          - new_profile = build_profile(new_seqs, seq_type)
          - node_objects[n.node_id] = new_profile
          - seq_groups[n.node_id] = new_seqs
          - del node_objects[n.left.node_id], node_objects[n.right.node_id]
          - del seq_groups[n.left.node_id], seq_groups[n.right.node_id]
          - gc.collect()
    6. Вернуть seq_groups[root.node_id]  # list[str] финального MSA"""
    ...
```

### `msa/iterative_refine.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# MUSCLE-style итеративное уточнение MSA.
# 3 прохода: на каждом проходе перевыравниваем каждую последовательность
# против профиля всех остальных (sequence-vs-profile режим).
# Принять новое выравнивание если sp_score_internal улучшился.
# При N > 100: уточнять только случайные 30% последовательностей за проход.

import numpy as np
import random
from msa.progressive_msa import build_profile, apply_gaps_to_seqs
from model.evaluate import BandPredictorInference
from features.profile_features import make_input
from scoring.metrics import sp_score_internal
import aligner

N_ITER       = 3
SAMPLE_FRAC  = 0.30

def remove_and_compact(msa: list[str],
                        idx: int) -> tuple[list[str], list[int], str]:
    """Убрать строку idx из MSA. Удалить пустые колонки (только гэпы).
    Возвращает:
      compact_msa:  list[str] без строки idx, без пустых колонок
      kept_cols:    list[int] — индексы сохранённых колонок
      removed_seq:  str — строка idx без гэпов"""
    ...

def reinsert_sequence(compact_msa: list[str],
                       aligned_seq: str,
                       aligned_profile_repr: str,
                       kept_cols: list[int],
                       original_len: int) -> list[str]:
    """Вставить переобновлённую строку обратно в MSA.
    aligned_seq: строка из результата выравнивания (с гэпами)
    aligned_profile_repr: consensus строка профиля (с гэпами) из того же выравнивания
    kept_cols: для восстановления полной длины MSA"""
    ...

def iterative_refine(msa: list[str],
                     sequences: list[str],
                     predictor: BandPredictorInference,
                     seq_type: str = "dna",
                     n_iter: int = N_ITER) -> list[str]:
    """Итеративное уточнение MSA.

    Для каждого прохода (n_iter раз):
      Если N > 100: выбрать случайные SAMPLE_FRAC*N индексов
      Для каждого выбранного idx:
        1. compact_msa, kept_cols, raw_seq = remove_and_compact(msa, idx)
        2. profile_rest = build_profile(compact_msa, seq_type)
        3. matrix, scalars = make_input(raw_seq, profile_rest, seq_type)
           # make_input поддерживает смешанный режим: str + np.ndarray
        4. centre, hw = predictor.predict_single(raw_seq, profile_rest, seq_type)
        5. r = aligner.align_with_doubling(raw_seq, profile_consensus(profile_rest),
                                            centre, hw)
           # profile_consensus: для каждой колонки взять argmax символ
        6. new_msa = reinsert_sequence(compact_msa, r.alignment.aligned_seq1,
                                        r.alignment.aligned_seq2, kept_cols, len(msa[0]))
        7. if sp_score_internal(new_msa) > sp_score_internal(msa):
               msa = new_msa
    Вернуть msa"""
    ...
```

---

## Раздел 6: Метрики и сравнение

### `scoring/band_metrics.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# Метрики качества предсказания band нейросетью.
# band_recall@1x — главная метрика: доля пар где путь попал в band без doubling.
# width_efficiency > 1 = переоцениваем (безопасно).
# width_efficiency < 1 = недооцениваем (опасно, будет doubling).

import numpy as np

def band_recall(true_path: list[tuple[int,int]],
                centre_diag: int,
                half_width: int) -> float:
    """Доля точек пути внутри band.
    Для каждой (i,j) в true_path: diag = i-j; inside = abs(diag-centre_diag) <= half_width
    return sum(inside) / len(true_path)"""
    ...

def width_efficiency(pred_hw: int, true_hw: int) -> float:
    """pred_hw / true_hw. 1.0=идеально, <1=опасно."""
    return pred_hw / max(true_hw, 1)

def mean_doublings(doubling_results: list) -> float:
    """Среднее n_doublings по списку DoublingResult."""
    ...

def band_recall_at(pred_hws: np.ndarray,
                   true_hws: np.ndarray,
                   multiplier: float) -> float:
    """Доля пар где true_hw <= pred_hw * multiplier."""
    return float((true_hws <= pred_hws * multiplier).mean())
```

### `scoring/metrics.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# Стандартные метрики качества MSA.
# SP-score и TC-score вычисляются относительно эталонного выравнивания.
# sp_score_internal — для итеративного уточнения (без эталона).

import numpy as np
import time, tracemalloc
from tqdm import tqdm

def sp_score(predicted_msa: list[str], reference_msa: list[str]) -> float:
    """Sum-of-Pairs score.
    Для каждой пары последовательностей (i, j):
      Для каждой позиции k в эталоне:
        Если pred_msa[i][k] != '-' AND pred_msa[j][k] != '-':
          Проверить что эта пара остатков совпадает с эталоном
    SP = правильных пар / всего пар в эталоне
    Учитывать только non-gap позиции в эталоне."""
    ...

def tc_score(predicted_msa: list[str], reference_msa: list[str]) -> float:
    """Total Column score.
    Для каждой колонки k в эталоне:
      Извлечь все non-gap символы → проверить что они совпадают с predicted
    TC = совпадающих колонок / всего колонок в эталоне"""
    ...

def sp_score_internal(msa: list[str]) -> float:
    """SP-score без эталона — для итеративного уточнения.
    Сумма match-score по всем попарным позициям (non-gap).
    Нормировать на количество пар × длину."""
    ...

def profile_consensus(profile: np.ndarray, seq_type: str = "dna") -> str:
    """Consensus строка профиля: для каждой колонки argmax символ.
    Если argmax = gap символ → вставить '-'."""
    ...

def benchmark(aligner_func, dataset: list[dict],
              measure_memory: bool = True) -> dict:
    """Прогнать aligner_func на всех примерах и замерить метрики.
    aligner_func(sequences: list[str], seq_ids: list[str]) → list[str]
    Измерять: SP-score, TC-score, время (perf_counter), пиковую память (tracemalloc).
    Возвращает dict с mean±std каждой метрики, сгруппированный по ref_class."""
    ...
```

### `baselines/classical.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# Обёртки над внешними MSA инструментами через subprocess.
# Все функции принимают list[str] последовательностей, возвращают list[str] выравнивания.
# Бинари должны быть в PATH: mafft, muscle, clustalw2.
# Использовать tempfile.NamedTemporaryFile для временных FASTA файлов.

import subprocess, tempfile, os
from Bio import AlignIO
from io import StringIO

def _seqs_to_fasta(sequences: list[str],
                   ids: list[str] | None = None) -> str:
    """Сформировать FASTA строку из списка последовательностей."""
    ...

def _parse_fasta_alignment(text: str) -> list[str]:
    """Разобрать FASTA выравнивание (с гэпами), вернуть list[str]."""
    ...

def run_mafft(sequences: list[str], ids: list[str] | None = None,
              extra_args: list[str] | None = None) -> list[str]:
    """Запустить MAFFT: mafft --auto --quiet [extra_args] input.fasta
    Парсить stdout как FASTA. Поднять RuntimeError если returncode != 0."""
    ...

def run_muscle(sequences: list[str], ids: list[str] | None = None) -> list[str]:
    """Запустить MUSCLE: muscle -in input.fasta -out output.fasta -quiet"""
    ...

def run_clustalw(sequences: list[str], ids: list[str] | None = None) -> list[str]:
    """Запустить ClustalW2: clustalw2 -INFILE=... -OUTFILE=... -OUTPUT=FASTA -QUIET"""
    ...
```

---

## Раздел 7: Эксперименты

### `experiments/compare.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# Финальный бенчмарк: сравнить все методы на синтетических ДНК-группах.
# ВАЖНО: ablation study с фиксированными band (W=30, W=100) обязателен —
#         он доказывает что улучшение от нейросети, а не от banded подхода.
#
# Методы для сравнения:
#   1. ClustalW   (baseline)
#   2. MAFFT      (baseline)
#   3. MUSCLE     (baseline)
#   4. Fixed W=30 (ablation: наш aligner без нейросети)
#   5. Fixed W=100 (ablation)
#   6. Neural band (наш метод)
#   7. Neural band + iterative refine (финальный)
#
# Группировать результаты по ref_class (RV11..RV50).
# Сохранить в CSV + построить графики (boxplot SP, scatter time vs SP).

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from experiments.run_all import _generate_dna_msa_group
from baselines.classical import run_mafft, run_muscle, run_clustalw
from msa.progressive_msa import progressive_msa
from msa.iterative_refine import iterative_refine
from scoring.metrics import benchmark
from model.evaluate import BandPredictorInference
import aligner

def fixed_band_aligner(sequences: list[str], seq_ids: list[str],
                        half_width: int = 30,
                        seq_type: str = "dna") -> list[str]:
    """MSA через прогрессивный алгоритм с ФИКСИРОВАННЫМ band.
    Вместо нейросети: centre_diag=0, half_width=half_width для всех пар.
    Используется для ablation study."""
    ...

def run_all(model_checkpoint: str,
            output_dir: str,
            device: str = "cuda"):
    """Запустить все 7 методов на синтетических ДНК-группах.
    Сохранить results.csv в output_dir.
    Построить и сохранить графики."""
    ...
```

---

## Раздел 8: Сборка C++

### `CMakeLists.txt`

```cmake
# КОНТЕКСТ ДЛЯ COPILOT:
# Единый модуль 'aligner' из 8 C++ файлов через pybind11.
# Флаги: -O3 -march=native -std=c++17 -DNDEBUG.
# AVX2: проверить через CheckCXXCompilerFlag → -mavx2 + -DHAVE_AVX2.
# OpenMP: опционально для параллелизации внутри C++.
# Устанавливать .so в корень проекта (рядом с Python кодом).

cmake_minimum_required(VERSION 3.15)
project(msa_band_neural CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# pybind11 через pip: pip install pybind11
find_package(pybind11 REQUIRED)
find_package(OpenMP)
include(CheckCXXCompilerFlag)

# Все исходники
set(ALIGNER_SOURCES
    aligner/full_nw.cpp
    aligner/banded_nw.cpp
    aligner/simd_banded_nw.cpp
    aligner/four_russians.cpp
    aligner/hirschberg.cpp
    aligner/band_doubling.cpp
    aligner/profile_dp.cpp
    aligner/anchored_align.cpp
)

pybind11_add_module(aligner ${ALIGNER_SOURCES})
target_compile_options(aligner PRIVATE -O3 -march=native -DNDEBUG)

# AVX2
check_cxx_compiler_flag("-mavx2" COMPILER_HAS_AVX2)
if(COMPILER_HAS_AVX2)
    target_compile_options(aligner PRIVATE -mavx2)
    target_compile_definitions(aligner PRIVATE HAVE_AVX2)
    message(STATUS "AVX2 enabled")
endif()

# OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(aligner PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(aligner PRIVATE HAVE_OPENMP)
    message(STATUS "OpenMP enabled")
endif()

# Установить в корень проекта
set_target_properties(aligner PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
```

---

## Раздел 9: Requirements и порядок сборки

### `requirements.txt`

```
# Deep Learning
torch>=2.2.0

# Данные
numpy>=1.26
pandas>=2.2
pyarrow>=15.0       # .parquet файлы
scipy>=1.12         # NJ дерево, zoom для dot-plot
biopython>=1.83     # парсинг FASTA, AlignIO

# ML утилиты
wandb               # логирование экспериментов
tqdm                # прогресс-бары
scikit-learn>=1.4   # WeightedRandomSampler, metrics

# C++ сборка
pybind11>=2.12
cmake>=3.15

# Параллелизация
joblib>=1.3         # параллельная матрица дистанций

# Визуализация
matplotlib>=3.8
seaborn>=0.13

# Тестирование
pytest>=8.0
pytest-benchmark    # бенчмарки C++ функций из Python
```

### Порядок сборки и запуска

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Собрать C++ модуль (из корня проекта)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..
# После сборки: aligner.cpython-311-x86_64-linux-gnu.so в корне проекта

# 3. Скачать BLOSUM62 матрицу
wget https://www.ncbi.nlm.nih.gov/Class/BLAST/BLOSUM/BLOSUM62.txt \
     -O data/blosum62.txt

# 4. Сгенерировать обучающие данные
python -m data.simulate --n_samples 500000 --seq_type dna \
       --output data/processed/train_dna.parquet --n_workers 16
python -m data.simulate --n_samples 200000 --seq_type protein \
       --output data/processed/train_protein.parquet --n_workers 16

# 5. Предвычислить признаки (опционально, ускоряет обучение)
python -m model.train precompute --data_dir data/processed/ \
       --cache_dir data/cache/ --n_workers 8

# 6. Обучить нейросеть
python -m model.train train \
       --data_dir data/processed/ \
       --cache_dir data/cache/ \
       --checkpoint_dir checkpoints/ \
       --device cuda

# 7. Финальный бенчмарк
python -m experiments.compare \
       --model_checkpoint checkpoints/best.pt \
       --output_dir results/ \
       --device cuda
```

---

## [ИНСТРУКЦИЯ ДЛЯ COPILOT — ПРАВИЛА ГЕНЕРАЦИИ]

> Это блок инструкций специально для Claude Opus/Sonnet при генерации кода.
> Следуй им строго при каждом запросе.

### Что делать при получении запроса на генерацию файла

1. **Читай весь Copilot System Context** в начале этого файла перед генерацией.
   Там определены все типы, константы, зависимости и паттерны.

2. **Используй точные имена** из раздела "Глобальные константы":
   `SCALAR_DIM=70`, `MATRIX_SIZE=64`, `CNN_OUT_DIM=256` и т.д.

3. **Не изобретай новые интерфейсы** — используй `make_input()`, `build_profile()`,
   `align_with_doubling()` как определено в Системном контексте.

4. **Каждую функцию реализовывай полностью** — не оставляй `...` в теле.
   Если что-то неясно — пиши минимальную рабочую реализацию, не заглушку.

5. **Для C++ файлов**: строго соблюдай порядок `#include`.
   Не добавляй отдельный `PYBIND11_MODULE` — он только в `band_doubling.cpp`.

6. **Типизация везде**: Python 3.11 type hints на все функции и методы.

7. **Комментарии**: только для неочевидной логики, не дублируй docstring.

8. **Тесты**: при генерации функции добавляй в конец файла блок
   `if __name__ == "__main__": ...` с базовой проверкой корректности.

### Какой файл генерировать первым

Соблюдай этот порядок (каждый следующий зависит от предыдущих):

```
Фаза 1 — C++ ядро:
  aligner/full_nw.cpp → aligner/banded_nw.cpp → aligner/simd_banded_nw.cpp
  → aligner/four_russians.cpp → aligner/hirschberg.cpp
  → aligner/band_doubling.cpp → aligner/profile_dp.cpp → aligner/anchored_align.cpp
  → CMakeLists.txt → [СОБРАТЬ И ПРОТЕСТИРОВАТЬ]

Фаза 2 — Данные и признаки:
  data/loaders.py → data/simulate.py
  → features/kmer.py → features/dotplot.py → features/profile_features.py
  → features/anchors.py

Фаза 3 — Нейросеть:
  model/band_predictor.py → model/train.py → model/evaluate.py
  → [ОБУЧИТЬ НА СИНТЕТИКЕ]

Фаза 4 — MSA пайплайн:
  msa/guide_tree.py → msa/progressive_msa.py → msa/iterative_refine.py

Фаза 5 — Эксперименты:
  baselines/classical.py → scoring/metrics.py → scoring/band_metrics.py
  → experiments/ablation.py → experiments/compare.py
```

### Частые ошибки которых нужно избегать

```
❌ Создавать PYBIND11_MODULE в hirschberg.cpp или four_russians.cpp
✓ PYBIND11_MODULE только в band_doubling.cpp

❌ make_input() с разными размерами выходных тензоров для разных режимов
✓ Всегда (1, 64, 64) + (SCALAR_DIM,) независимо от режима

❌ Хранить все профили дерева в памяти одновременно
✓ del child_profiles + gc.collect() после каждого объединения

❌ Вызывать нейросеть по одной паре в цикле
✓ Собирать батч по уровням дерева, predict_batch()

❌ Создавать FourRussiansAligner внутри Hirschberg при каждом рекурсивном вызове
✓ Создать один раз в hirschberg_banded(), передавать по ссылке

❌ Симметричное расширение band при doubling
✓ Асимметричное: расширять только в направлении эскейпа

❌ BatchNorm в ScalarMLP (не работает при batch_size=1)
✓ LayerNorm в ScalarMLP
```

---

## Раздел 10: Тесты производительности

> Эти тесты обязательны для диссертации — они количественно доказывают
> выигрыш предложенного метода.

### `tests/test_speed_pairwise.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# Сравнение скорости: полный NW vs banded NW vs наш метод (Hirschberg+FR+SIMD).
# Запускать через: pytest tests/test_speed_pairwise.py -v -s
# Результаты сохранять в CSV для таблиц диссертации.

import time
import numpy as np
import pandas as pd
import pytest
import sys
sys.path.insert(0, ".")
import aligner

DNA_ALPHABET = "ACGT"

def generate_pair(length: int, divergence: float,
                  seed: int = 42) -> tuple[str, str]:
    """Сгенерировать пару последовательностей с заданной дивергенцией.
    divergence = доля позиций которые отличаются (только замены, без indel).
    Для band: centre_diag=0, true_hw ≈ length * indel_rate."""
    rng = np.random.default_rng(seed)
    seq1 = ''.join(rng.choice(list(DNA_ALPHABET), length))
    seq2 = list(seq1)
    n_mut = int(length * divergence)
    positions = rng.choice(length, n_mut, replace=False)
    for p in positions:
        choices = [c for c in DNA_ALPHABET if c != seq2[p]]
        seq2[p] = rng.choice(choices)
    return seq1, ''.join(seq2)

def time_function(fn, n_runs: int = 5) -> float:
    """Замерить среднее время выполнения функции."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))

@pytest.mark.parametrize("length,divergence,true_hw", [
    (300,  0.05,  8),
    (500,  0.10,  20),
    (1000, 0.15,  50),
    (2000, 0.20,  120),
    (5000, 0.10,  80),
])
def test_pairwise_speedup(length, divergence, true_hw):
    """Сравнить Full NW vs Banded+FR+SIMD+Hirschberg.

    Ожидаемые speedup (теоретические):
      div=5%,  len=300:  ~15-30x
      div=10%, len=500:  ~20-40x
      div=15%, len=1000: ~25-50x
      div=20%, len=2000: ~15-25x
      div=10%, len=5000: ~30-60x (Hirschberg включается)"""
    seq1, seq2 = generate_pair(length, divergence)

    # Baseline: полный NW
    t_full = time_function(lambda: aligner.full_nw_align(seq1, seq2))

    # Наш метод: banded с точным band (имитация идеального предсказания нейросети)
    t_banded = time_function(
        lambda: aligner.align_with_doubling(seq1, seq2, 0, true_hw)
    )

    # Banded с завышенным band (имитация предсказания с запасом)
    t_wide = time_function(
        lambda: aligner.align_with_doubling(seq1, seq2, 0, true_hw * 2)
    )

    speedup_exact = t_full / t_banded
    speedup_wide  = t_full / t_wide

    print(f"\nlen={length:5d}, div={divergence:.0%}: "
          f"full={t_full*1000:.1f}ms, "
          f"banded(exact)={t_banded*1000:.1f}ms ({speedup_exact:.1f}x), "
          f"banded(wide)={t_wide*1000:.1f}ms ({speedup_wide:.1f}x)")

    assert speedup_exact > 3.0, f"Speedup слишком мал: {speedup_exact:.1f}x"

def test_save_results():
    """Сохранить результаты в CSV для диссертации."""
    rows = []
    configs = [
        (300,  0.05,  8,   "short_similar"),
        (500,  0.10,  20,  "medium_low"),
        (1000, 0.15,  50,  "medium"),
        (2000, 0.20,  120, "long_high"),
        (5000, 0.10,  80,  "verylong_low"),
    ]
    for length, div, hw, label in configs:
        seq1, seq2 = generate_pair(length, div)
        t_full   = time_function(lambda: aligner.full_nw_align(seq1, seq2))
        t_exact  = time_function(lambda: aligner.align_with_doubling(seq1, seq2, 0, hw))
        t_wide   = time_function(lambda: aligner.align_with_doubling(seq1, seq2, 0, hw*2))
        rows.append({
            "label":         label,
            "length":        length,
            "divergence":    div,
            "true_hw":       hw,
            "t_full_ms":     round(t_full * 1000, 2),
            "t_exact_ms":    round(t_exact * 1000, 2),
            "t_wide_ms":     round(t_wide * 1000, 2),
            "speedup_exact": round(t_full / t_exact, 1),
            "speedup_wide":  round(t_full / t_wide, 1),
        })
    df = pd.DataFrame(rows)
    df.to_csv("results/speed_pairwise.csv", index=False)
    print("\n" + df.to_string(index=False))
```

### `tests/test_four_russians.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# Проверить что FourRussiansAligner накапливает таблицу и ускоряется со временем.
# hit_ratio должен расти с каждой новой парой последовательностей.

import aligner
import numpy as np
import time

def test_fr_accumulation():
    """Hit ratio растёт по мере накопления lookup table."""
    fr = aligner.FourRussiansAligner(
        block_size=0, is_protein=False,
        gap_open=-10.0, gap_extend=-0.5, quant_levels=16
    )

    rng = np.random.default_rng(42)
    hit_ratios = []

    for i in range(200):
        seq1 = ''.join(rng.choice(list("ACGT"), 300))
        seq2 = ''.join(rng.choice(list("ACGT"), 300))
        fr.last_row(seq1, seq2, centre_diag=0, half_width=30)
        if i % 20 == 19:
            stats = fr.get_stats()
            hit_ratios.append(stats.hit_ratio)
            print(f"После {i+1} пар: hit_ratio={stats.hit_ratio:.1%}, "
                  f"table={fr.table_memory_bytes()//1024}KB")

    # hit_ratio должен расти
    assert hit_ratios[-1] > hit_ratios[0], "Hit ratio не растёт"
    assert hit_ratios[-1] > 0.3, f"Финальный hit ratio слишком мал: {hit_ratios[-1]:.1%}"
    print(f"\nИтого: {hit_ratios[0]:.1%} → {hit_ratios[-1]:.1%}")

def test_fr_vs_scalar_speed():
    """FourRussians должен быть быстрее обычного banded NW при большом half_width."""
    rng = np.random.default_rng(0)
    seqs = [(''.join(rng.choice(list("ACGT"), 1000)),
             ''.join(rng.choice(list("ACGT"), 1000))) for _ in range(50)]

    # Прогреть таблицу
    fr = aligner.FourRussiansAligner(0, False, -10.0, -0.5, 16)
    for s1, s2 in seqs[:20]:
        fr.last_row(s1, s2, 0, 60)

    # Сравнить скорость
    t0 = time.perf_counter()
    for s1, s2 in seqs[20:]:
        fr.align(s1, s2, 0, 60)
    t_fr = time.perf_counter() - t0

    t0 = time.perf_counter()
    for s1, s2 in seqs[20:]:
        aligner.align_banded(s1, s2, 0, 60)
    t_scalar = time.perf_counter() - t0

    speedup = t_scalar / t_fr
    print(f"\nFR speedup vs scalar: {speedup:.2f}x")
    assert speedup > 1.5, f"FR не даёт ускорения: {speedup:.2f}x"
```

### `tests/test_correctness.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# Проверить что banded NW с doubling даёт тот же результат что и полный NW.
# Это критически важно — корректность это наш главный инвариант.

import aligner
import numpy as np
import pytest

def generate_pair(length, divergence, seed):
    rng = np.random.default_rng(seed)
    seq1 = ''.join(rng.choice(list("ACGT"), length))
    seq2 = list(seq1)
    for i in rng.choice(length, int(length*divergence), replace=False):
        seq2[i] = rng.choice([c for c in "ACGT" if c != seq2[i]])
    # Добавить несколько indel для реализма
    n_indel = max(1, int(length * divergence * 0.3))
    ins_pos = sorted(rng.choice(len(seq2), n_indel, replace=False), reverse=True)
    for p in ins_pos[:n_indel//2]:
        seq2.insert(p, rng.choice(list("ACGT")))
    del_pos = sorted(rng.choice(len(seq2), n_indel//2, replace=False), reverse=True)
    for p in del_pos:
        if len(seq2) > 1:
            seq2.pop(p)
    return seq1, ''.join(seq2)

@pytest.mark.parametrize("length,div,seed", [
    (100,  0.05, 1),
    (200,  0.15, 2),
    (500,  0.25, 3),
    (1000, 0.10, 4),
    (300,  0.30, 5),
])
def test_banded_equals_full(length, div, seed):
    """Banded+doubling должен давать тот же score что и полный NW."""
    seq1, seq2 = generate_pair(length, div, seed)

    full   = aligner.full_nw_align(seq1, seq2)
    banded = aligner.align_with_doubling(seq1, seq2,
                                          pred_centre=0, pred_hw=5)  # намеренно узкий

    assert abs(full.score - banded.alignment.score) < 0.01, (
        f"Score mismatch: full={full.score:.2f}, banded={banded.alignment.score:.2f}"
    )
    print(f"\nlen={length}, div={div}: score={full.score:.2f}, "
          f"n_doublings={banded.n_doublings}, "
          f"used_hirschberg={banded.used_hirschberg}")

def test_hirschberg_equals_banded():
    """Hirschberg должен давать тот же score что и обычный banded NW."""
    rng = np.random.default_rng(99)
    for _ in range(20):
        seq1 = ''.join(rng.choice(list("ACGT"), 500))
        seq2 = ''.join(rng.choice(list("ACGT"), 500))
        r1 = aligner.align_banded(seq1, seq2, 0, 100)
        r2 = aligner.align_hirschberg(seq1, seq2, 0, 100)
        assert abs(r1.score - r2.score) < 0.01
```

### `tests/test_neural_vs_fixed.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# Ablation study: нейросеть vs фиксированный band.
# Запускать ПОСЛЕ обучения нейросети.
# Показывает что именно нейросеть даёт прирост, а не просто banded подход.

import time
import numpy as np
import pandas as pd
import pytest
from model.evaluate import BandPredictorInference
import aligner

CHECKPOINT = "checkpoints/best.pt"

@pytest.fixture(scope="module")
def predictor():
    return BandPredictorInference(CHECKPOINT, device="cpu")

def generate_test_pairs(n: int = 100, seed: int = 0):
    """Сгенерировать тестовые пары разных уровней дивергенции."""
    rng = np.random.default_rng(seed)
    pairs = []
    for div in [0.05, 0.10, 0.20, 0.30]:
        for i in range(n // 4):
            length = rng.integers(200, 1000)
            seq1 = ''.join(rng.choice(list("ACGT"), length))
            seq2 = list(seq1)
            for p in rng.choice(length, int(length*div), replace=False):
                seq2[p] = rng.choice([c for c in "ACGT" if c != seq2[p]])
            pairs.append((seq1, ''.join(seq2), div))
    return pairs

def test_ablation_study(predictor):
    """Сравнить нейросеть vs W=30 vs W=100 по скорости и n_doublings."""
    pairs = generate_test_pairs(100)
    rows = []

    for seq1, seq2, div in pairs:
        # Нейросеть
        centre, pred_hw = predictor.predict_single(seq1, seq2, seq_type="dna")
        t0 = time.perf_counter()
        r_neural = aligner.align_with_doubling(seq1, seq2, centre, pred_hw)
        t_neural = time.perf_counter() - t0

        # Фиксированный W=30
        t0 = time.perf_counter()
        r_fixed30 = aligner.align_with_doubling(seq1, seq2, 0, 30)
        t_fixed30 = time.perf_counter() - t0

        # Фиксированный W=100
        t0 = time.perf_counter()
        r_fixed100 = aligner.align_with_doubling(seq1, seq2, 0, 100)
        t_fixed100 = time.perf_counter() - t0

        rows.append({
            "divergence":       div,
            "length":           len(seq1),
            "pred_hw":          pred_hw,
            "n_doublings":      r_neural.n_doublings,
            "t_neural_ms":      t_neural * 1000,
            "t_fixed30_ms":     t_fixed30 * 1000,
            "t_fixed100_ms":    t_fixed100 * 1000,
            "speedup_vs_30":    t_fixed30 / t_neural,
            "speedup_vs_100":   t_fixed100 / t_neural,
            "score_match":      abs(r_neural.alignment.score - r_fixed100.alignment.score) < 0.01,
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/ablation_neural_vs_fixed.csv", index=False)

    print("\n=== ABLATION STUDY ===")
    print(df.groupby("divergence")[
        ["pred_hw", "n_doublings", "speedup_vs_30", "speedup_vs_100"]
    ].mean().round(2).to_string())
    print(f"\nScore match rate: {df.score_match.mean():.1%}")
    assert df.score_match.mean() > 0.99, "Нейросеть даёт неправильные scores!"
    assert df.speedup_vs_30.mean() > 1.2, "Нейросеть не быстрее W=30!"
```

### `tests/test_msa_quality.py`

```python
# КОНТЕКСТ ДЛЯ COPILOT:
# Финальный тест на синтетических ДНК: сравнение SP-score и времени.
# Запускать после полного обучения нейросети.
# Генерирует финальную таблицу для диссертации.

import time
import pandas as pd
import pytest
from experiments.run_all import _generate_dna_msa_group
from baselines.classical import run_mafft, run_muscle, run_clustalw
from msa.progressive_msa import progressive_msa
from msa.iterative_refine import iterative_refine
from scoring.metrics import sp_score, tc_score
from model.evaluate import BandPredictorInference
import aligner

CHECKPOINT   = "checkpoints/best.pt"

@pytest.fixture(scope="module")
def dna_test_groups():
    import numpy as np
    rng = np.random.RandomState(42)
    groups = []
    for div in ['low', 'medium', 'high']:
        for _ in range(3):
            groups.append(_generate_dna_msa_group(rng, divergence=div))
    return groups

@pytest.fixture(scope="module")
def predictor():
    return BandPredictorInference(CHECKPOINT, device="cpu")

def fixed_band_msa(sequences, seq_ids, half_width=30):
    """MSA с фиксированным band (ablation)."""
    # Использовать guide tree + align_with_doubling с centre=0, hw=half_width
    ...

def run_benchmark(aligner_fn, groups: list[dict]) -> pd.DataFrame:
    rows = []
    for g in groups:
        seqs = g["sequences"]
        ids  = g["seq_ids"]
        ref  = g["reference"]
        ref_class = g["ref_class"]
        t0 = time.perf_counter()
        try:
            msa = aligner_fn(seqs, ids)
            elapsed = time.perf_counter() - t0
            sp = sp_score(msa, ref)
            tc = tc_score(msa, ref)
        except Exception as e:
            elapsed, sp, tc = 999.0, 0.0, 0.0
        rows.append({"ref_class": ref_class, "sp": sp, "tc": tc, "time_s": elapsed})
    return pd.DataFrame(rows)

def test_full_comparison(dna_test_groups, predictor):
    """Финальная таблица сравнения всех методов."""
    methods = {
        "ClustalW":        lambda s, ids: run_clustalw(s, ids),
        "MAFFT":           lambda s, ids: run_mafft(s, ids),
        "MUSCLE":          lambda s, ids: run_muscle(s, ids),
        "Fixed_W30":       lambda s, ids: fixed_band_msa(s, ids, 30),
        "Fixed_W100":      lambda s, ids: fixed_band_msa(s, ids, 100),
        "Neural_band":     lambda s, ids: progressive_msa(s, ids, predictor),
        "Neural_+_refine": lambda s, ids: iterative_refine(
                               progressive_msa(s, ids, predictor), s, predictor),
    }

    all_results = {}
    for name, fn in methods.items():
        print(f"\nЗапуск {name}...")
        df = run_benchmark(fn, dna_test_groups)
        all_results[name] = df
        print(f"  SP={df.sp.mean():.3f}, TC={df.tc.mean():.3f}, "
              f"Time={df.time_s.mean():.2f}s")

    # Сводная таблица
    summary = pd.DataFrame({
        name: {
            "SP_mean":   df.sp.mean(),
            "TC_mean":   df.tc.mean(),
            "Time_mean": df.time_s.mean(),
        }
        for name, df in all_results.items()
    }).T.round(3)

    summary.to_csv("results/final_comparison.csv")
    print("\n=== ФИНАЛЬНАЯ ТАБЛИЦА ===")
    print(summary.to_string())
```

---

## [ПЕРВЫЙ ЗАПРОС ДЛЯ COPILOT — СКОПИРОВАТЬ ЦЕЛИКОМ]

```
You are generating a complete codebase for a master's thesis in bioinformatics.

STEP 1: Read the entire file `msa_band_prediction_plan.md` before writing any code.
Pay special attention to:
- [COPILOT SYSTEM CONTEXT] section — domain knowledge and data types
- Global constants (SCALAR_DIM=70, MATRIX_SIZE=64, etc.)
- File dependency map
- [ИНСТРУКЦИЯ ДЛЯ COPILOT] section — generation rules

STEP 2: Generate the complete codebase in this exact order, phase by phase.
After each phase, stop and tell me:
  ✓ What was generated
  ⚡ Command to compile/test this phase
  ▶ What comes next

GENERATION RULES (never violate these):
- Implement every function FULLY — no `...` or `pass` placeholders
- PYBIND11_MODULE only in band_doubling.cpp, nowhere else
- make_input() always returns (1,64,64) + (70,) regardless of input type
- LayerNorm (not BatchNorm) in ScalarMLP
- FourRussiansAligner created ONCE in hirschberg_banded(), passed by reference
- Asymmetric band doubling: expand only toward escape direction
- del child_profiles + gc.collect() after every tree node merge
- Add `if __name__ == "__main__":` smoke test at end of each Python file

PHASE ORDER:
Phase 1 (C++ kernel):
  aligner/full_nw.cpp
  aligner/banded_nw.cpp
  aligner/simd_banded_nw.cpp
  aligner/four_russians.cpp
  aligner/hirschberg.cpp
  aligner/band_doubling.cpp  ← PYBIND11_MODULE goes here
  aligner/profile_dp.cpp
  aligner/anchored_align.cpp
  CMakeLists.txt

Phase 2 (Data & Features):
  data/loaders.py
  data/simulate.py
  features/kmer.py
  features/dotplot.py
  features/profile_features.py
  features/anchors.py

Phase 3 (Neural Network):
  model/band_predictor.py
  model/train.py
  model/evaluate.py

Phase 4 (MSA Pipeline):
  msa/guide_tree.py
  msa/progressive_msa.py
  msa/iterative_refine.py
  msa/profile_align.py

Phase 5 (Experiments & Tests):
  baselines/classical.py
  scoring/metrics.py
  scoring/band_metrics.py
  experiments/ablation.py
  experiments/compare.py
  tests/test_correctness.py
  tests/test_speed_pairwise.py
  tests/test_four_russians.py
  tests/test_neural_vs_fixed.py
  tests/test_msa_quality.py

Start now with Phase 1. Generate aligner/full_nw.cpp first.
```
