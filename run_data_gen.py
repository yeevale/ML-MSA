"""run_data_gen.py — Standalone data generation runner."""
import sys
import os
# Disable tqdm so terminal output streams cleanly (no carriage-return overwrites)
os.environ["TQDM_DISABLE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from data.simulate import generate_dataset

Path("data/processed").mkdir(parents=True, exist_ok=True)

print("=== Generating train.parquet (50 000 samples) ===", flush=True)
generate_dataset(50000, "data/processed/train.parquet",
                 seq_type="dna", n_workers=1, seed=42)

print("=== Generating val.parquet (10 000 samples) ===", flush=True)
generate_dataset(10000, "data/processed/val.parquet",
                 seq_type="dna", n_workers=1, seed=9999)

print("=== ALL DONE ===", flush=True)
