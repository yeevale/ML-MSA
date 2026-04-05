# model/train.py — Two-stage training loop for the band predictor.
#   Stage 1 (pretrain):  synthetic data only
#   Stage 2 (finetune):  synthetic 80% + BAliBASE 20%
# Features are precomputed and cached → 5-10× speedup.
# WeightedRandomSampler balances low/medium/high divergence.
# Best model selected by val band_recall@1x (fraction without doubling).

import os
import sys
import math
import argparse
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, ConcatDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.band_predictor import BandPredictor, band_loss, SCALAR_DIM, MATRIX_SIZE
from features.profile_features import make_input


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _divergence_group(div: float) -> int:
    """Map divergence to group index: 0=low, 1=medium, 2=high."""
    if div < 0.10:
        return 0
    elif div < 0.25:
        return 1
    else:
        return 2


class BandDataset(Dataset):
    """Dataset loading .parquet files for band predictor training.

    Columns: seq1, seq2, centre_diag, true_half_width, divergence, seq_type
    Returns dict with tensors ready for model.forward().
    Optional cache_dir for precomputed features (.npz).
    """

    def __init__(self, parquet_paths: list[str],
                 cache_dir: str | None = None):
        frames = []
        for p in parquet_paths:
            frames.append(pd.read_parquet(p))
        self.df = pd.concat(frames, ignore_index=True)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.df)

    def _cache_path(self, idx: int) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"sample_{idx}.npz"

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        cp = self._cache_path(idx)

        if cp is not None and cp.exists():
            data = np.load(cp)
            matrix = data["matrix"]
            scalars = data["scalars"]
        else:
            seq_type = row["seq_type"] if "seq_type" in row.index else "dna"
            matrix, scalars = make_input(row["seq1"], row["seq2"], seq_type)
            if cp is not None:
                np.savez_compressed(cp, matrix=matrix, scalars=scalars)

        seq_type_str = row.get("seq_type", "dna")
        return {
            "matrix":   torch.from_numpy(matrix),
            "scalars":  torch.from_numpy(scalars),
            "seq_type": torch.tensor(0 if seq_type_str == "dna" else 1,
                                     dtype=torch.long),
            "centre":   torch.tensor(row["centre_diag"], dtype=torch.float32),
            "true_hw":  torch.tensor(row["true_half_width"], dtype=torch.float32),
        }

    def get_sample_weights(self) -> torch.Tensor:
        """Weights inversely proportional to divergence group frequency."""
        groups = self.df["divergence"].apply(_divergence_group).values
        counts = np.bincount(groups, minlength=3).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        class_weights = 1.0 / counts
        weights = class_weights[groups]
        return torch.from_numpy(weights).float()


# ---------------------------------------------------------------------------
# Precompute features
# ---------------------------------------------------------------------------

def _precompute_one(args: tuple) -> bool:
    """Worker for parallel feature precomputation."""
    idx, seq1, seq2, seq_type, cache_path = args
    if os.path.exists(cache_path):
        return False
    try:
        matrix, scalars = make_input(seq1, seq2, seq_type)
        np.savez_compressed(cache_path, matrix=matrix, scalars=scalars)
        return True
    except Exception:
        return False


def precompute_features(parquet_path: str, cache_dir: str,
                        n_workers: int = 8) -> None:
    """Precompute all features in parallel and save to .npz."""
    df = pd.read_parquet(parquet_path)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    tasks = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        cp = str(cache / f"sample_{idx}.npz")
        st = row.get("seq_type", "dna")
        tasks.append((idx, row["seq1"], row["seq2"], st, cp))

    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = list(tqdm(pool.imap(_precompute_one, tasks),
                                total=len(tasks), desc="Precomputing"))
    else:
        results = [_precompute_one(t) for t in tqdm(tasks, desc="Precomputing")]

    computed = sum(results)
    print(f"Precomputed {computed} new, {len(tasks) - computed} cached")


# ---------------------------------------------------------------------------
# Train / evaluate loops
# ---------------------------------------------------------------------------

def train_epoch(model: BandPredictor, loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device: str, lam: float = 2.0,
                penalty: float = 5.0) -> dict[str, float]:
    """One training epoch. Returns {'loss', 'mae_centre'}."""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n = 0

    for batch in loader:
        mat = batch["matrix"].to(device)
        scl = batch["scalars"].to(device)
        st = batch["seq_type"].to(device)
        centre = batch["centre"].to(device)
        true_hw = batch["true_hw"].to(device)

        pred = model(mat, scl, st)
        loss = band_loss(pred, centre, true_hw, lam=lam, penalty=penalty)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = mat.size(0)
        total_loss += loss.item() * bs
        total_mae += (pred[:, 0] - centre).abs().sum().item()
        n += bs

    return {"loss": total_loss / max(n, 1),
            "mae_centre": total_mae / max(n, 1)}


@torch.no_grad()
def evaluate(model: BandPredictor, loader: DataLoader,
             device: str, lam: float = 2.0, penalty: float = 5.0,
             multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)
             ) -> dict[str, float]:
    """Validation. Returns loss, band_recall@Nx, mae_centre, width_ratio."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n = 0

    recall_counts = {m: 0 for m in multipliers}
    width_ratios: list[float] = []

    for batch in loader:
        mat = batch["matrix"].to(device)
        scl = batch["scalars"].to(device)
        st = batch["seq_type"].to(device)
        centre = batch["centre"].to(device)
        true_hw = batch["true_hw"].to(device)

        pred = model(mat, scl, st)
        loss = band_loss(pred, centre, true_hw, lam=lam, penalty=penalty)

        bs = mat.size(0)
        total_loss += loss.item() * bs
        total_mae += (pred[:, 0] - centre).abs().sum().item()

        pred_hw = torch.exp(pred[:, 1]).clamp(min=1.0)

        for m in multipliers:
            recall_counts[m] += (true_hw <= pred_hw * m).sum().item()

        ratios = (pred_hw / true_hw.clamp(min=1.0)).cpu().numpy()
        width_ratios.extend(ratios.tolist())
        n += bs

    metrics: dict[str, float] = {
        "loss": total_loss / max(n, 1),
        "mae_centre": total_mae / max(n, 1),
        "width_ratio": float(np.mean(width_ratios)) if width_ratios else 0.0,
    }
    for m in multipliers:
        metrics[f"band_recall@{m}x"] = recall_counts[m] / max(n, 1)

    return metrics


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: dict) -> None:
    """Full two-stage training loop.

    config keys:
      data_dir, cache_dir, checkpoint_dir, results_dir
      train_parquet (optional, explicit path to train file)
      balibase_parquet (optional, for stage 2)
      epochs_pretrain=20, epochs_finetune=10
      batch_size=128, lr=1e-3, weight_decay=1e-4
      lam=2.0, penalty=5.0
      patience=5  (early stopping)
      wandb_project, wandb_run_name (optional)
      device='cuda'
    """
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(config["data_dir"])
    cache_dir = Path(config.get("cache_dir", "data/cache"))
    ckpt_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config.get("results_dir", "results/training"))
    results_dir.mkdir(parents=True, exist_ok=True)

    epochs_pre = config.get("epochs_pretrain", 20)
    epochs_ft = config.get("epochs_finetune", 10)
    batch_size = config.get("batch_size", 128)
    num_workers = config.get("num_workers", min(8, os.cpu_count() or 1))
    lr = config.get("lr", 1e-3)
    wd = config.get("weight_decay", 1e-4)
    lam = config.get("lam", 2.0)
    penalty = config.get("penalty", 5.0)
    patience = config.get("patience", 5)

    # wandb init (optional)
    use_wandb = "wandb_project" in config
    if use_wandb:
        import wandb
        wandb.init(project=config["wandb_project"],
                   name=config.get("wandb_run_name"),
                   config=config)

    # Collect parquet files
    # If --train_parquet is given explicitly, use only that file
    explicit_train = config.get("train_parquet")
    if explicit_train and Path(explicit_train).exists():
        train_parquets = [Path(explicit_train)]
        print(f"Using explicit train parquet: {explicit_train}")
    else:
        # Fallback: prefer train_combined > train_full > train.parquet
        # Do NOT glob train_*.parquet — it can pick up duplicates
        for name in ["train_combined.parquet", "train_full.parquet", "train.parquet"]:
            candidate = data_dir / name
            if candidate.exists():
                train_parquets = [candidate]
                print(f"Auto-detected train parquet: {candidate}")
                break
        else:
            # Last resort: glob
            train_parquets = sorted(data_dir.glob("train_*.parquet"))
            if not train_parquets:
                train_parquets = sorted(data_dir.glob("train.parquet"))

    val_parquets = sorted(data_dir.glob("val_*.parquet"))
    if not val_parquets:
        val_parquets = sorted(data_dir.glob("val.parquet"))
    if not train_parquets:
        raise FileNotFoundError(f"No train_*.parquet in {data_dir}")
    if not val_parquets:
        # Use 10% of train as validation
        val_parquets = train_parquets[-1:]
        train_parquets = train_parquets[:-1]

    train_ds = BandDataset([str(p) for p in train_parquets],
                           str(cache_dir / "train"))
    val_ds = BandDataset([str(p) for p in val_parquets],
                         str(cache_dir / "val"))

    weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=(device != "cpu"),
                              persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=(device != "cpu"),
                            persistent_workers=(num_workers > 0))

    model = BandPredictor().to(device)

    # Resume from checkpoint if specified
    resume_path = config.get("resume")
    if resume_path and Path(resume_path).exists():
        print(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state"]
        # Strip _orig_mod. prefix from torch.compile'd checkpoints
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned)

    # GPU optimizations
    if device != "cpu":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")  # use TensorFloat32 on RTX 4090
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception:
            pass  # torch.compile not available in older PyTorch

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs_pre + epochs_ft)

    best_recall = -1.0
    wait = 0
    history_epochs = []  # collect per-epoch metrics for training_history.json
    best_epoch_metrics = {}

    # Get the underlying (non-compiled) model for clean state_dict saving
    _raw_model = getattr(model, '_orig_mod', model)

    def run_stage(n_epochs: int, stage_name: str, start_epoch: int) -> int:
        nonlocal best_recall, wait, best_epoch_metrics
        for ep in range(n_epochs):
            epoch = start_epoch + ep
            train_metrics = train_epoch(model, train_loader, optimizer,
                                        device, lam, penalty)
            val_metrics = evaluate(model, val_loader, device, lam, penalty)
            scheduler.step()

            lr_now = optimizer.param_groups[0]["lr"]
            log = {f"train/{k}": v for k, v in train_metrics.items()}
            log.update({f"val/{k}": v for k, v in val_metrics.items()})
            log["lr"] = lr_now
            log["epoch"] = epoch

            # Record for history
            epoch_record = {"epoch": epoch, "stage": stage_name, "lr": lr_now}
            epoch_record.update({f"train_{k}": v for k, v in train_metrics.items()})
            epoch_record.update(val_metrics)
            history_epochs.append(epoch_record)

            print(f"[{stage_name}] Epoch {epoch}: "
                  f"train_loss={train_metrics['loss']:.4f} "
                  f"val_recall@1x={val_metrics['band_recall@1.0x']:.4f} "
                  f"val_mae={val_metrics['mae_centre']:.2f}")

            if use_wandb:
                import wandb
                wandb.log(log)

            # Checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state": _raw_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": config,
                }, ckpt_dir / f"checkpoint_epoch{epoch}.pt")

            # Best model
            recall = val_metrics["band_recall@1.0x"]
            if recall > best_recall:
                best_recall = recall
                wait = 0
                best_epoch_metrics = {"epoch": epoch, **val_metrics}
                torch.save({
                    "epoch": epoch,
                    "model_state": _raw_model.state_dict(),
                    "config": config,
                    "val_metrics": val_metrics,
                }, ckpt_dir / "best_model.pt")
                print(f"  → New best recall@1x: {best_recall:.4f}")
            else:
                wait += 1
                if wait >= patience:
                    print(f"  Early stopping after {patience} epochs without improvement")
                    return epoch + 1

        return start_epoch + n_epochs

    # Stage 1: pretrain on synthetic data
    print(f"=== Stage 1: Pretrain ({epochs_pre} epochs) ===")
    next_epoch = run_stage(epochs_pre, "pretrain", 0)

    # Stage 2: finetune with BAliBASE (if available)
    balibase_path = config.get("balibase_parquet")
    if balibase_path and Path(balibase_path).exists():
        print(f"\n=== Stage 2: Finetune ({epochs_ft} epochs) ===")
        # Mix synthetic 80% + BAliBASE 20%
        balibase_ds = BandDataset([balibase_path],
                                  str(cache_dir / "balibase"))
        combined_ds = ConcatDataset([train_ds, balibase_ds])

        # Recompute sampler weights
        n_syn = len(train_ds)
        n_bal = len(balibase_ds)
        w_syn = torch.full((n_syn,), 0.8 / max(n_syn, 1))
        w_bal = torch.full((n_bal,), 0.2 / max(n_bal, 1))
        combined_weights = torch.cat([w_syn, w_bal])
        combined_sampler = WeightedRandomSampler(
            combined_weights, len(combined_weights), replacement=True)

        train_loader = DataLoader(combined_ds, batch_size=batch_size,
                                  sampler=combined_sampler, num_workers=num_workers,
                                  pin_memory=(device != "cpu"),
                                  persistent_workers=(num_workers > 0))

        # Lower LR for finetuning
        for pg in optimizer.param_groups:
            pg["lr"] = lr * 0.1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_ft)

        wait = 0  # reset patience
        run_stage(epochs_ft, "finetune", next_epoch)
    else:
        print("\nNo BAliBASE parquet found, skipping Stage 2")

    if use_wandb:
        import wandb
        wandb.finish()

    # Save training history JSON
    import json
    history = {
        "best_epoch_metrics": best_epoch_metrics,
        "epochs": history_epochs,
        "config": {k: str(v) if isinstance(v, Path) else v
                   for k, v in config.items()},
    }
    history_path = results_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"Training history saved to: {history_path}")

    print(f"\nTraining complete. Best recall@1x: {best_recall:.4f}")
    print(f"Best model saved to: {ckpt_dir / 'best_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train band predictor")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--train_parquet", default=None,
                        help="Explicit path to training parquet file")
    parser.add_argument("--cache_dir", default="data/cache")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--results_dir", default="results/training")
    parser.add_argument("--balibase_parquet", default=None)
    parser.add_argument("--epochs_pretrain", type=int, default=20)
    parser.add_argument("--epochs_finetune", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lam", type=float, default=2.0)
    parser.add_argument("--penalty", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from (e.g. checkpoints/best_model.pt)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if len(sys.argv) == 1:
        # Smoke test with random data
        print("Running smoke test with random in-memory data...")
        import tempfile, pyarrow as pa, pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create tiny parquet
            n = 20
            rng = np.random.default_rng(42)
            data = {
                "seq1": ["ACGT" * 10 + "".join(rng.choice(list("ACGT"), 5))
                         for _ in range(n)],
                "seq2": ["ACGT" * 10 + "".join(rng.choice(list("ACGT"), 5))
                         for _ in range(n)],
                "centre_diag": rng.integers(-5, 5, n).tolist(),
                "true_half_width": (rng.integers(3, 15, n)).tolist(),
                "divergence": rng.uniform(0, 0.4, n).tolist(),
                "seq_type": ["dna"] * n,
            }
            pq.write_table(pa.Table.from_pydict(data),
                           os.path.join(tmpdir, "train_smoke.parquet"))
            pq.write_table(pa.Table.from_pydict(data),
                           os.path.join(tmpdir, "val_smoke.parquet"))

            cfg = {
                "data_dir": tmpdir,
                "cache_dir": os.path.join(tmpdir, "cache"),
                "checkpoint_dir": os.path.join(tmpdir, "ckpt"),
                "epochs_pretrain": 2,
                "epochs_finetune": 0,
                "batch_size": 8,
                "lr": 1e-3,
                "weight_decay": 0,
                "lam": 2.0,
                "penalty": 5.0,
                "patience": 100,
                "device": "cpu",
            }
            train(cfg)
        print("Smoke test passed!")
    else:
        cfg = vars(args)
        # Remove None values
        cfg = {k: v for k, v in cfg.items() if v is not None}
        train(cfg)
