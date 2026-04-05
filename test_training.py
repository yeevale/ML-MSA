"""Quick smoke test: 3 epochs on 150 samples to verify pipeline."""
import os, sys, shutil
os.environ['TQDM_DISABLE'] = '1'

# Clean previous test artifacts
for d in ['data/test_cache', 'checkpoints/test_run']:
    if os.path.exists(d):
        shutil.rmtree(d)

from model.train import train

config = {
    'data_dir': 'data/test_run',
    'cache_dir': 'data/test_cache',
    'checkpoint_dir': 'checkpoints/test_run',
    'epochs_pretrain': 3,
    'epochs_finetune': 0,
    'batch_size': 32,
    'device': 'cpu',
    'patience': 10,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'lam': 2.0,
    'penalty': 5.0,
}

print("Starting training...", flush=True)
train(config)
print("TRAINING COMPLETE", flush=True)

# Verify checkpoint
import torch
ckpt_path = 'checkpoints/test_run/best_model.pt'
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"Checkpoint epoch: {ckpt['epoch']}", flush=True)
    print(f"Val metrics: {ckpt.get('val_metrics', {})}", flush=True)
    print("CHECKPOINT OK", flush=True)
else:
    print("ERROR: no best_model.pt found!", flush=True)
    sys.exit(1)
