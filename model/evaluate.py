# model/evaluate.py — Batched GPU inference for the MSA pipeline.
# The neural net is called N-1 times per MSA; nodes at the same tree level
# are grouped into a single batch.
# Features computed in parallel on CPU (ThreadPoolExecutor),
# then one batched GPU forward pass.
# torch.compile() applied for PyTorch >= 2.0 on CUDA.

import math
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from model.band_predictor import BandPredictor
from features.profile_features import make_input


class BandPredictorInference:
    """Batched GPU inference wrapper for the MSA pipeline."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """Load model from checkpoint, set to eval mode.
        Apply torch.compile if PyTorch >= 2.0 and device is cuda."""
        self.device = device
        checkpoint = torch.load(checkpoint_path, map_location=device,
                                weights_only=False)
        self.model = BandPredictor()
        state_dict = checkpoint["model_state"]
        # Strip _orig_mod. prefix from torch.compile'd checkpoints
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(cleaned)
        self.model.to(device)
        self.model.eval()

        # torch.compile for PyTorch 2.x CUDA acceleration
        if device != "cpu":
            major = int(torch.__version__.split(".")[0])
            if major >= 2:
                try:
                    self.model = torch.compile(self.model)
                except Exception:
                    pass  # fallback to eager mode

    @torch.no_grad()
    def predict_batch(
        self,
        pairs: list[tuple[str | np.ndarray, str | np.ndarray]],
        seq_type: str = "dna",
    ) -> list[tuple[int, int]]:
        """Batched inference for a list of (obj1, obj2) pairs.

        1. Compute features in parallel on CPU via ThreadPoolExecutor
        2. Stack into batch tensors
        3. GPU forward pass
        4. Decode output → list of (centre_diag, half_width)
        """
        if not pairs:
            return []

        # 1. Parallel feature computation
        def compute_features(pair: tuple) -> tuple[np.ndarray, np.ndarray]:
            obj1, obj2 = pair
            return make_input(obj1, obj2, seq_type)

        n_workers = min(4, len(pairs))
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                features = list(pool.map(compute_features, pairs))
        else:
            features = [compute_features(p) for p in pairs]

        # 2. Stack into batch
        matrices = torch.stack([torch.from_numpy(m) for m, _ in features])
        scalars = torch.stack([torch.from_numpy(s) for _, s in features])
        seq_types = torch.full((len(pairs),),
                               0 if seq_type == "dna" else 1,
                               dtype=torch.long)

        # 3. Move to device and forward
        matrices = matrices.to(self.device)
        scalars = scalars.to(self.device)
        seq_types = seq_types.to(self.device)

        pred = self.model(matrices, scalars, seq_types)  # (B, 2)

        # 4. Decode
        results: list[tuple[int, int]] = []
        centres = pred[:, 0].cpu().numpy()
        log_hws = pred[:, 1].cpu().numpy()
        for i in range(len(pairs)):
            cd = int(round(float(centres[i])))
            hw = max(1, int(round(math.exp(float(log_hws[i])))))
            results.append((cd, hw))

        return results

    def predict_single(
        self,
        obj1: str | np.ndarray,
        obj2: str | np.ndarray,
        seq_type: str = "dna",
    ) -> tuple[int, int]:
        """Wrapper for a single pair."""
        return self.predict_batch([(obj1, obj2)], seq_type)[0]


if __name__ == "__main__":
    import tempfile, os

    # Smoke test: create a dummy checkpoint, load it, run inference
    model = BandPredictor()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_model.pt")
        torch.save({
            "model_state": model.state_dict(),
            "config": {},
        }, ckpt_path)

        inf = BandPredictorInference(ckpt_path, device="cpu")

        # Batch of 3 sequence pairs
        pairs = [
            ("ACGTACGTACGT" * 10, "ACGTACATACGT" * 10),
            ("GGGCCCTTTAAA" * 5, "GGGCCCTTAAAA" * 5),
            ("ATATATATATATAT", "ATAATATATATAT"),
        ]
        results = inf.predict_batch(pairs, seq_type="dna")
        assert len(results) == 3
        for cd, hw in results:
            assert isinstance(cd, int)
            assert isinstance(hw, int) and hw >= 1

        print(f"Batch results: {results}")

        # Single pair
        cd, hw = inf.predict_single("ACGT" * 20, "ACGT" * 20, "dna")
        print(f"Single: centre_diag={cd}, half_width={hw}")

        # Profile pairs
        p1 = np.random.dirichlet(np.ones(5), size=50).astype(np.float32)
        p2 = np.random.dirichlet(np.ones(5), size=40).astype(np.float32)
        cd, hw = inf.predict_single(p1, p2, "dna")
        print(f"Profile: centre_diag={cd}, half_width={hw}")

    print("Smoke test passed!")
