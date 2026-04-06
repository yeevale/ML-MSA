# model/band_predictor.py — Neural network predicting (centre_diag, half_width)
# for banded Needleman-Wunsch alignment.
#
# Two inputs:  similarity matrix (1,64,64) + scalar vector (70,)
# Third input: seq_type (0=DNA, 1=protein) via Embedding
# Output:      [centre_diag, log_half_width]
#
# Asymmetric loss: underestimation of half_width is penalised 5× more
# (underestimate → band doubling, overestimate → just extra compute).

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

SCALAR_DIM  = 70
MATRIX_SIZE = 64
CNN_OUT_DIM = 256
MLP_OUT_DIM = 64
EMB_DIM     = 8
TOTAL_DIM   = CNN_OUT_DIM + MLP_OUT_DIM + EMB_DIM  # 328


class DotPlotCNN(nn.Module):
    """CNN branch: (batch, 1, 64, 64) → (batch, 256).

    4 blocks of Conv-BN-ReLU-Pool:
      Block 1: Conv2d(1→32,  k=3, p=1) → BN → ReLU → MaxPool2d(2)
      Block 2: Conv2d(32→64, k=3, p=1) → BN → ReLU → MaxPool2d(2)
      Block 3: Conv2d(64→128,k=3, p=1) → BN → ReLU → MaxPool2d(2)
      Block 4: Conv2d(128→128,k=3,p=1) → BN → ReLU → AdaptiveAvgPool2d(4)
    Flatten → Linear(2048→256) → ReLU → Dropout(0.3)
    """

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, CNN_OUT_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class ScalarMLP(nn.Module):
    """MLP branch: (batch, SCALAR_DIM) → (batch, 64).

    Linear(70→128) → LayerNorm → ReLU → Dropout(0.2)
    Linear(128→128) → LayerNorm → ReLU → Dropout(0.2)
    Linear(128→64)

    LayerNorm instead of BatchNorm so it works correctly at batch_size=1
    during inference.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SCALAR_DIM, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, MLP_OUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BandPredictor(nn.Module):
    """Main model predicting band parameters.

    Inputs:
      matrix:   (batch, 1, 64, 64) float32
      scalars:  (batch, SCALAR_DIM) float32
      seq_type: (batch,) int64  {0=DNA, 1=protein}

    Output: (batch, 2) float32
      [:, 0] = centre_diag     (continuous)
      [:, 1] = log_half_width  (exp() gives positive half_width)
    """

    def __init__(self):
        super().__init__()
        self.cnn = DotPlotCNN()
        self.mlp = ScalarMLP()
        self.seq_type_emb = nn.Embedding(2, EMB_DIM)
        self.head = nn.Sequential(
            nn.Linear(TOTAL_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

    def forward(self, matrix: torch.Tensor,
                scalars: torch.Tensor,
                seq_type: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn(matrix)                        # (B, 256)
        mlp_out = self.mlp(scalars)                       # (B, 64)
        type_emb = self.seq_type_emb(seq_type)            # (B, 8)
        combined = torch.cat([cnn_out, mlp_out, type_emb], dim=1)  # (B, 328)
        return self.head(combined)                        # (B, 2)

    @torch.no_grad()
    def predict(self, matrix: np.ndarray, scalars: np.ndarray,
                seq_type: str = "dna", device: str = "cpu") -> tuple[int, int]:
        """Single-pair inference (no batch dimension expected in inputs).
        Returns (centre_diag, half_width)."""
        self.eval()
        mat_t = torch.from_numpy(matrix).unsqueeze(0).to(device)    # (1,1,64,64)
        scl_t = torch.from_numpy(scalars).unsqueeze(0).to(device)   # (1,70)
        st_t = torch.tensor([0 if seq_type == "dna" else 1],
                            dtype=torch.long, device=device)        # (1,)
        out = self.forward(mat_t, scl_t, st_t)                      # (1,2)
        centre_diag = int(round(out[0, 0].item()))
        half_width = max(1, int(round(math.exp(out[0, 1].item()))))
        return centre_diag, half_width


def asymmetric_huber_loss(pred_log_hw: torch.Tensor,
                          true_hw: torch.Tensor,
                          delta: float = 1.0,
                          penalty: float = 15.0,
                          margin: float = 0.3) -> torch.Tensor:
    """Asymmetric Huber loss for band width.
    Underestimation (pred < true) penalised `penalty`× more.
    `margin` shifts the target upward in log-space so the model
    learns to predict ~exp(margin) wider than the true band,
    boosting recall@1x."""
    true_log_hw = torch.log(true_hw.float() + 1.0) + margin
    err = pred_log_hw - true_log_hw
    # Element-wise Huber
    abs_err = err.abs()
    base = torch.where(abs_err <= delta,
                       0.5 * err * err,
                       delta * (abs_err - 0.5 * delta))
    weight = torch.where(err < 0,
                         torch.full_like(err, penalty),
                         torch.ones_like(err))
    return (base * weight).mean()


def band_loss(pred: torch.Tensor,
              true_centre: torch.Tensor,
              true_hw: torch.Tensor,
              lam: float = 4.0,
              penalty: float = 15.0) -> torch.Tensor:
    """Combined loss = MSE(centre) + lam * AsymmetricHuber(width).
    pred: (batch, 2) — pred[:,0]=centre, pred[:,1]=log_hw."""
    centre_loss = F.mse_loss(pred[:, 0], true_centre.float())
    hw_loss = asymmetric_huber_loss(pred[:, 1], true_hw, penalty=penalty)
    return centre_loss + lam * hw_loss


if __name__ == "__main__":
    # Smoke test
    model = BandPredictor()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    batch = 4
    mat = torch.randn(batch, 1, MATRIX_SIZE, MATRIX_SIZE)
    scl = torch.randn(batch, SCALAR_DIM)
    st = torch.zeros(batch, dtype=torch.long)
    out = model(mat, scl, st)
    assert out.shape == (batch, 2), f"Bad output shape: {out.shape}"
    print(f"Forward: input mat={mat.shape}, scl={scl.shape} → out={out.shape}")

    # Loss
    true_c = torch.randn(batch)
    true_hw = torch.abs(torch.randn(batch)) * 50 + 1
    loss = band_loss(out, true_c, true_hw)
    assert loss.dim() == 0
    print(f"Loss: {loss.item():.4f}")

    # Single predict
    mat_np = np.random.randn(1, 64, 64).astype(np.float32)
    scl_np = np.random.randn(70).astype(np.float32)
    cd, hw = model.predict(mat_np, scl_np, "dna", "cpu")
    print(f"Predict: centre_diag={cd}, half_width={hw}")

    # Test batch_size=1 (MSA inference mode)
    model.eval()
    m = torch.randn(1, 1, 64, 64)
    s = torch.randn(1, 70)
    t = torch.zeros(1, dtype=torch.long)
    out = model(m, s, t)
    assert out.shape == (1, 2), f"Wrong shape: {out.shape}"
    print("OK: batch_size=1 works correctly")
    # Test batch_size=128 (training mode)
    m = torch.randn(128, 1, 64, 64)
    s = torch.randn(128, 70)
    t = torch.zeros(128, dtype=torch.long)
    out = model(m, s, t)
    assert out.shape == (128, 2)
    print("OK: batch_size=128 works correctly")

    print("Smoke test passed!")
