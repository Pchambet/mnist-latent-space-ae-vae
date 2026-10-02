"""Evaluation: trivial baselines, latent-space diagnostics and a sample-quality score.

A denoiser is only useful if it beats doing nothing, and a VAE latent is only a latent
if the model uses it. Each metric here exists to make one of those checks explicit.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F


def per_image_mse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mean squared error per image, shape (N,)."""
    return (x_hat - x).pow(2).flatten(1).mean(1)


def psnr(mse: torch.Tensor) -> float:
    """Mean PSNR in dB over images, for intensities in [0, 1]."""
    return float((10 * torch.log10(1.0 / mse.clamp_min(1e-10))).mean())


def median_filter(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    """k x k median filter with reflect padding: the classical baseline for impulse-like noise."""
    pad = k // 2
    patches = F.unfold(F.pad(x, (pad, pad, pad, pad), mode="reflect"), k)  # (N, k*k, H*W)
    return patches.median(dim=1).values.view_as(x)


def paired_bootstrap_ci(
    a: torch.Tensor, b: torch.Tensor, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (a - b) over test images with a 95% percentile bootstrap interval.

    The interval reflects test-set sampling only, not training randomness.
    """
    d = (a - b).double().cpu().numpy()
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def active_units(mu: torch.Tensor, threshold: float = 0.01) -> tuple[int, torch.Tensor]:
    """Latent dimensions whose posterior mean varies across inputs (Burda et al., 2016).

    A unit is active when Var_x(E_q[z_u | x]) > threshold. A collapsed unit has the same
    posterior for every image, so it carries no information about the input.
    """
    var = mu.double().var(dim=0)
    return int((var > threshold).sum()), var.float()


def probe_accuracy(
    z_train: np.ndarray, y_train: np.ndarray, z_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Test accuracy of a standardised logistic regression predicting the digit from a code.

    It measures how much label information is linearly readable from the latent code;
    chance level is about 0.11 on MNIST.
    """
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(z_train, y_train)
    return float(clf.score(z_test, y_test))


class DigitClassifier(nn.Module):
    """Small CNN used only as a judge of generated samples (~99% test accuracy)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def classifier_score(probs: torch.Tensor) -> dict[str, float]:
    """Inception-score analogue on MNIST: exp E_x[KL(p(y|x) || p(y))].

    It ranges from 1 (every sample gets the same label distribution, e.g. one blurry
    "mean digit") to 10 (confident and evenly spread over the ten digits). Reported with
    its two ingredients: mean confidence and the entropy of the label marginal.
    """
    probs = probs.double().clamp_min(1e-12)
    marginal = probs.mean(0)
    kl = (probs * (probs.log() - marginal.log())).sum(1)
    return {
        "score": float(kl.mean().exp()),
        "confidence": float(probs.max(1).values.mean()),
        "label_entropy_nats": float(-(marginal * marginal.log()).sum()),
    }
