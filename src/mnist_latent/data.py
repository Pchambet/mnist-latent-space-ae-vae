"""MNIST loading, the fixed train/val/test split, and the input-noise model.

Everything is held as in-memory float tensors in [0, 1]: MNIST is about 220 MB as float32,
and skipping per-item PIL transforms makes CPU training several times faster.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

RAW_DIR = Path("data/raw")
N_VAL = 10_000


@dataclass(frozen=True)
class Split:
    """Images as (N, 1, 28, 28) float tensors in [0, 1] and integer labels."""

    x: torch.Tensor
    y: torch.Tensor

    def __len__(self) -> int:
        return len(self.y)


def download(root: Path = RAW_DIR) -> Path:
    """Download MNIST once (torchvision mirrors, checksums verified); no-op when cached."""
    from torchvision import datasets

    root.mkdir(parents=True, exist_ok=True)
    for train in (True, False):
        datasets.MNIST(root=root, train=train, download=True)
    return root


def _load(train: bool, root: Path) -> Split:
    from torchvision import datasets

    ds = datasets.MNIST(root=root, train=train, download=False)
    x = ds.data.unsqueeze(1).float() / 255.0
    return Split(x=x, y=ds.targets.clone())


def load_splits(root: Path = RAW_DIR, seed: int = 0) -> dict[str, Split]:
    """Official 60k/10k split, with the last 10k of a seeded shuffle of train held out as val.

    The test set is never used for model selection: checkpoints are chosen on val.
    """
    full = _load(True, root)
    perm = torch.randperm(len(full), generator=torch.Generator().manual_seed(seed))
    tr, va = perm[:-N_VAL], perm[-N_VAL:]
    return {
        "train": Split(full.x[tr], full.y[tr]),
        "val": Split(full.x[va], full.y[va]),
        "test": _load(False, root),
    }


def add_uniform_noise(x: torch.Tensor, amplitude: float, seed: int) -> torch.Tensor:
    """Corrupt images with additive U(-amplitude/2, amplitude/2) noise, clipped to [0, 1].

    This is the corruption of the original implementation (``a * (U(0,1) - 0.5)``).
    Clipping matters: about 80% of MNIST pixels are exactly 0, so half of the noise on
    them is clipped away and the noisy input is much closer to the clean image than the
    nominal noise variance a^2 / 12 suggests. The noise is drawn once per split with a fixed seed, so every
    model is evaluated on the very same corrupted test images.
    """
    g = torch.Generator().manual_seed(seed)
    noise = torch.rand(x.shape, generator=g) - 0.5
    return (x + amplitude * noise).clamp_(0.0, 1.0)


def random_amplitude_noise(x: torch.Tensor, max_amplitude: float) -> torch.Tensor:
    """Fresh uniform noise with a per-image amplitude ~ U(0, max_amplitude), clipped to [0, 1].

    Used to train a single "blind" denoiser that must work at any noise level.
    """
    amp = torch.rand(x.shape[0], 1, 1, 1, device=x.device) * max_amplitude
    noise = torch.rand(x.shape, device=x.device) - 0.5
    return (x + amp * noise).clamp(0.0, 1.0)
