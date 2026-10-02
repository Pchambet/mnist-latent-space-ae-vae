"""Training loops for the autoencoder and the VAE, with checkpoint selection on validation.

The original lab kept ``best_state = model.state_dict()``, which is a live view of the
parameters: the "best" checkpoint silently became the last epoch. Here the best state
is deep-copied, and the epoch it came from is recorded.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn

from .losses import legacy_loss, negative_elbo
from .models import AE, VAE


@dataclass
class History:
    train: list[float] = field(default_factory=list)
    val: list[float] = field(default_factory=list)
    val_recon: list[float] = field(default_factory=list)
    val_kl: list[float] = field(default_factory=list)
    best_epoch: int = 0
    seconds: float = 0.0


def _batches(n: int, batch_size: int, gen: torch.Generator | None):
    order = torch.randperm(n, generator=gen) if gen is not None else torch.arange(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def _fit(
    model: nn.Module,
    step: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    n_train: int,
    evaluate: Callable[[], tuple[float, float, float]],
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> History:
    """Generic loop: ``step(idx)`` returns (loss, recon, kl) on a training mini-batch."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist, best, best_state = History(), float("inf"), None
    device = next(model.parameters()).device
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for idx in _batches(n_train, batch_size, gen):
            loss, _, _ = step(idx.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        val, val_recon, val_kl = evaluate()
        hist.train.append(total / n_train)
        hist.val.append(val)
        hist.val_recon.append(val_recon)
        hist.val_kl.append(val_kl)
        if val < best:
            best, best_state, hist.best_epoch = val, copy.deepcopy(model.state_dict()), epoch
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    hist.seconds = time.perf_counter() - t0
    return hist


@torch.no_grad()
def _batched_mean(fn: Callable[[slice], tuple], n: int, batch_size: int = 1000) -> tuple:
    """Size-weighted mean of a per-batch metric tuple over a whole split."""
    acc = None
    for start in range(0, n, batch_size):
        s = slice(start, min(start + batch_size, n))
        vals = torch.tensor([float(v) for v in fn(s)]) * (s.stop - s.start)
        acc = vals if acc is None else acc + vals
    return tuple((acc / n).tolist())


def train_ae(
    model: AE,
    x_clean: torch.Tensor,
    val_in: torch.Tensor,
    val_clean: torch.Tensor,
    *,
    x_in: torch.Tensor | None = None,
    corrupt: Callable[[torch.Tensor], torch.Tensor] | None = None,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
) -> History:
    """Fit ``model(input) ~ x_clean`` under per-pixel MSE.

    Inputs are either a fixed corrupted copy ``x_in`` (the lab protocol: one noise draw
    per image for the whole run) or produced on the fly by ``corrupt`` (fresh noise at
    every step, which the model cannot memorise).
    """
    if (x_in is None) == (corrupt is None):
        raise ValueError("pass exactly one of x_in and corrupt")
    mse = nn.MSELoss()

    def step(idx):
        target = x_clean[idx]
        inputs = x_in[idx] if x_in is not None else corrupt(target)
        loss = mse(model(inputs), target)
        return loss, loss.detach(), torch.zeros(())

    def evaluate():
        (val,) = _batched_mean(lambda s: (mse(model(val_in[s]), val_clean[s]),), len(val_in))
        return val, val, 0.0

    return _fit(model, step, len(x_clean), evaluate, epochs, batch_size, lr, seed)


def vae_objective(model: VAE, objective: str, beta: float):
    """Return ``f(x) -> (loss, recon, kl)`` for the 'elbo' or the 'legacy' objective."""
    if objective == "elbo":
        if model.head != "sigmoid":
            raise ValueError("the ELBO objective needs the Bernoulli (sigmoid) head")

        def f(x):
            logits, mu, logvar = model(x)
            return negative_elbo(x, logits, mu, logvar, beta)
    elif objective == "legacy":

        def f(x):
            x_hat, mu, logvar = model(x)
            return legacy_loss(x, model.mean_image(x_hat), mu, logvar, beta)
    else:
        raise ValueError(f"unknown objective {objective!r}")
    return f


def train_vae(
    model: VAE,
    x: torch.Tensor,
    val_x: torch.Tensor,
    *,
    objective: str = "elbo",
    beta: float = 1.0,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
) -> History:
    """Fit a VAE; checkpoints are selected on the validation value of the training objective."""
    f = vae_objective(model, objective, beta)

    def evaluate():
        # Same posterior samples at every epoch (comparable val curves), without
        # disturbing the training random stream.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            return _batched_mean(lambda s: f(val_x[s]), len(val_x))

    return _fit(model, lambda idx: f(x[idx]), len(x), evaluate, epochs, batch_size, lr, seed)
