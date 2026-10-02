"""VAE objectives, written per image so that their scales are explicit.

The second bug of the original lab lives here. Its loss was

    mean_over_pixels((x_hat - x)^2) + beta * KL(q(z|x) || p(z)),   beta = 1,

i.e. a reconstruction term averaged over the 784 pixels next to a KL term summed over
the latent dimensions. Rewritten on a per-image scale, that is the KL term weighted
784 times more than a per-image squared error: a beta-VAE with beta ~ 784, far past
the point where the optimum is to ignore the input (posterior collapse). The fix is to
optimise the actual evidence lower bound, with both terms summed per image.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Closed-form KL(N(mu, diag(exp(logvar))) || N(0, I)) per sample, in nats. Shape (B,)."""
    return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def bernoulli_nll(x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """-log p(x|z) per image for a Bernoulli decoder with grey levels as targets. Shape (B,)."""
    return F.binary_cross_entropy_with_logits(logits, x, reduction="none").flatten(1).sum(1)


def negative_elbo(
    x: torch.Tensor, logits: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch-mean of reconstruction NLL + beta * KL, in nats per image (beta=1: the true -ELBO)."""
    recon = bernoulli_nll(x, logits).mean()
    kl = kl_divergence(mu, logvar).mean()
    return recon + beta * kl, recon.detach(), kl.detach()


def legacy_loss(
    x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The original lab objective, kept verbatim to reproduce its behaviour."""
    recon = F.mse_loss(x_hat, x)
    kl = kl_divergence(mu, logvar).mean()
    return recon + beta * kl, recon.detach(), kl.detach()


@torch.no_grad()
def iwae_bound(model, x: torch.Tensor, k: int = 64) -> torch.Tensor:
    """Importance-weighted lower bound on log p(x) (Burda et al., 2016), per image, in nats.

    log p(x) >= log (1/k) sum_i p(x|z_i) p(z_i) / q(z_i|x),  z_i ~ q(z|x).
    It is tighter than the ELBO (k=1) and converges to log p(x) as k grows, so it is the
    fairer number to report as a test log-likelihood. Requires the Bernoulli head.
    """
    mu, logvar = model.encode(x)
    std = torch.exp(0.5 * logvar)
    log_w = []
    for _ in range(k):
        eps = torch.randn_like(mu)
        z = mu + std * eps
        log_px_z = -bernoulli_nll(x, model.decode(z))
        log_pz = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(1)
        log_qz = -0.5 * (eps.pow(2) + logvar + math.log(2 * math.pi)).sum(1)
        log_w.append(log_px_z + log_pz - log_qz)
    return torch.logsumexp(torch.stack(log_w), dim=0) - math.log(k)
