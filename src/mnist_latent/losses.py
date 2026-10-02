"""VAE objectives, written per image so that their scales are explicit.

The second bug of the original implementation lives here. Its loss was

    mean_over_pixels((x_hat - x)^2) + KL(q(z|x) || p(z)),

i.e. a reconstruction term averaged over the 784 pixels next to a KL term summed over
the latent dimensions. On a per-image scale, the KL term is weighted 784 times more
than the per-image squared error; equivalently, the decoder is a Gaussian with a tiny
fixed variance. The exact "effective beta" depends on that variance, so the pipeline
does not rely on it: it measures the consequence (how many latent units stay active).
The fix is to optimise the actual evidence lower bound, with both terms summed per image.

Pixels are grey levels in [0, 1], not binarised, so the Bernoulli "likelihood" is a
cross-entropy rather than a normalised density. ELBO and IWAE values in nats are
therefore comparable between models of this repository only, not with the ~80-90 nats
usually reported on binarised MNIST.
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
    """The original objective, kept verbatim to reproduce its behaviour."""
    recon = F.mse_loss(x_hat, x)
    kl = kl_divergence(mu, logvar).mean()
    return recon + beta * kl, recon.detach(), kl.detach()


@torch.no_grad()
def iwae_bound(model, x: torch.Tensor, k: int = 64) -> torch.Tensor:
    """Importance-weighted lower bound on log p(x) (Burda et al., 2016), per image, in nats.

    log p(x) >= log (1/k) sum_i p(x|z_i) p(z_i) / q(z_i|x),  z_i ~ q(z|x).
    It is tighter than the ELBO (k=1) and converges to log p(x) as k grows. On grey-level
    pixels it bounds a cross-entropy, not a true log-likelihood (see the module docstring),
    so it is used to compare models within this repository. Requires the Bernoulli head.
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
