import math

import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence as torch_kl

from mnist_latent.losses import (
    bernoulli_nll,
    iwae_bound,
    kl_divergence,
    legacy_loss,
    negative_elbo,
)
from mnist_latent.models import VAE


def test_kl_matches_torch_distributions():
    mu, logvar = torch.randn(5, 3), torch.randn(5, 3)
    expected = torch_kl(Normal(mu, (0.5 * logvar).exp()), Normal(0.0, 1.0)).sum(1)
    torch.testing.assert_close(kl_divergence(mu, logvar), expected)


def test_kl_is_zero_at_the_prior_and_known_by_hand():
    assert kl_divergence(torch.zeros(2, 4), torch.zeros(2, 4)).abs().max() < 1e-7
    # mu = 1, sigma = 1 in one dimension: KL = mu^2 / 2 = 0.5 nats
    assert math.isclose(float(kl_divergence(torch.ones(1, 1), torch.zeros(1, 1))), 0.5)


def test_bernoulli_nll_is_summed_per_image():
    x = torch.full((2, 1, 2, 2), 1.0)
    logits = torch.zeros(2, 1, 2, 2)  # p = 0.5 everywhere: 4 pixels * log 2
    torch.testing.assert_close(bernoulli_nll(x, logits), torch.full((2,), 4 * math.log(2)))


def test_negative_elbo_is_recon_plus_beta_kl():
    x = torch.rand(3, 1, 4, 4)
    logits, mu, logvar = torch.randn(3, 1, 4, 4), torch.randn(3, 2), torch.randn(3, 2)
    loss, recon, kl = negative_elbo(x, logits, mu, logvar, beta=4.0)
    torch.testing.assert_close(loss, recon + 4.0 * kl)
    torch.testing.assert_close(recon, bernoulli_nll(x, logits).mean())


def test_legacy_loss_weights_kl_784_times_more_than_a_per_image_error():
    """The bug in one line: per-pixel MSE + summed KL == (per-image SSE + 784 KL) / 784."""
    x, x_hat = torch.rand(4, 1, 28, 28), torch.rand(4, 1, 28, 28)
    mu, logvar = torch.randn(4, 16), torch.randn(4, 16)
    loss, _, kl = legacy_loss(x, x_hat, mu, logvar)
    sse = (x_hat - x).pow(2).flatten(1).sum(1).mean()
    torch.testing.assert_close(loss, (sse + 784 * kl) / 784)


def test_iwae_is_tighter_than_the_elbo():
    model = VAE(latent_dim=4).eval()
    x = torch.rand(64, 1, 28, 28)
    with torch.no_grad():
        logits, mu, logvar = model(x)
        neg_elbo, _, _ = negative_elbo(x, logits, mu, logvar)
    iw = iwae_bound(model, x, k=32).mean()
    assert iw > -neg_elbo  # log p(x) >= IWAE_k >= ELBO (in expectation)
