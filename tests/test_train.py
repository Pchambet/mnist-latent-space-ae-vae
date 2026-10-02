"""Training loops on synthetic data where the clean signal is known exactly."""

import torch

from mnist_latent.data import add_uniform_noise
from mnist_latent.models import AE, VAE
from mnist_latent.train import train_ae, train_vae


def _templates(n: int, seed: int = 0) -> torch.Tensor:
    """n copies of 4 fixed blob 'digits': the ground truth a denoiser should recover."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(28.0), torch.arange(28.0), indexing="ij")
    blobs = []
    for cy, cx in [(8, 8), (8, 20), (20, 8), (20, 20)]:
        blobs.append(torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 18.0))
    t = torch.stack(blobs).unsqueeze(1)
    return t[torch.randint(0, 4, (n,), generator=g)]


def test_denoiser_recovers_the_clean_templates_better_than_doing_nothing():
    clean, val_clean = _templates(512), _templates(256, seed=1)
    noisy, val_noisy = add_uniform_noise(clean, 1.0, 0), add_uniform_noise(val_clean, 1.0, 1)
    model = AE()
    hist = train_ae(model, clean, val_noisy, val_clean, x_in=noisy, epochs=8, batch_size=32)
    identity = float((val_noisy - val_clean).pow(2).mean())
    assert min(hist.val) < 0.5 * identity


def test_best_checkpoint_is_the_one_restored():
    clean, val_clean = _templates(256), _templates(128, seed=1)
    noisy, val_noisy = add_uniform_noise(clean, 1.0, 0), add_uniform_noise(val_clean, 1.0, 1)
    model = AE()
    hist = train_ae(model, clean, val_noisy, val_clean, x_in=noisy, epochs=4, batch_size=32)
    with torch.no_grad():
        val = float(torch.nn.functional.mse_loss(model(val_noisy), val_clean))
    assert abs(val - min(hist.val)) < 1e-6
    assert hist.val[hist.best_epoch - 1] == min(hist.val)


def test_vae_elbo_training_decreases_the_bound_and_uses_the_latent():
    x, val = _templates(512), _templates(256, seed=1)
    model = VAE(latent_dim=4)
    hist = train_vae(model, x, val, epochs=6, batch_size=32, lr=3e-3)
    assert min(hist.val) < hist.val[0]
    assert hist.val_kl[hist.best_epoch - 1] > 0.5  # four templates need about log 4 nats


def test_legacy_objective_collapses_the_posterior():
    """With the lab loss the KL term dominates: the posterior is pushed onto the prior."""
    x, val = _templates(512), _templates(256, seed=1)
    model = VAE(latent_dim=4, head="legacy")
    hist = train_vae(model, x, val, objective="legacy", epochs=6, batch_size=32, lr=3e-3)
    assert hist.val_kl[hist.best_epoch - 1] < 0.05
