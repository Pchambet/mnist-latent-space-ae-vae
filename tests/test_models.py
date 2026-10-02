import torch
from torch import nn

from mnist_latent.models import AE, VAE


def test_autoencoder_shapes_and_output_range():
    x = torch.rand(4, 1, 28, 28)
    with torch.no_grad():
        out = AE(head="sigmoid")(x)
    assert out.shape == x.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_legacy_head_batch_normalises_the_output_image():
    last = AE(head="legacy").decoder[-1]
    assert isinstance(last, nn.Sequential) and isinstance(last[1], nn.BatchNorm2d)
    assert isinstance(AE(head="sigmoid").decoder[-1], nn.ConvTranspose2d)


def test_bottleneck_sizes():
    x = torch.rand(2, 1, 28, 28)
    assert AE().encoder(x).shape[1:] == (16, 1, 1)
    assert AE(channels=(32, 64, 128, 32)).encoder(x).shape[1:] == (32, 2, 2)


def test_vae_shapes_and_decoding_from_the_prior():
    vae = VAE(latent_dim=5)
    x = torch.rand(3, 1, 28, 28)
    logits, mu, logvar = vae(x)
    assert logits.shape == x.shape and mu.shape == logvar.shape == (3, 5)
    assert vae.decode(torch.randn(7, 5)).shape == (7, 1, 28, 28)


def test_reparameterize_is_differentiable_and_has_the_right_moments():
    mu = torch.full((20000, 1), 2.0, requires_grad=True)
    logvar = torch.full((20000, 1), 2 * torch.log(torch.tensor(3.0)).item(), requires_grad=True)
    z = VAE.reparameterize(mu, logvar)
    assert abs(float(z.mean()) - 2.0) < 0.1 and abs(float(z.std()) - 3.0) < 0.1
    z.sum().backward()
    assert mu.grad is not None and logvar.grad is not None
