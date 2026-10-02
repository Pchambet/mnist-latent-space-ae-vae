import torch

from mnist_latent.data import add_uniform_noise, random_amplitude_noise


def test_noise_is_seeded_clipped_and_vanishes_at_zero_amplitude():
    x = torch.rand(10, 1, 28, 28)
    a, b = add_uniform_noise(x, 1.0, seed=3), add_uniform_noise(x, 1.0, seed=3)
    torch.testing.assert_close(a, b)
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0
    torch.testing.assert_close(add_uniform_noise(x, 0.0, seed=3), x)


def test_noise_error_matches_theory_on_mid_grey():
    """Away from the clipping bounds the MSE of U(-a/2, a/2) noise is a^2 / 12."""
    x = torch.full((200, 1, 28, 28), 0.5)
    mse = float((add_uniform_noise(x, 0.5, seed=0) - x).pow(2).mean())
    assert abs(mse - 0.5**2 / 12) < 2e-4


def test_clipping_halves_the_error_on_black_pixels():
    x = torch.zeros(200, 1, 28, 28)
    mse = float((add_uniform_noise(x, 0.5, seed=0) - x).pow(2).mean())
    assert abs(mse - 0.5 * 0.5**2 / 12) < 2e-4


def test_random_amplitude_noise_stays_in_range():
    x = torch.rand(50, 1, 28, 28)
    out = random_amplitude_noise(x, 2.0)
    assert out.shape == x.shape and float(out.min()) >= 0 and float(out.max()) <= 1
