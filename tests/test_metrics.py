import torch

from mnist_latent import metrics as M


def test_median_filter_removes_isolated_impulses():
    x = torch.zeros(1, 1, 7, 7)
    noisy = x.clone()
    noisy[0, 0, 3, 3] = 1.0
    noisy[0, 0, 1, 5] = 1.0
    torch.testing.assert_close(M.median_filter(noisy), x)


def test_median_filter_keeps_a_constant_image():
    x = torch.full((2, 1, 5, 5), 0.3)
    torch.testing.assert_close(M.median_filter(x), x)


def test_active_units_recovers_the_informative_dimensions():
    """Ground truth: 3 dimensions vary with the input, 5 are constant up to tiny jitter."""
    g = torch.Generator().manual_seed(0)
    informative = torch.randn(2000, 3, generator=g)
    collapsed = 1e-3 * torch.randn(2000, 5, generator=g) + 0.2
    n, var = M.active_units(torch.cat([informative, collapsed], 1))
    assert n == 3
    assert var.shape == (8,)


def test_classifier_score_bounds():
    confident_spread = torch.eye(10).repeat(50, 1)
    one_label = torch.zeros(500, 10)
    one_label[:, 3] = 1.0
    uniform = torch.full((500, 10), 0.1)
    assert abs(M.classifier_score(confident_spread)["score"] - 10.0) < 1e-6
    assert abs(M.classifier_score(one_label)["score"] - 1.0) < 1e-6
    assert abs(M.classifier_score(uniform)["score"] - 1.0) < 1e-6


def test_paired_bootstrap_covers_the_true_shift():
    g = torch.Generator().manual_seed(1)
    b = torch.rand(4000, generator=g)
    a = b + 0.05 + 0.01 * torch.randn(4000, generator=g)
    mean, lo, hi = M.paired_bootstrap_ci(a, b, n_boot=500)
    assert lo < 0.05 < hi
    assert abs(mean - 0.05) < 1e-3


def test_psnr_of_known_mse():
    assert abs(M.psnr(torch.tensor([0.01, 0.01])) - 20.0) < 1e-5


def test_probe_reads_a_linearly_separable_code():
    g = torch.Generator().manual_seed(2)
    y = torch.randint(0, 3, (600,), generator=g)
    z = torch.nn.functional.one_hot(y, 3).float() * 4 + 0.3 * torch.randn(600, 3, generator=g)
    acc = M.probe_accuracy(z[:400].numpy(), y[:400].numpy(), z[400:].numpy(), y[400:].numpy())
    assert acc > 0.95
