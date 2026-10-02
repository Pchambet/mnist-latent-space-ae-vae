"""The full study: train every model once (cached), evaluate, and write results/*.json.

Experiments
-----------
Denoising autoencoder
    ae_original      original architecture and output head (BatchNorm on the output
                     image), original noise protocol (amplitude 0.5, one fixed draw per
                     image), retrained here with the protocol below
    ae_fixed_head    same, with the corrected sigmoid output head
    ae_blind_narrow  original 5-layer network (16-dim code), trained on fresh noise of
                     random amplitude
    ae_blind_wide    a different 4-layer network (32x2x2 = 128-dim code, more parameters),
                     same training; it differs in depth and width, not only in code size
Variational autoencoder
    vae_original     original loss (pixel-mean MSE + summed KL)
    vae_elbo_b{k}    Bernoulli ELBO with KL weight beta = k (beta = 1 is the unweighted bound)
    vae_elbo_2d      two-dimensional latent, for the latent-space map

Every model gets the same initial random state (``Config.seed``) right before it is
built, so its weights do not depend on which other models were trained or loaded from
the cache.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import metrics as M
from .data import Split, add_uniform_noise, load_splits, random_amplitude_noise
from .losses import bernoulli_nll, iwae_bound, kl_divergence
from .models import AE, VAE
from .train import History, _batched_mean, train_ae, train_vae

CKPT_DIR = Path("data/interim/checkpoints")
RESULTS_DIR = Path("results")
ARRAYS_PATH = Path("data/interim/arrays.npz")


@dataclass(frozen=True)
class Config:
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 0
    ref_amplitude: float = 0.5
    max_amplitude: float = 2.0
    eval_amplitudes: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    latent_dim: int = 16
    betas: tuple[float, ...] = (1.0, 4.0, 16.0, 64.0)
    iwae_k: int = 64
    n_samples: int = 10_000
    wide_channels: tuple[int, ...] = field(default=(32, 64, 128, 32))


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cached(name: str, model: nn.Module, fit, retrain: bool) -> dict:
    """Train ``model`` with ``fit()`` unless a checkpoint exists; return its history dict."""
    path = CKPT_DIR / f"{name}.pt"
    if path.exists() and not retrain:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state"])
        model.eval()
        return ckpt["history"]
    hist: History = fit()
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"state": state, "history": asdict(hist)}, path)
    print(f"  trained {name}: best epoch {hist.best_epoch}, {hist.seconds:.0f}s", flush=True)
    return asdict(hist)


@torch.no_grad()
def _predict(model: nn.Module, x: torch.Tensor, device: torch.device, bs: int = 1000):
    return torch.cat([model(x[i : i + bs].to(device)).cpu() for i in range(0, len(x), bs)])


# --------------------------------------------------------------------------- AE


def _fixed_random_amplitude_noise(x: torch.Tensor, max_amp: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    amp = torch.rand(len(x), 1, 1, 1, generator=g) * max_amp
    return (x + amp * (torch.rand(x.shape, generator=g) - 0.5)).clamp(0.0, 1.0)


def run_ae(splits: dict[str, Split], cfg: Config, device: torch.device, retrain: bool) -> dict:
    tr, va, te = splits["train"], splits["val"], splits["test"]
    ctr, cva = tr.x.to(device), va.x.to(device)
    kw = {"epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr, "seed": cfg.seed}
    models: dict[str, AE] = {}
    histories: dict[str, dict] = {}

    ref_tr = add_uniform_noise(tr.x, cfg.ref_amplitude, seed=1).to(device)
    ref_va = add_uniform_noise(va.x, cfg.ref_amplitude, seed=2).to(device)
    for name, head in (("ae_original", "legacy"), ("ae_fixed_head", "sigmoid")):
        torch.manual_seed(cfg.seed)
        m = models[name] = AE(head=head).to(device)
        histories[name] = _cached(
            name, m, lambda m=m: train_ae(m, ctr, ref_va, cva, x_in=ref_tr, **kw), retrain
        )

    blind_va = _fixed_random_amplitude_noise(va.x, cfg.max_amplitude, seed=3).to(device)

    def corrupt(x: torch.Tensor) -> torch.Tensor:
        return random_amplitude_noise(x, cfg.max_amplitude)

    for name, channels in (("ae_blind_narrow", None), ("ae_blind_wide", cfg.wide_channels)):
        torch.manual_seed(cfg.seed)
        m = AE(head="sigmoid", **({"channels": channels} if channels else {})).to(device)
        models[name] = m
        histories[name] = _cached(
            name, m, lambda m=m: train_ae(m, ctr, blind_va, cva, corrupt=corrupt, **kw), retrain
        )

    # Every model and baseline sees the very same corrupted test images at each level.
    curve: dict[str, list[float]] = {k: [] for k in ["identity", "median_3x3", *models]}
    at_ref: dict[str, dict] = {}
    examples: dict[str, np.ndarray] = {}
    for i, amp in enumerate(cfg.eval_amplitudes):
        noisy = add_uniform_noise(te.x, amp, seed=100 + i)
        outs = {"identity": noisy, "median_3x3": M.median_filter(noisy)}
        outs |= {k: _predict(m, noisy, device) for k, m in models.items()}
        errs = {k: M.per_image_mse(v, te.x) for k, v in outs.items()}
        for k, e in errs.items():
            curve[k].append(float(e.mean()))
        if amp == cfg.ref_amplitude:
            for k, e in errs.items():
                mean, lo, hi = M.paired_bootstrap_ci(e, errs["identity"])
                at_ref[k] = {
                    "mse": float(e.mean()),
                    "psnr_db": M.psnr(e),
                    "mse_minus_identity": mean,
                    "ci95": [lo, hi],
                }
        if amp == 1.0:
            examples = {k: v[:8, 0].numpy() for k, v in outs.items()} | {
                "clean": te.x[:8, 0].numpy()
            }

    return {
        "amplitudes": list(cfg.eval_amplitudes),
        "test_mse": curve,
        "at_ref_amplitude": at_ref,
        "params": {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()},
        "bottleneck_dim": {
            k: int(m.encoder(te.x[:1].to(device)).numel()) for k, m in models.items()
        },
        "_histories": histories,
        "_examples": examples,
    }


# -------------------------------------------------------------------------- VAE


def _train_classifier(
    splits, cfg: Config, device, retrain: bool
) -> tuple[M.DigitClassifier, float]:
    torch.manual_seed(cfg.seed)
    clf = M.DigitClassifier().to(device)
    x, y = splits["train"].x.to(device), splits["train"].y.to(device)
    loss_fn = nn.CrossEntropyLoss()

    def fit() -> History:
        t0 = time.perf_counter()
        torch.manual_seed(cfg.seed)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
        gen = torch.Generator().manual_seed(cfg.seed)
        clf.train()
        for _ in range(3):
            for idx in torch.randperm(len(x), generator=gen).split(cfg.batch_size):
                idx = idx.to(device)
                loss = loss_fn(clf(x[idx]), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        clf.eval()
        return History(best_epoch=3, seconds=time.perf_counter() - t0)

    _cached("digit_classifier", clf, fit, retrain)
    clf.eval()
    te = splits["test"]
    acc = float((_predict(clf, te.x, device).argmax(1) == te.y).float().mean())
    return clf, acc


@torch.no_grad()
def _evaluate_vae(m: VAE, splits, cfg: Config, device, clf) -> tuple[dict, dict]:
    te, tr = splits["test"], splits["train"]
    torch.manual_seed(cfg.seed)
    xt = te.x.to(device)
    mu, logvar = (
        torch.cat(t)
        for t in zip(*[m.encode(xt[i : i + 1000]) for i in range(0, len(xt), 1000)], strict=True)
    )
    n_active, unit_var = M.active_units(mu.cpu())
    recon = _predict(lambda b: m.mean_image(m.decode(m.encode(b)[0])), te.x, device)
    out = {
        "latent_dim": m.latent_dim,
        "active_units": n_active,
        "unit_variance": unit_var.tolist(),
        "kl_nats": float(kl_divergence(mu, logvar).mean()),
        "recon_mse_at_mean": float(M.per_image_mse(recon, te.x).mean()),
    }
    if m.head == "sigmoid":

        def elbo_terms(s):
            logits, mu_, logvar_ = m(xt[s])
            return bernoulli_nll(xt[s], logits).mean(), kl_divergence(mu_, logvar_).mean()

        nll, kl = _batched_mean(elbo_terms, len(xt))
        iw = torch.cat(
            [iwae_bound(m, xt[i : i + 500], cfg.iwae_k).cpu() for i in range(0, len(xt), 500)]
        )
        out |= {
            "neg_elbo_nats": nll + kl,
            "recon_nll_nats": nll,
            f"neg_iwae{cfg.iwae_k}_nats": float(-iw.mean()),
        }

    # Linear probe on a sampled code z ~ q(z|x): the information the decoder actually receives.
    def codes(x: torch.Tensor) -> np.ndarray:
        x = x.to(device)
        zs = [m.reparameterize(*m.encode(x[i : i + 1000])) for i in range(0, len(x), 1000)]
        return torch.cat(zs).cpu().numpy()

    out["probe_accuracy"] = M.probe_accuracy(
        codes(tr.x[:10_000]), tr.y[:10_000].numpy(), codes(te.x), te.y.numpy()
    )

    z = torch.randn(cfg.n_samples, m.latent_dim, device=device)
    samples = torch.cat(
        [m.mean_image(m.decode(z[i : i + 1000])).cpu() for i in range(0, len(z), 1000)]
    )
    probs = torch.softmax(_predict(clf, samples.clamp(0, 1), device), dim=1)
    out["sample_score"] = M.classifier_score(probs)
    arrays = {"samples": samples[:64, 0].numpy(), "recon": recon[:8, 0].numpy()}
    return out, arrays


def run_vae(splits, cfg: Config, device, retrain: bool) -> dict:
    tr, va, te = splits["train"], splits["val"], splits["test"]
    ctr, cva = tr.x.to(device), va.x.to(device)
    kw = {"epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr, "seed": cfg.seed}
    clf, clf_acc = _train_classifier(splits, cfg, device, retrain)
    real_probs = torch.softmax(_predict(clf, te.x, device), dim=1)

    specs: list[tuple[str, dict, dict]] = [
        ("vae_original", {"head": "legacy"}, {"objective": "legacy", "beta": 1.0}),
    ]
    for b in cfg.betas:
        specs.append((f"vae_elbo_b{b:g}", {}, {"objective": "elbo", "beta": b}))
    specs.append(("vae_elbo_2d", {"latent_dim": 2}, {"objective": "elbo", "beta": 1.0}))

    results, histories, arrays = {}, {}, {}
    for name, model_kw, train_kw in specs:
        model_kw = {"latent_dim": cfg.latent_dim} | model_kw
        torch.manual_seed(cfg.seed)
        m = VAE(**model_kw).to(device)
        histories[name] = _cached(
            name, m, lambda m=m, t=train_kw: train_vae(m, ctr, cva, **t, **kw), retrain
        )
        results[name], arrays[name] = _evaluate_vae(m, splits, cfg, device, clf)
        results[name] |= {"beta": train_kw["beta"], "objective": train_kw["objective"]}
        if name == "vae_elbo_2d":
            arrays[name] |= _latent_map(m, te, device)
    return {
        "models": results,
        "classifier_test_accuracy": clf_acc,
        "real_test_score": M.classifier_score(real_probs),
        "_histories": histories,
        "_arrays": arrays,
    }


@torch.no_grad()
def _latent_map(m: VAE, te: Split, device, n: int = 15) -> dict[str, np.ndarray]:
    """Posterior means of test images, and decoded images on a grid of prior quantiles."""
    mu = torch.cat(
        [m.encode(te.x[i : i + 1000].to(device))[0].cpu() for i in range(0, len(te.x), 1000)]
    )
    q = torch.distributions.Normal(0, 1).icdf(torch.linspace(0.03, 0.97, n))
    zz = torch.stack(torch.meshgrid(q, q.flip(0), indexing="xy"), dim=-1).reshape(-1, 2)
    imgs = m.mean_image(m.decode(zz.to(device))).cpu()[:, 0]
    return {"mu": mu.numpy(), "labels": te.y.numpy(), "grid": imgs.numpy(), "grid_q": q.numpy()}


# -------------------------------------------------------------------------- run


def run_all(
    cfg: Config | None = None,
    device: str = "auto",
    retrain: bool = False,
    threads: int | None = None,
) -> dict:
    cfg = cfg or Config()
    dev = pick_device(device)
    if threads:
        torch.set_num_threads(threads)
    print(f"device: {dev}", flush=True)
    splits = load_splits(seed=cfg.seed)
    ae = run_ae(splits, cfg, dev, retrain)
    vae = run_vae(splits, cfg, dev, retrain)

    RESULTS_DIR.mkdir(exist_ok=True)
    public = {
        "config": asdict(cfg),
        "device": str(dev),
        "torch": torch.__version__,
        "ae": {k: v for k, v in ae.items() if not k.startswith("_")},
        "vae": {k: v for k, v in vae.items() if not k.startswith("_")},
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(public, indent=2))
    hist = {"ae": ae["_histories"], "vae": vae["_histories"]}
    (RESULTS_DIR / "history.json").write_text(json.dumps(hist, indent=1))

    flat = {f"ae_example_{k}": v for k, v in ae["_examples"].items()}
    for name, arr in vae["_arrays"].items():
        flat |= {f"{name}__{k}": v for k, v in arr.items()}
    ARRAYS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARRAYS_PATH, **flat)
    return public
