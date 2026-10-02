"""Static figures for the README (matplotlib, PNG at 200 dpi in docs/figures/).

Titles state the finding, so they are built from results/metrics.json rather than
written by hand: if a rerun changes the direction of a result, the title follows.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("docs/figures")
RESULTS = Path("results/metrics.json")
ARRAYS = Path("data/interim/arrays.npz")

INK, TEAL, AMBER, SLATE, GRID = "#0f172a", "#0d9488", "#d97706", "#64748b", "#e2e8f0"
LABELS = {
    "identity": "do nothing (noisy input)",
    "median_3x3": "3x3 median filter",
    "ae_original": "original AE (BatchNorm output, retrained)",
    "ae_fixed_head": "original AE, sigmoid output",
    "ae_blind_narrow": "narrow AE (16-d code)",
    "ae_blind_wide": "wide AE (128-d code)",
}


def first_win(model: list[float], ref: list[float], amps: list[float]) -> float | None:
    """Smallest amplitude from which ``model`` has a lower error than ``ref`` at every
    larger amplitude too; None if it never does at the largest one."""
    win = None
    for a, m, r in zip(reversed(amps), reversed(model), reversed(ref), strict=True):
        if m >= r:
            break
        win = a
    return win


def versus_identity(diff: float, ci: list[float], identity: float) -> str:
    """How a method compares with doing nothing, given its paired MSE difference and CI."""
    if ci[0] <= 0.0 <= ci[1]:
        return "statistically indistinguishable from doing nothing"
    side = "worse" if diff > 0 else "better"
    return (
        f"marginally {side} than doing nothing"
        if abs(diff) < 0.1 * identity
        else (f"{side} than doing nothing")
    )


def relative_change(new: float, old: float) -> str:
    """'7% lower' / '3% higher' for an error going from ``old`` to ``new``."""
    return f"{abs(1 - new / old):.0%} {'lower' if new < old else 'higher'}"


def ratio_versus(model: float, baseline: float) -> str:
    """'2.3x lower error than' or 'worse than', for a model against a baseline."""
    return f"{baseline / model:.1f}× lower than" if model < baseline else "worse than"


def _median_title(ae: dict, ref_amp: float) -> str:
    """Which autoencoders beat the 3x3 median filter at the reference noise level."""
    at = ae["at_ref_amplitude"]
    aes = [k for k in LABELS if k.startswith("ae_")]
    beats = [k for k in aes if at[k]["mse"] < at["median_3x3"]["mse"]]
    if beats == ["ae_blind_wide"]:
        return f"At a = {ref_amp:g}, only the wide AE (128-d code) beats the median filter"
    return f"At a = {ref_amp:g}, {len(beats)} of {len(aes)} AEs beat the median filter"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": SLATE,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def _tile(images: np.ndarray, rows: int, cols: int, pad: int = 2) -> np.ndarray:
    """Arrange (N, 28, 28) images into one mosaic; the gutter is background (white once
    ``_show`` renders ink on paper)."""
    h, w = images.shape[1:]
    out = np.zeros((rows * (h + pad) - pad, cols * (w + pad) - pad))
    for i in range(min(rows * cols, len(images))):
        r, c = divmod(i, cols)
        out[r * (h + pad) : r * (h + pad) + h, c * (w + pad) : c * (w + pad) + w] = images[i]
    return out


def _show(ax, mosaic: np.ndarray, title: str) -> None:
    # Ink-on-paper rendering: digits dark on white, like the page around them.
    ax.imshow(1.0 - np.clip(mosaic, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _denoising_axes(ax, ae: dict, ref_amp: float) -> None:
    amps = np.array(ae["amplitudes"])
    styles = {
        "identity": (SLATE, "--", None),
        "median_3x3": (SLATE, ":", None),
        "ae_blind_narrow": (AMBER, "-", "o"),
        "ae_blind_wide": (TEAL, "-", "o"),
    }
    for key, (color, ls, marker) in styles.items():
        y = np.array(ae["test_mse"][key])
        ax.plot(amps, y, color=color, ls=ls, marker=marker, ms=3.5, lw=1.8)
        ax.annotate(
            LABELS[key],
            (amps[-1], y[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            va="center",
            fontsize=8.5,
        )
    at = ae["at_ref_amplitude"]
    orig, head = at["ae_original"], at["ae_fixed_head"]
    ax.scatter([ref_amp], [orig["mse"]], color=AMBER, marker="X", s=60, zorder=5)
    ax.scatter([ref_amp], [head["mse"]], color=TEAL, marker="X", s=60, zorder=5)
    lo, hi = orig["ci95"]
    ax.annotate(
        f"a = {ref_amp:g}, 16-d code, trained at this level only:\n"
        f"original AE {orig['mse']:.4f} ({orig['mse_minus_identity']:+.4f} vs doing nothing,\n"
        f"95% CI [{lo:+.4f}, {hi:+.4f}]); sigmoid output {head['mse']:.4f}",
        (ref_amp, orig["mse"]),
        xytext=(0.04, 0.6),
        textcoords="axes fraction",
        fontsize=8,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": SLATE, "lw": 0.8},
    )
    ax.set_xlabel("noise amplitude a  (noise ~ U(-a/2, a/2), clipped to [0, 1])")
    ax.set_ylabel("test MSE per pixel")
    ax.set_xlim(0, amps[-1] * 1.42)
    ax.set_ylim(0, None)


def hero(m: dict, arr) -> None:
    ae, vae = m["ae"], m["vae"]["models"]
    leg, fix = vae["vae_original"], vae["vae_elbo_b1"]
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1, 1], wspace=0.18)
    ax = fig.add_subplot(gs[0])
    _denoising_axes(ax, ae, m["config"]["ref_amplitude"])
    ax.set_title(_median_title(ae, m["config"]["ref_amplitude"]))
    for i, (key, res, title) in enumerate(
        [
            ("vae_original", leg, "VAE, original loss"),
            ("vae_elbo_b1", fix, "VAE, ELBO (beta = 1)"),
        ]
    ):
        a = fig.add_subplot(gs[i + 1])
        _show(a, _tile(arr[f"{key}__samples"], 6, 6), "")
        a.set_title(
            f"{title}: prior samples\n"
            f"{res['active_units']}/{res['latent_dim']} latent units used\n"
            f"classifier score {res['sample_score']['score']:.1f}/10",
            fontsize=10,
            loc="left",
        )
    at = ae["at_ref_amplitude"]
    ref = m["config"]["ref_amplitude"]
    orig = at["ae_original"]
    median_side = "loses to" if orig["mse"] > at["median_3x3"]["mse"] else "beats"
    fig.suptitle(
        f"At a = {ref:g}, the original 16-d-code denoiser is "
        f"{versus_identity(orig['mse_minus_identity'], orig['ci95'], at['identity']['mse'])} "
        f"and {median_side} a 3x3 median filter;\n"
        f"the original VAE loss leaves {leg['active_units']} of {leg['latent_dim']} "
        "latent units in use",
        x=0.08,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
        y=1.04,
    )
    fig.savefig(FIG_DIR / "hero.png")
    plt.close(fig)


def denoising_curve(m: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    _denoising_axes(ax, m["ae"], m["config"]["ref_amplitude"])
    ae = m["ae"]
    amps = np.array(ae["amplitudes"])
    narrow = ae["test_mse"]["ae_blind_narrow"]
    cross = first_win(narrow, ae["test_mse"]["identity"], list(amps))
    wide = ae["test_mse"]["ae_blind_wide"]
    title = (
        f"On clean inputs the 16-d code has an error floor of {narrow[0]:.4f} "
        f"(128-d code: {wide[0]:.4f})"
    )
    if cross is not None:
        title += f";\nthe narrow AE beats doing nothing only from a = {cross:g}"
    ax.set_title(title)
    fig.savefig(FIG_DIR / "denoising_curve.png")
    plt.close(fig)


def denoising_examples(m: dict, arr) -> None:
    rows = [
        ("clean", "clean target"),
        ("identity", "noisy input (a = 1)"),
        ("median_3x3", LABELS["median_3x3"]),
        ("ae_blind_narrow", LABELS["ae_blind_narrow"]),
        ("ae_blind_wide", LABELS["ae_blind_wide"]),
    ]
    fig, axes = plt.subplots(len(rows), 1, figsize=(7.2, 6.4))
    for ax, (key, label) in zip(axes, rows, strict=True):
        _show(ax, _tile(arr[f"ae_example_{key}"], 1, 8), "")
        ax.text(-4, 14, label, ha="right", va="center", fontsize=9.5, color=INK)
    ae = m["ae"]
    i = ae["amplitudes"].index(1.0)
    err = {k: ae["test_mse"][k][i] for k in ("identity", "median_3x3", *_BLIND)}
    fig.suptitle(
        "Same test digits at a = 1: test MSE "
        + ", ".join(f"{err[k]:.4f} ({_SHORT[k]})" for k in err),
        fontsize=11,
        fontweight="bold",
        color=INK,
        x=0.02,
        ha="left",
    )
    fig.savefig(FIG_DIR / "denoising_examples.png")
    plt.close(fig)


_BLIND = ("ae_blind_narrow", "ae_blind_wide")
_SHORT = {
    "identity": "noisy input",
    "median_3x3": "median",
    "ae_blind_narrow": "narrow AE",
    "ae_blind_wide": "wide AE",
}


def beta_sweep(m: dict) -> None:
    vae = m["vae"]["models"]
    keys = sorted((k for k in vae if k.startswith("vae_elbo_b")), key=lambda k: vae[k]["beta"])
    betas = np.array([vae[k]["beta"] for k in keys])
    leg = vae["vae_original"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.3})
    for ax, metric, label in (
        (a1, lambda r: r["active_units"], "active latent units (of 16)"),
        (a2, lambda r: r["sample_score"]["score"], "classifier score of samples (1-10)"),
    ):
        y = np.array([metric(vae[k]) for k in keys])
        ax.plot(betas, y, color=TEAL, marker="o", lw=1.8)
        for b, v in zip(betas, y, strict=True):
            fmt = f"{v:.0f}" if ax is a1 else f"{v:.1f}"
            ax.annotate(
                fmt,
                (b, v),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=TEAL,
            )
        ax.axhline(metric(leg), color=AMBER, ls="--", lw=1.4)
        ax.annotate(
            f"original loss: {metric(leg):.0f}"
            if ax is a1
            else f"original loss: {metric(leg):.2f}",
            (betas[0], metric(leg)),
            xytext=(0, 6),
            textcoords="offset points",
            color=AMBER,
            fontsize=8.5,
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(betas, [f"{b:g}" for b in betas])
        ax.set_xlabel("KL weight beta (log scale; 1 = unweighted ELBO)")
        ax.set_ylabel(label)
        ax.set_ylim(0, None)
    a2.axhline(m["vae"]["real_test_score"]["score"], color=SLATE, ls=":", lw=1.2)
    a2.annotate(
        f"real test digits: {m['vae']['real_test_score']['score']:.1f}",
        (betas[-1], m["vae"]["real_test_score"]["score"]),
        xytext=(0, -12),
        textcoords="offset points",
        ha="right",
        color=SLATE,
        fontsize=8.5,
    )
    lo, hi = vae[keys[0]], vae[keys[-1]]
    a1.set_title(
        f"Raising beta from {betas[0]:g} to {betas[-1]:g}: "
        f"{lo['active_units']} -> {hi['active_units']} of {lo['latent_dim']} units active"
    )
    a2.set_title(
        f"Sample score {lo['sample_score']['score']:.1f} -> {hi['sample_score']['score']:.1f} "
        f"(original loss: {leg['sample_score']['score']:.1f})"
    )
    fig.savefig(FIG_DIR / "vae_beta_sweep.png")
    plt.close(fig)


def reconstructions(m: dict, arr) -> None:
    leg, fix = m["vae"]["models"]["vae_original"], m["vae"]["models"]["vae_elbo_b1"]
    rows = [
        (arr["ae_example_clean"], "test digit"),
        (arr["vae_original__recon"], "VAE, original loss"),
        (arr["vae_elbo_b1__recon"], "VAE, ELBO (beta = 1)"),
    ]
    fig, axes = plt.subplots(len(rows), 1, figsize=(7.2, 3.9))
    for ax, (imgs, label) in zip(axes, rows, strict=True):
        _show(ax, _tile(imgs, 1, 8), "")
        ax.text(-4, 14, label, ha="right", va="center", fontsize=9.5, color=INK)
    fig.suptitle(
        f"Reconstructions from the posterior mean: test MSE {leg['recon_mse_at_mean']:.4f} "
        f"with {leg['active_units']}/{leg['latent_dim']} units active (original loss)\n"
        f"vs {fix['recon_mse_at_mean']:.4f} with {fix['active_units']}/{fix['latent_dim']} "
        "(ELBO, beta = 1)",
        fontsize=11,
        fontweight="bold",
        color=INK,
        x=0.02,
        ha="left",
    )
    fig.savefig(FIG_DIR / "vae_reconstructions.png")
    plt.close(fig)


def latent_map(m: dict, arr) -> None:
    mu, labels = arr["vae_elbo_2d__mu"], arr["vae_elbo_2d__labels"]
    grid, q = arr["vae_elbo_2d__grid"], arr["vae_elbo_2d__grid_q"]
    n = len(q)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.4), gridspec_kw={"wspace": 0.08})
    a1.scatter(mu[:, 0], mu[:, 1], c=labels, cmap="tab10", s=2.5, alpha=0.55, linewidths=0)
    for d in range(10):
        cx, cy = np.median(mu[labels == d], axis=0)
        a1.text(
            cx,
            cy,
            str(d),
            fontsize=13,
            fontweight="bold",
            ha="center",
            va="center",
            color=INK,
            bbox={
                "boxstyle": "circle,pad=0.2",
                "fc": "white",
                "ec": SLATE,
                "lw": 0.6,
                "alpha": 0.85,
            },
        )
    a1.set_xlabel("z1 (posterior mean)")
    a1.set_ylabel("z2 (posterior mean)")
    a1.set_title("10,000 test digits encoded in a 2-D latent space")
    a1.set_aspect("equal", adjustable="datalim")
    _show(a2, _tile(grid, n, n, pad=0), "")
    a2.set_title(
        "Decoder output on a grid of prior quantiles (3% to 97%)",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    res = m["vae"]["models"]["vae_elbo_2d"]
    fig.suptitle(
        f"A two-dimensional latent space: a linear probe reads the digit from a sampled code "
        f"with {res['probe_accuracy']:.0%} test accuracy",
        fontsize=12,
        fontweight="bold",
        color=INK,
        x=0.06,
        ha="left",
        y=1.0,
    )
    fig.savefig(FIG_DIR / "latent_2d.png")
    plt.close(fig)


def draw_all() -> None:
    _style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    m = json.loads(RESULTS.read_text())
    arr = np.load(ARRAYS)
    hero(m, arr)
    denoising_curve(m)
    denoising_examples(m, arr)
    beta_sweep(m)
    reconstructions(m, arr)
    latent_map(m, arr)
    print(f"figures written to {FIG_DIR}/")
