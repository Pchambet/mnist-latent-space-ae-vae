"""Static report (site/index.html) and the generated README tables.

Every number on the page and in the README tables is read from results/metrics.json,
so the documents cannot drift from what the pipeline measured.
"""

from __future__ import annotations

import base64
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from string import Template

import plotly.graph_objects as go

from .figures import (
    AMBER,
    FIG_DIR,
    LABELS,
    SLATE,
    TEAL,
    ratio_versus,
    relative_change,
    versus_identity,
)

RESULTS = Path("results/metrics.json")
HISTORY = Path("results/history.json")
SITE = Path("site")
README = Path("README.md")
REPO = "https://github.com/Pchambet/mnist-latent-space-ae-vae"
PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"

AE_ROWS = [
    "identity",
    "median_3x3",
    "ae_original",
    "ae_fixed_head",
    "ae_blind_narrow",
    "ae_blind_wide",
]
VAE_LABELS = {
    "vae_original": "original loss (pixel-mean MSE + KL)",
    "vae_elbo_b1": "Bernoulli ELBO, beta = 1",
    "vae_elbo_b4": "Bernoulli ELBO, beta = 4",
    "vae_elbo_b16": "Bernoulli ELBO, beta = 16",
    "vae_elbo_b64": "Bernoulli ELBO, beta = 64",
    "vae_elbo_2d": "Bernoulli ELBO, beta = 1, 2-D latent",
}


# --- tables (markdown for the README, HTML for the page) ------------------------------


def _ae_rows(m: dict) -> list[list[str]]:
    at_ref, dims, params = (
        m["ae"]["at_ref_amplitude"],
        m["ae"]["bottleneck_dim"],
        m["ae"]["params"],
    )
    rows = []
    for k in AE_ROWS:
        r = at_ref[k]
        diff = (
            "reference"
            if k == "identity"
            else f"{r['mse_minus_identity']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
        )
        rows.append(
            [
                LABELS[k],
                str(dims.get(k, "-")),
                f"{params[k]:,}" if k in params else "-",
                f"{r['mse']:.4f}",
                f"{r['psnr_db']:.1f}",
                diff,
            ]
        )
    return rows


AE_HEAD = [
    "Method",
    "Code dim",
    "Params",
    "Test MSE",
    "PSNR (dB)",
    "MSE minus do-nothing [95% CI]",
]


def _curve_rows(m: dict) -> tuple[list[str], list[list[str]]]:
    ae = m["ae"]
    head = ["Method"] + [f"a = {a:g}" for a in ae["amplitudes"]]
    rows = [[LABELS[k]] + [f"{v:.4f}" for v in ae["test_mse"][k]] for k in AE_ROWS]
    return head, rows


def _vae_rows(m: dict) -> list[list[str]]:
    k_iw = f"neg_iwae{m['config']['iwae_k']}_nats"
    rows = []
    for k, label in VAE_LABELS.items():
        r = m["vae"]["models"][k]
        rows.append(
            [
                label,
                f"{r['active_units']}/{r['latent_dim']}",
                f"{r['kl_nats']:.2f}",
                f"{r['neg_elbo_nats']:.1f}" if "neg_elbo_nats" in r else "n/a",
                f"{r[k_iw]:.1f}" if k_iw in r else "n/a",
                f"{r['recon_mse_at_mean']:.4f}",
                f"{r['probe_accuracy']:.1%}",
                f"{r['sample_score']['score']:.2f}",
            ]
        )
    return rows


def _vae_head(m: dict) -> list[str]:
    return [
        "Objective",
        "Active units",
        "KL (nats)",
        "-ELBO (nats)*",
        f"-IWAE{m['config']['iwae_k']} (nats)*",
        "Recon. MSE",
        "Probe acc.",
        "Sample score",
    ]


def _md(head: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(head) + " |",
        "|" + "|".join(["---"] + ["--:"] * (len(head) - 1)) + "|",
    ]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def _html(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def readme_blocks(m: dict) -> dict[str, str]:
    return {
        "ae-table": _md(AE_HEAD, _ae_rows(m)),
        "ae-curve": _md(*_curve_rows(m)),
        "vae-table": _md(_vae_head(m), _vae_rows(m)),
    }


def replace_blocks(text: str, blocks: dict[str, str]) -> str:
    """Replace the body between ``<!-- BEGIN:key -->`` and ``<!-- END:key -->`` markers."""
    for key, body in blocks.items():
        # The body may be empty (a fresh README), so the newline before END is optional.
        pattern = re.compile(rf"(<!-- BEGIN:{key} -->\n)(?:.*?\n)?(<!-- END:{key} -->)", re.DOTALL)
        text = pattern.sub(lambda mt, b=body: mt.group(1) + b + "\n" + mt.group(2), text)
    return text


# --- interactive charts ---------------------------------------------------------------


def _fig_json(fig: go.Figure) -> dict:
    """Plotly JSON without the default template: the page applies its own light/dark theme."""
    out = fig.to_plotly_json()
    out["layout"].pop("template", None)
    return out


def _layout(title: str, xtitle: str, ytitle: str, **kw) -> dict:
    return {
        "title": {"text": title, "x": 0, "font": {"size": 14}},
        "xaxis": {"title": {"text": xtitle}, "zeroline": False},
        "yaxis": {"title": {"text": ytitle}, "zeroline": False, "rangemode": "tozero"},
        "margin": {"l": 60, "r": 20, "t": 50, "b": 55},
        "legend": {"orientation": "h", "y": -0.25},
        "hovermode": "x unified",
        **kw,
    }


def chart_denoising(m: dict) -> dict:
    ae = m["ae"]
    style = {
        "identity": (SLATE, "dash"),
        "median_3x3": (SLATE, "dot"),
        "ae_original": (AMBER, "dashdot"),
        "ae_fixed_head": (TEAL, "dashdot"),
        "ae_blind_narrow": (AMBER, "solid"),
        "ae_blind_wide": (TEAL, "solid"),
    }
    fig = go.Figure()
    for k, (color, dash) in style.items():
        fig.add_scatter(
            x=ae["amplitudes"],
            y=ae["test_mse"][k],
            name=LABELS[k],
            mode="lines+markers",
            line={"color": color, "dash": dash, "width": 2},
            hovertemplate="%{y:.4f}",
        )
    fig.update_layout(
        _layout("Test MSE by noise amplitude", "noise amplitude a", "test MSE per pixel")
    )
    return _fig_json(fig)


def chart_kl_curves(h: dict) -> dict:
    fig = go.Figure()
    for k, color in (("vae_original", AMBER), ("vae_elbo_b1", TEAL), ("vae_elbo_b16", SLATE)):
        kl = h["vae"][k]["val_kl"]
        fig.add_scatter(
            x=list(range(1, len(kl) + 1)),
            y=kl,
            name=VAE_LABELS[k],
            mode="lines+markers",
            line={"color": color, "width": 2},
            hovertemplate="%{y:.2f} nats",
        )
    fig.update_layout(
        _layout("Validation KL per epoch (nats per image)", "epoch", "KL(q(z|x) || p(z))")
    )
    return _fig_json(fig)


def chart_beta(m: dict) -> dict:
    vae = m["vae"]["models"]
    keys = sorted((k for k in vae if k.startswith("vae_elbo_b")), key=lambda k: vae[k]["beta"])
    betas = [vae[k]["beta"] for k in keys]
    fig = go.Figure()
    fig.add_scatter(
        x=betas,
        y=[vae[k]["active_units"] for k in keys],
        name="active units (of 16)",
        mode="lines+markers",
        line={"color": TEAL, "width": 2},
    )
    fig.add_scatter(
        x=betas,
        y=[vae[k]["sample_score"]["score"] for k in keys],
        name="sample classifier score (1-10)",
        mode="lines+markers",
        line={"color": AMBER, "width": 2},
        yaxis="y2",
    )
    layout = _layout("Turning up the KL weight", "beta (log scale)", "active units")
    layout["xaxis"] |= {"type": "log", "tickvals": betas, "ticktext": [f"{b:g}" for b in betas]}
    layout["yaxis2"] = {
        "title": {"text": "sample score"},
        "overlaying": "y",
        "side": "right",
        "rangemode": "tozero",
        "showgrid": False,
    }
    fig.update_layout(layout)
    return _fig_json(fig)


# --- page -------------------------------------------------------------------------------


def _img(name: str, alt: str) -> str:
    data = base64.b64encode((FIG_DIR / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{alt}">'


TEMPLATE = Path(__file__).parent / "report_template.html"


def build() -> str:
    m = json.loads(RESULTS.read_text())
    h = json.loads(HISTORY.read_text())
    ae, vae = m["ae"], m["vae"]["models"]
    at_ref = ae["at_ref_amplitude"]
    leg, fix = vae["vae_original"], vae["vae_elbo_b1"]
    k_iw = f"neg_iwae{m['config']['iwae_k']}_nats"
    betas = sorted((k for k in vae if k.startswith("vae_elbo_b")), key=lambda k: vae[k]["beta"])
    b_hi = vae[betas[-1]]
    curve_head, curve_rows = _curve_rows(m)
    dims = ae["bottleneck_dim"]
    floors_16 = [ae["test_mse"][k][0] for k in dims if dims[k] == 16]
    values = {
        "plotly": PLOTLY,
        "repo": REPO,
        "epochs": m["config"]["epochs"],
        "device": m["device"],
        "ref_amp": f"{m['config']['ref_amplitude']:g}",
        "max_amp": f"{m['config']['max_amplitude']:g}",
        "ident_mse": f"{at_ref['identity']['mse']:.4f}",
        "median_mse": f"{at_ref['median_3x3']['mse']:.4f}",
        "orig_mse": f"{at_ref['ae_original']['mse']:.4f}",
        "orig_diff": f"{at_ref['ae_original']['mse_minus_identity']:+.4f}",
        "orig_ci": "[{:+.4f}, {:+.4f}]".format(*at_ref["ae_original"]["ci95"]),
        "orig_vs_identity": versus_identity(
            at_ref["ae_original"]["mse_minus_identity"],
            at_ref["ae_original"]["ci95"],
            at_ref["identity"]["mse"],
        ),
        "head_mse": f"{at_ref['ae_fixed_head']['mse']:.4f}",
        "head_gain": relative_change(at_ref["ae_fixed_head"]["mse"], at_ref["ae_original"]["mse"]),
        "head_vs_median": "which still loses to"
        if at_ref["ae_fixed_head"]["mse"] > at_ref["median_3x3"]["mse"]
        else "which beats",
        "wide_mse": f"{at_ref['ae_blind_wide']['mse']:.4f}",
        "wide_vs_median": ratio_versus(at_ref["ae_blind_wide"]["mse"], at_ref["median_3x3"]["mse"]),
        "floor_16": f"{min(floors_16):.4f} to {max(floors_16):.4f}",
        "floor_narrow": f"{ae['test_mse']['ae_blind_narrow'][0]:.4f}",
        "floor_wide": f"{ae['test_mse']['ae_blind_wide'][0]:.4f}",
        "params_narrow": f"{ae['params']['ae_blind_narrow']:,}",
        "params_wide": f"{ae['params']['ae_blind_wide']:,}",
        "leg_au": leg["active_units"],
        "fix_au": fix["active_units"],
        "latent_dim": fix["latent_dim"],
        "leg_kl": f"{leg['kl_nats']:.3f}",
        "fix_kl": f"{fix['kl_nats']:.1f}",
        "fix_elbo": f"{fix['neg_elbo_nats']:.1f}",
        "fix_iwae": f"{fix[k_iw]:.1f}",
        "iwae_k": m["config"]["iwae_k"],
        "leg_probe": f"{leg['probe_accuracy']:.0%}",
        "fix_probe": f"{fix['probe_accuracy']:.0%}",
        "leg_score": f"{leg['sample_score']['score']:.2f}",
        "fix_score": f"{fix['sample_score']['score']:.2f}",
        "real_score": f"{m['vae']['real_test_score']['score']:.2f}",
        "clf_acc": f"{m['vae']['classifier_test_accuracy']:.1%}",
        "beta_hi": f"{b_hi['beta']:g}",
        "beta_hi_au": b_hi["active_units"],
        "beta_hi_score": f"{b_hi['sample_score']['score']:.2f}",
        "probe_2d": f"{vae['vae_elbo_2d']['probe_accuracy']:.0%}",
        "ae_table": _html(AE_HEAD, _ae_rows(m)),
        "curve_table": _html(curve_head, curve_rows),
        "vae_table": _html(_vae_head(m), _vae_rows(m)),
        "img_hero": _img("hero.png", "Denoising curve and VAE samples from the prior"),
        "img_examples": _img("denoising_examples.png", "Denoising examples at a = 1"),
        "img_recon": _img("vae_reconstructions.png", "VAE reconstructions"),
        "img_latent": _img("latent_2d.png", "Two-dimensional latent space"),
        "c_denoise": json.dumps(chart_denoising(m), default=str),
        "c_kl": json.dumps(chart_kl_curves(h), default=str),
        "c_beta": json.dumps(chart_beta(m), default=str),
        "generated": datetime.now(UTC).strftime("%Y-%m-%d"),
    }
    SITE.mkdir(exist_ok=True)
    out = SITE / "index.html"
    out.write_text(Template(TEMPLATE.read_text()).substitute(values))
    if README.exists():
        README.write_text(replace_blocks(README.read_text(), readme_blocks(m)))
    print(f"report written to {out}; README tables refreshed")
    return str(out)
