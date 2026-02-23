"""
plot_distance_comparison.py
===========================
Compare SAFT distances (thermodynamic + structural) against two naive
group-count metrics:

  * **Euclidean** — D_vec = sqrt( sum_k (n_k,c - n_k,t)^2 )
  * **Cosine**    — D_cos = 1 - (a . b) / (|a| |b|)

All distances are normalised to [0, 1] (divide by max) so that the
*relative ordering* of molecules can be compared directly.

Figures produced (per naive metric)
------------------------------------
1. 2-panel scatter: D_struct vs D_naive  |  D_SAFT vs D_naive
2. 2-panel residual histograms
3. 2-panel rank-rank correlation

Usage
-----
    python plot_distance_comparison.py [path/to/ranking_vs_NCCO.json]
"""

from __future__ import annotations
import json, sys, os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


# ═════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════

def _cosine_distance(a, b):
    """Return 1 - cos(a, b).  Returns 1.0 when either vector is zero."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return 1.0 - np.dot(a, b) / denom


def load_ranking(path: str):
    """Load ranking JSON and return structured arrays."""
    with open(path) as f:
        data = json.load(f)
    target_sig = data["target_signature"]
    target_vec = data["target_vector"]
    entries = data["ranking"]
    N = len(entries)

    d_saft   = np.array([e["distance_thermo"] for e in entries])
    d_struct = np.array([e["distance_struct"]  for e in entries])
    d_vec    = np.array([e["distance_vector"]  for e in entries])
    d_cos    = np.array([_cosine_distance(target_vec, e["candidate_vector"])
                         for e in entries])
    names    = [e["compound"] for e in entries]
    dFa      = np.array([e["signature"]["F_assoc"] - target_sig["F_assoc"]
                         for e in entries])

    return {
        "d_saft": d_saft, "d_struct": d_struct,
        "d_vec": d_vec, "d_cos": d_cos,
        "names": names, "dFa": dFa, "N": N,
        "target": data["target"],
    }


def norm01(arr):
    """Normalise array to [0, 1] by dividing by max."""
    mx = arr.max()
    return arr / mx if mx > 0 else arr


# ═════════════════════════════════════════════════════════════════════
# Generic plot functions (work for any naive metric)
# ═════════════════════════════════════════════════════════════════════

DOT_COLOR = "#5B7FA5"
MARKER_COLORS = ["#2ECC71", "#E74C3C", "#3498DB", "#F39C12", "#9B59B6"]


def fig_scatter(stn, sn, dn, names, target, metric_label, rho_struct, rho_saft, path):
    """2-panel scatter: D_struct vs D_naive  |  D_SAFT vs D_naive."""

    # 5 molecules closest to identity in each panel
    on_a = np.argsort(np.abs(stn - dn))[:5]
    on_b = np.argsort(np.abs(sn  - dn))[:5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    for ax, yn, rho, on_line, ylabel, panel_label in [
        (ax1, stn, rho_struct, on_a,
         "Normalised $D_{\\mathrm{struct}}$  (log-Euclidean: m, σ³, S̄)",
         "(a) Structural"),
        (ax2, sn, rho_saft, on_b,
         "Normalised $D_{\\mathrm{SAFT}}$  (thermodynamic Euclidean)",
         "(b) Thermodynamic"),
    ]:
        ax.plot([0, 1], [0, 1], "--", color="#444", lw=1.2, zorder=2)
        ax.scatter(dn, yn, c=DOT_COLOR, s=16, alpha=0.55,
                   edgecolors="none", zorder=3)

        leg = []
        for k, idx in enumerate(on_line):
            ax.scatter(dn[idx], yn[idx], s=80, facecolors="none",
                       edgecolors=MARKER_COLORS[k], linewidths=2, zorder=6)
            leg.append(plt.Line2D(
                [0], [0], marker="o", color="w",
                markeredgecolor=MARKER_COLORS[k], markeredgewidth=2,
                markersize=8, markerfacecolor="none", label=names[idx]))

        ax.set_xlabel(f"Normalised $D_{{\\mathrm{{{metric_label}}}}}$", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{panel_label} vs {metric_label} distance\n"
                     f"Spearman ρ = {rho:.3f}",
                     fontsize=13, fontweight="bold")
        ax.set_xlim(-0.02, 1.04)
        ax.set_ylim(-0.02, 1.04)
        ax.set_aspect("equal")
        ax.legend(handles=leg, loc="upper left", fontsize=8,
                  title="On identity line", title_fontsize=9, framealpha=0.9)

    fig.suptitle(f"SAFT distances vs {metric_label} distance  "
                 f"(target: {target}, normalised to [0, 1])",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_residuals(stn, sn, dn, metric_label, rho_struct, rho_saft, path):
    """2-panel residual histograms."""
    res_struct = stn - dn
    res_saft   = sn  - dn
    bins = np.linspace(-0.8, 0.8, 45)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for ax, res, color, label in [
        (ax1, res_struct, "#27AE60", "D_struct"),
        (ax2, res_saft,   "#EB5757", "D_SAFT"),
    ]:
        mae = np.mean(np.abs(res))
        sigma = np.std(res)
        med = np.median(res)

        ax.hist(res, bins=bins, color=color, alpha=0.7,
                edgecolor="white", lw=0.5)
        ax.axvline(0, color="#666", lw=1, ls="--")
        ax.axvline(med, color=color, lw=2,
                   label=f"median = {med:+.3f}")
        ax.set_xlabel(
            f"${label.replace('_', chr(95))}^{{\\mathrm{{norm}}}} "
            f"- D_{{\\mathrm{{{metric_label}}}}}^{{\\mathrm{{norm}}}}$",
            fontsize=12)
        ax.set_title(f"{label} − D_{metric_label}\n"
                     f"MAE = {mae:.3f},  σ = {sigma:.3f}",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    ax1.set_ylabel("Count", fontsize=11)

    fig.suptitle(f"Deviation from {metric_label} distance  (normalised)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_rank_rank(d_struct, d_saft, d_naive, N, metric_label,
                  rho_struct, rho_saft, target, path):
    """2-panel rank-rank scatter."""
    rank_struct = np.argsort(np.argsort(d_struct)) + 1
    rank_saft   = np.argsort(np.argsort(d_saft))   + 1
    rank_naive  = np.argsort(np.argsort(d_naive))   + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    for ax, rank_y, rho, ylabel, panel in [
        (ax1, rank_struct, rho_struct,
         "Rank by $D_{\\mathrm{struct}}$", "(a) Structural"),
        (ax2, rank_saft, rho_saft,
         "Rank by $D_{\\mathrm{SAFT}}$", "(b) Thermodynamic"),
    ]:
        ax.plot([1, N], [1, N], "--", color="#888", lw=1, zorder=1)
        ax.scatter(rank_naive, rank_y, c=DOT_COLOR, s=14, alpha=0.6,
                   edgecolors="none", zorder=3)
        ax.set_xlabel(f"Rank by $D_{{\\mathrm{{{metric_label}}}}}$",
                      fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{panel} vs {metric_label} rank\n"
                     f"Spearman ρ = {rho:.3f}",
                     fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlim(0, N + 5)
        ax.set_ylim(0, N + 5)

    fig.suptitle(f"Rank correlation with {metric_label} distance  "
                 f"(target: {target})",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════
# Euclidean vs Cosine comparison
# ═════════════════════════════════════════════════════════════════════

def fig_euclidean_vs_cosine(d_vec, d_cos, d_struct, d_saft, names, target, out_dir):
    """4-panel figure comparing Euclidean and Cosine naive distances.

    (a) Scatter D_vec^norm vs D_cos^norm
    (b) Histogram of D_vec^norm - D_cos^norm
    (c) Scatter D_SAFT vs diff(vec, cos)
    (d) Scatter D_struct vs diff(vec, cos)
    """
    vn = norm01(d_vec)
    cn = norm01(d_cos)
    diff = vn - cn

    rho_vc, _ = spearmanr(d_vec, d_cos)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    (ax1, ax2), (ax3, ax4) = axes

    # ── (a) scatter: normalised Euclidean vs Cosine ──
    ax1.plot([0, 1], [0, 1], "--", color="#444", lw=1.2, zorder=2)
    ax1.scatter(cn, vn, c=DOT_COLOR, s=16, alpha=0.55,
                edgecolors="none", zorder=3)
    # highlight 5 largest absolute differences
    top5 = np.argsort(np.abs(diff))[-5:]
    leg = []
    for k, idx in enumerate(top5):
        ax1.scatter(cn[idx], vn[idx], s=80, facecolors="none",
                    edgecolors=MARKER_COLORS[k], linewidths=2, zorder=6)
        leg.append(plt.Line2D(
            [0], [0], marker="o", color="w",
            markeredgecolor=MARKER_COLORS[k], markeredgewidth=2,
            markersize=8, markerfacecolor="none", label=names[idx]))
    ax1.set_xlabel("Normalised $D_{\\mathrm{cosine}}$", fontsize=11)
    ax1.set_ylabel("Normalised $D_{\\mathrm{vec}}$  (Euclidean)", fontsize=11)
    ax1.set_title(f"(a) Euclidean vs Cosine\nSpearman ρ = {rho_vc:.3f}",
                  fontsize=12, fontweight="bold")
    ax1.set_xlim(-0.02, 1.04)
    ax1.set_ylim(-0.02, 1.04)
    ax1.set_aspect("equal")
    ax1.legend(handles=leg, loc="upper left", fontsize=8,
              title="Largest |diff|", title_fontsize=9, framealpha=0.9)

    # ── (b) histogram of difference ──
    bins = np.linspace(-0.8, 0.8, 50)
    mae = np.mean(np.abs(diff))
    med = np.median(diff)
    sigma = np.std(diff)
    ax2.hist(diff, bins=bins, color="#8E44AD", alpha=0.7,
             edgecolor="white", lw=0.5)
    ax2.axvline(0, color="#666", lw=1, ls="--")
    ax2.axvline(med, color="#8E44AD", lw=2,
                label=f"median = {med:+.3f}")
    ax2.set_xlabel(
        r"$D_{\mathrm{vec}}^{\mathrm{norm}} - D_{\mathrm{cosine}}^{\mathrm{norm}}$",
        fontsize=12)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title(f"(b) Distribution of difference\n"
                  f"MAE = {mae:.3f},  σ = {sigma:.3f}",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)

    # ── (c) D_SAFT vs difference ──
    sn = norm01(d_saft)
    ax3.axvline(0, color="#666", lw=1, ls="--", zorder=1)
    ax3.scatter(diff, sn, c="#EB5757", s=16, alpha=0.55,
                edgecolors="none", zorder=3)
    rho_saft_diff, _ = spearmanr(diff, d_saft)
    ax3.set_xlabel(
        r"$D_{\mathrm{vec}}^{\mathrm{norm}} - D_{\mathrm{cosine}}^{\mathrm{norm}}$",
        fontsize=11)
    ax3.set_ylabel("Normalised $D_{\\mathrm{SAFT}}$", fontsize=11)
    ax3.set_title(f"(c) D_SAFT vs naive difference\n"
                  f"Spearman ρ = {rho_saft_diff:.3f}",
                  fontsize=12, fontweight="bold")

    # ── (d) D_struct vs difference ──
    stn = norm01(d_struct)
    ax4.axvline(0, color="#666", lw=1, ls="--", zorder=1)
    ax4.scatter(diff, stn, c="#27AE60", s=16, alpha=0.55,
                edgecolors="none", zorder=3)
    rho_struct_diff, _ = spearmanr(diff, d_struct)
    ax4.set_xlabel(
        r"$D_{\mathrm{vec}}^{\mathrm{norm}} - D_{\mathrm{cosine}}^{\mathrm{norm}}$",
        fontsize=11)
    ax4.set_ylabel("Normalised $D_{\\mathrm{struct}}$", fontsize=11)
    ax4.set_title(f"(d) D_struct vs naive difference\n"
                  f"Spearman ρ = {rho_struct_diff:.3f}",
                  fontsize=12, fontweight="bold")

    fig.suptitle(f"Euclidean vs Cosine naive distance  (target: {target})",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_euclidean_vs_cosine.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ═════════════════════════════════════════════════════════════════════
# Run one comparison suite
# ═════════════════════════════════════════════════════════════════════

def run_comparison(d_struct, d_saft, d_naive, names, target, N,
                   metric_label, tag, out_dir):
    """Generate all 3 figures for one naive metric."""

    stn = norm01(d_struct)
    sn  = norm01(d_saft)
    dn  = norm01(d_naive)

    rho_struct, _ = spearmanr(d_struct, d_naive)
    rho_saft,  _ = spearmanr(d_saft,   d_naive)

    print(f"\n  ── {metric_label} comparison ──")
    print(f"     Spearman ρ(D_struct, D_{tag}) = {rho_struct:.3f}")
    print(f"     Spearman ρ(D_SAFT,   D_{tag}) = {rho_saft:.3f}")

    p1 = os.path.join(out_dir, f"fig_{tag}_scatter.png")
    fig_scatter(stn, sn, dn, names, target, metric_label,
                rho_struct, rho_saft, p1)
    print(f"     [1] Scatter   → {p1}")

    p2 = os.path.join(out_dir, f"fig_{tag}_residuals.png")
    fig_residuals(stn, sn, dn, metric_label, rho_struct, rho_saft, p2)
    print(f"     [2] Residuals → {p2}")

    p3 = os.path.join(out_dir, f"fig_{tag}_rank_rank.png")
    fig_rank_rank(d_struct, d_saft, d_naive, N, metric_label,
                  rho_struct, rho_saft, target, p3)
    print(f"     [3] Rank-rank → {p3}")

    return rho_struct, rho_saft


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main(json_path: str):
    D = load_ranking(json_path)
    print(f"Loaded {D['N']} candidates  (target = {D['target']})")

    out_dir = os.path.join(os.path.dirname(json_path) or ".", "figures")
    os.makedirs(out_dir, exist_ok=True)

    # ── Euclidean comparison ──
    rho_st_vec, rho_sa_vec = run_comparison(
        D["d_struct"], D["d_saft"], D["d_vec"], D["names"],
        D["target"], D["N"],
        metric_label="vec", tag="euclidean", out_dir=out_dir)

    # ── Cosine comparison ──
    rho_st_cos, rho_sa_cos = run_comparison(
        D["d_struct"], D["d_saft"], D["d_cos"], D["names"],
        D["target"], D["N"],
        metric_label="cosine", tag="cosine", out_dir=out_dir)

    # ── Euclidean vs Cosine comparison ──
    p_vc = fig_euclidean_vs_cosine(
        D["d_vec"], D["d_cos"], D["d_struct"], D["d_saft"],
        D["names"], D["target"], out_dir)
    print(f"\n  ── Euclidean vs Cosine ──")
    print(f"     [4] Comparison → {p_vc}")

    # ── Summary table ──
    rho_ss, _ = spearmanr(D["d_saft"], D["d_struct"])
    rho_vc, _ = spearmanr(D["d_vec"],  D["d_cos"])

    print(f"\n{'='*70}")
    print(f"  Summary — Spearman ρ between all distance pairs")
    print(f"{'='*70}")
    print(f"{'Pair':<35s}  {'ρ':>8s}")
    print(f"{'-'*45}")
    print(f"{'D_struct  vs  D_vec (Euclidean)':<35s}  {rho_st_vec:8.3f}")
    print(f"{'D_SAFT    vs  D_vec (Euclidean)':<35s}  {rho_sa_vec:8.3f}")
    print(f"{'D_struct  vs  D_cosine':<35s}  {rho_st_cos:8.3f}")
    print(f"{'D_SAFT    vs  D_cosine':<35s}  {rho_sa_cos:8.3f}")
    print(f"{'D_SAFT    vs  D_struct':<35s}  {rho_ss:8.3f}")
    print(f"{'D_vec     vs  D_cosine':<35s}  {rho_vc:8.3f}")

    print(f"\n  All figures → {out_dir}/")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "ranking_vs_NCCO.json"
    main(p)