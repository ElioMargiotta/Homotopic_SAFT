"""
Graphical representation of SAFT-γ Mie group-level similarity.

Uses the **same cross-pair quantities** computed in ``saft_similarity.py``:
  - D_{kl}  -- dispersion a1-proxy (Mie prefactor x eps x sig^3 x Sutherland integral)
  - Delta_{kl}  -- Wertheim TPT1 association strength (F . K . I)
  - sigma_{kl}  -- cross segment diameter (arithmetic combining rule)

The group-group distance is built from these cross interactions, not from
self-pair comparisons alone.  This ensures the figures faithfully reflect
the SAFT-gamma Mie physics used in the molecule-ranking workflow.

Figures generated
-----------------
  1. **Heatmap**       -- symmetric matrix of pairwise group distances.
  2. **MDS map**       -- 2-D classical MDS embedding, colour-coded by
                          chemical family.
  3. **Network**       -- spring-layout graph; edge thickness proportional
                          to similarity.
  4. **Amine/OH bars** -- horizontal bar chart of distances among nitrogen-
                          and oxygen-containing groups.
  5. **Radar chart**   -- per-group raw SAFT descriptors on a common scale.

Usage
-----
    python scripts/plot_group_similarity.py

Outputs are saved to ``figures/``.
"""

from __future__ import annotations

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from itertools import combinations_with_replacement

# -- import SAFT-gamma Mie helpers from the main script ----------------------
sys.path.insert(0, os.path.dirname(__file__))
from saft_similarity import (
    load_database,
    build_pair_tables,
    get_pair_params,
    dispersion_a1_proxy,
    delta_pair,
    mie_prefactor,
    combining_sigma,
    combining_lambda,
    combining_epsilon,
    T_REF, ETA_REF,
)

# =============================================================================
# Configuration
# =============================================================================

GROUPS_OF_INTEREST = [
    "CH3", "CH2", "CH", "C",
    "cCH2", "cCH",
    "CH2OH", "CH2OH_Short","CHOH",
    "OH_Short",
    #"NH2", "NH2_2nd", "NH", "NH_2nd",
    #"N", "N_2nd",
    #"cNH", "cN",
    #"cCHNH", "cCHN",
    #"H2O", "CO2",
]

FAMILY_COLOURS = {
    "Alkyl":       "#4C72B0",
    "Cycloalkyl":  "#55A868",
    "Hydroxyl":    "#C44E52",
    "Amine":       "#8172B2",
    "Cyclo-N":     "#CCB974",
    "Small mol.":  "#64B5CD",
}

def _family(g: str) -> str:
    if g in ("CH3", "CH2", "CH", "C"):
        return "Alkyl"
    if g in ("cCH2", "cCH"):
        return "Cycloalkyl"
    if g in ("CH2OH", "CH2OH_Short", "CHOH", "OH_Short"):
        return "Hydroxyl"
    if g in ("NH2", "NH2_2nd", "NH", "NH_2nd", "N", "N_2nd"):
        return "Amine"
    if g in ("cNH", "cN", "cCHNH", "cCHN", "cCHNH_np"):
        return "Cyclo-N"
    if g in ("H2O", "CO2", "N2"):
        return "Small mol."
    return "Alkyl"


# =============================================================================
# Group-group distance matrix  (cross-interaction based)
# =============================================================================

# Floor for association log-ratio at group level.  Δ_{kl} values are
# in m³ (~10⁻²⁶ for associating groups, 0 for non-associating).
# The floor prevents ln(0) without distorting genuine association values.
_DELTA_FLOOR = 1e-30   # m³  (negligible vs any real Δ > 10⁻²⁸)


def _group_distance(k: str, l: str,
                    disp_table: dict, delta_table: dict,
                    param_table: dict) -> float:
    r"""
    Log-Euclidean distance between groups k and l using **cross-pair**
    quantities from the SAFT-gamma Mie pair tables.

    Three components, each measuring how much the cross interaction
    deviates from the geometric-mean reference of the self interactions:

        d_D = ln | D_{kl} / sqrt(|D_{kk}| * |D_{ll}|) |
              -> 0 when the cross dispersion equals the geo-mean
                 of the self dispersions (Berthelot-like).

        d_Delta = ln [ (Delta_{kl} + s0) / sqrt((Delta_{kk}+s0)(Delta_{ll}+s0)) ]
              -> 0 when cross association = geo-mean of self associations.
              s0 = _DELTA_FLOOR prevents ln(0) for non-associating groups.

        d_sigma = ln ( sigma_{kl}^3 / sqrt(sigma_{kk}^3 * sigma_{ll}^3) )
              -> 0 when the arithmetic sigma combining rule gives the same
                 effective volume as the self terms.

    These ratios measure *thermodynamic compatibility*: groups that
    interact with each other almost as strongly as with themselves
    (d ~ 0) are "similar" in the SAFT-gamma Mie sense.

    For self-pairs (k == l) the distance is exactly 0.
    """
    if k == l:
        return 0.0

    # -- Dispersion component -------------------------------------------------
    Dkk = abs(disp_table[(k, k)])
    Dll = abs(disp_table[(l, l)])
    Dkl = abs(disp_table[(k, l)])
    geo_D = math.sqrt(max(Dkk, 1e-300) * max(Dll, 1e-300))
    dD = math.log(max(Dkl, 1e-300) / max(geo_D, 1e-300))

    # -- Association component ------------------------------------------------
    Akk = delta_table[(k, k)] + _DELTA_FLOOR
    All_ = delta_table[(l, l)] + _DELTA_FLOOR
    Akl = delta_table[(k, l)] + _DELTA_FLOOR
    geo_A = math.sqrt(Akk * All_)
    dA = math.log(Akl / geo_A)

    # -- Size (sigma^3) component ---------------------------------------------
    skk = param_table[(k, k)]["sigma"] ** 3
    sll = param_table[(l, l)]["sigma"] ** 3
    skl = param_table[(k, l)]["sigma"] ** 3
    geo_s = math.sqrt(max(skk, 1e-300) * max(sll, 1e-300))
    ds = math.log(max(skl, 1e-300) / max(geo_s, 1e-300))

    return math.sqrt(dD**2 + dA**2 + ds**2)


def build_distance_matrix(group_names: list[str],
                          disp_table: dict,
                          delta_table: dict,
                          param_table: dict) -> np.ndarray:
    """Symmetric NxN distance matrix using cross-pair SAFT quantities."""
    N = len(group_names)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = _group_distance(group_names[i], group_names[j],
                                disp_table, delta_table, param_table)
            D[i, j] = d
            D[j, i] = d
    return D


# =============================================================================
# Classical MDS  (Torgerson 1952 -- pure numpy, no scipy)
# =============================================================================

def classical_mds(D: np.ndarray, ndim: int = 2) -> np.ndarray:
    N = D.shape[0]
    D2 = D ** 2
    H = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * H @ D2 @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1][:ndim]
    lam = np.maximum(eigvals[idx], 0.0)
    return eigvecs[:, idx] * np.sqrt(lam)[np.newaxis, :]


# =============================================================================
# Nearest-neighbour ordering (for heatmap visual clustering)
# =============================================================================

def _nn_order(D: np.ndarray) -> list[int]:
    N = D.shape[0]
    visited = [False] * N
    order = [0]
    visited[0] = True
    for _ in range(N - 1):
        last = order[-1]
        dists = D[last].copy()
        dists[visited] = np.inf
        nxt = int(np.argmin(dists))
        order.append(nxt)
        visited[nxt] = True
    return order


# =============================================================================
# Figure 1 -- Heatmap
# =============================================================================

def plot_heatmap(D: np.ndarray, labels: list[str], out_path: str):
    fig, ax = plt.subplots(figsize=(12, 10))

    order = _nn_order(D)
    D_ord = D[np.ix_(order, order)]
    lab_ord = [labels[i] for i in order]

    im = ax.imshow(D_ord, cmap="YlOrRd", interpolation="nearest",
                   origin="upper")

    ax.set_xticks(range(len(lab_ord)))
    ax.set_yticks(range(len(lab_ord)))
    ax.set_xticklabels(lab_ord, rotation=55, ha="right", fontsize=9)
    ax.set_yticklabels(lab_ord, fontsize=9)

    for i in range(len(lab_ord)):
        for j in range(len(lab_ord)):
            val = D_ord[i, j]
            colour = "white" if val > D_ord.max() * 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5.5, color=colour)

    cbar = fig.colorbar(im, ax=ax, shrink=0.80, pad=0.02)
    cbar.set_label(r"Log-Euclidean distance  $\sqrt{d_D^2 + d_\Delta^2 + d_\sigma^2}$",
                   fontsize=10)

    ax.set_title("SAFT-$\\gamma$ Mie group-group distance\n"
                 "(cross-interaction based: $D_{kl},\\;\\Delta_{kl},\\;\\sigma_{kl}$)",
                 fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved to {out_path}")


# =============================================================================
# Figure 2 -- MDS scatter
# =============================================================================

def plot_mds(D: np.ndarray, labels: list[str], out_path: str):
    coords = classical_mds(D, ndim=2)
    fig, ax = plt.subplots(figsize=(11, 8.5))

    # Light edges for closest 30% of pairs
    _draw_close_edges(ax, D, coords, quantile=0.30)

    plotted = set()
    for i, (x, y) in enumerate(coords):
        fam = _family(labels[i])
        col = FAMILY_COLOURS.get(fam, "#999999")
        lbl = fam if fam not in plotted else None
        plotted.add(fam)
        ax.scatter(x, y, s=200, c=col, edgecolors="white", linewidths=0.8,
                   zorder=5, label=lbl)
        ax.annotate(labels[i], (x, y), fontsize=8, fontweight="bold",
                    textcoords="offset points", xytext=(7, 5), zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.75))

    ax.legend(loc="best", fontsize=9, framealpha=0.9,
              title="Chemical family", title_fontsize=10)
    ax.set_xlabel("MDS dimension 1", fontsize=11)
    ax.set_ylabel("MDS dimension 2", fontsize=11)
    ax.set_title("2-D MDS embedding of SAFT-$\\gamma$ Mie groups\n"
                 "(distance from cross-pair $D_{kl}$, $\\Delta_{kl}$, "
                 "$\\sigma_{kl}^3$)",
                 fontsize=12, fontweight="bold", pad=14)
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  MDS map saved to {out_path}")


def _draw_close_edges(ax, D, coords, quantile):
    N = D.shape[0]
    dists = [D[i, j] for i in range(N) for j in range(i + 1, N)]
    if not dists:
        return
    threshold = np.quantile(dists, quantile)
    for i in range(N):
        for j in range(i + 1, N):
            if D[i, j] <= threshold:
                ax.plot([coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        color="#d0d0d0", lw=0.6, zorder=1)


# =============================================================================
# Figure 3 -- Network / spring graph
# =============================================================================

def plot_network(D: np.ndarray, labels: list[str], out_path: str,
                 k_neighbours: int = 4):
    N = D.shape[0]
    coords = _spring_layout(D, k_neighbours=k_neighbours, n_iter=500)
    fig, ax = plt.subplots(figsize=(12, 10))

    # Build kNN edge set
    edges = set()
    max_d = 0.0
    for i in range(N):
        nns = np.argsort(D[i])[1:k_neighbours + 1]
        for j in nns:
            pair = (min(i, j), max(i, j))
            edges.add(pair)
            if D[i, j] > max_d:
                max_d = D[i, j]

    for (i, j) in edges:
        d = D[i, j]
        lw = max(0.4, 3.5 * (1.0 - d / max(max_d, 1e-10)))
        alpha = max(0.15, 1.0 - 0.7 * d / max(max_d, 1e-10))
        ax.plot([coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color="#555555", lw=lw, alpha=alpha, zorder=1)
        mx = (coords[i, 0] + coords[j, 0]) / 2
        my = (coords[i, 1] + coords[j, 1]) / 2
        ax.text(mx, my, f"{d:.2f}", fontsize=5.5, ha="center",
                va="center", color="#888888", zorder=2)

    for i in range(N):
        col = FAMILY_COLOURS.get(_family(labels[i]), "#999999")
        ax.scatter(coords[i, 0], coords[i, 1], s=280, c=col,
                   edgecolors="white", linewidths=1.2, zorder=5)
        ax.annotate(labels[i], (coords[i, 0], coords[i, 1]),
                    fontsize=8.5, fontweight="bold",
                    textcoords="offset points", xytext=(8, 6), zorder=6,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="none", alpha=0.8))

    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=c, markersize=10, label=f)
               for f, c in FAMILY_COLOURS.items()]
    ax.legend(handles=handles, loc="best", fontsize=9, framealpha=0.9,
              title="Chemical family", title_fontsize=10)
    ax.set_title("SAFT-$\\gamma$ Mie group similarity network\n"
                 f"({k_neighbours}-NN edges, thickness ~ 1/distance, "
                 "cross-pair based)",
                 fontsize=12, fontweight="bold", pad=14)
    ax.set_xlabel("Spring-layout x", fontsize=10)
    ax.set_ylabel("Spring-layout y", fontsize=10)
    ax.grid(True, ls="--", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Network graph saved to {out_path}")


def _spring_layout(D: np.ndarray, k_neighbours: int = 4,
                   n_iter: int = 500, lr: float = 0.05) -> np.ndarray:
    """Fruchterman-Reingold layout initialised from MDS."""
    N = D.shape[0]
    pos = classical_mds(D, ndim=2)
    span = pos.max(axis=0) - pos.min(axis=0)
    span[span == 0] = 1.0
    pos = (pos - pos.min(axis=0)) / span

    adj = set()
    for i in range(N):
        nns = np.argsort(D[i])[1:k_neighbours + 1]
        for j in nns:
            adj.add((min(i, j), max(i, j)))

    ideal_len = 1.0 / math.sqrt(N + 1)
    for it in range(n_iter):
        forces = np.zeros_like(pos)
        temp = lr * (1.0 - it / n_iter)
        for i in range(N):
            diff = pos[i] - pos
            dist = np.linalg.norm(diff, axis=1)
            dist[dist == 0] = 1e-8
            rep = (ideal_len**2 / dist)[:, None] * (diff / dist[:, None])
            rep[i] = 0.0
            forces[i] += rep.sum(axis=0)
        for (i, j) in adj:
            diff = pos[j] - pos[i]
            dist = np.linalg.norm(diff)
            if dist < 1e-8:
                continue
            f = (dist**2 / ideal_len) * diff / dist
            forces[i] += f
            forces[j] -= f
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        pos += temp * forces / norms * np.minimum(norms, temp)
    return pos


# =============================================================================
# Figure 4 -- Amine / hydroxyl bar chart
# =============================================================================

def plot_amine_distances(D: np.ndarray, labels: list[str], out_path: str):
    """Bar chart of distances among N- and O-containing groups."""
    focus = {"NH2", "NH2_2nd", "NH", "NH_2nd", "N", "N_2nd",
             "CH2OH", "CH2OH_Short", "CHOH", "OH_Short"}
    focus_idx = [(i, g) for i, g in enumerate(labels) if g in focus]
    if len(focus_idx) < 2:
        print("  Too few amine/hydroxyl groups to plot bar chart.")
        return

    pairs, dists = [], []
    for a in range(len(focus_idx)):
        for b in range(a + 1, len(focus_idx)):
            i, gi = focus_idx[a]
            j, gj = focus_idx[b]
            pairs.append(f"{gi}  <->  {gj}")
            dists.append(D[i, j])

    order = np.argsort(dists)
    pairs = [pairs[o] for o in order]
    dists = [dists[o] for o in order]

    amines = {"NH2", "NH2_2nd", "NH", "NH_2nd", "N", "N_2nd"}
    hydroxyl = {"CH2OH", "CH2OH_Short", "CHOH", "OH_Short"}

    colours = []
    for p in pairs:
        g1, g2 = [s.strip() for s in p.split("<->")]
        if g1 in amines and g2 in amines:
            colours.append(FAMILY_COLOURS["Amine"])
        elif g1 in hydroxyl and g2 in hydroxyl:
            colours.append(FAMILY_COLOURS["Hydroxyl"])
        else:
            colours.append("#CCB974")

    fig, ax = plt.subplots(figsize=(10, max(6, len(pairs) * 0.35)))
    y_pos = np.arange(len(pairs))
    bars = ax.barh(y_pos, dists, color=colours, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pairs, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Log-Euclidean distance (cross-pair based)", fontsize=11)
    ax.set_title("Pairwise distances among amine & hydroxyl groups\n"
                 "(SAFT-$\\gamma$ Mie cross-pair: $D_{kl}$, "
                 "$\\Delta_{kl}$, $\\sigma_{kl}^3$)",
                 fontsize=12, fontweight="bold", pad=12)

    for bar, d in zip(bars, dists):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{d:.3f}", va="center", fontsize=7.5, color="#333333")

    ax.legend(handles=[
        Patch(facecolor=FAMILY_COLOURS["Amine"], label="Amine <-> Amine"),
        Patch(facecolor=FAMILY_COLOURS["Hydroxyl"], label="Hydroxyl <-> Hydroxyl"),
        Patch(facecolor="#CCB974", label="Amine <-> Hydroxyl"),
    ], loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="x", ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Amine/Hydroxyl bar chart saved to {out_path}")


# =============================================================================
# Figure 5 -- Radar chart of raw group descriptors
# =============================================================================

def plot_radar(group_names: list[str], groups: dict,
               disp_table: dict, delta_table: dict,
               param_table: dict, out_path: str):
    """
    Radar (spider) chart of normalised self-pair SAFT descriptors.

    Each group is drawn as a polygon on 6 axes:
      eps_{kk}, sigma_{kk}, lambda^r_{kk}, lambda^a_{kk}, |D_{kk}|, Delta_{kk}
    all min-max normalised to [0, 1].
    """
    # Select a readable subset (skip near-duplicates for clarity)
    show = ["CH3", "CH2", "CH2OH", "CHOH",
            "NH2", "NH", "N", "H2O"]
    show = [g for g in show if g in group_names]
    if len(show) < 3:
        show = group_names[:min(8, len(group_names))]

    axes_labels = [r"$\varepsilon_{kk}$", r"$\sigma_{kk}$",
                   r"$\lambda^r_{kk}$",  r"$\lambda^a_{kk}$",
                   r"$|D_{kk}|$",         r"$\Delta_{kk}$"]
    n_axes = len(axes_labels)

    # Gather raw values
    raw = {}
    for g in show:
        eps = groups[g]["epsilon"]
        sig = groups[g]["sigma"]
        lr  = groups[g]["lambdaRepulsive"]
        la  = groups[g]["lambdaAttractive"]
        Dkk = abs(disp_table[(g, g)])
        Akk = delta_table[(g, g)]
        raw[g] = [eps, sig, lr, la, Dkk, Akk]

    # Min-max normalise each axis
    vals = np.array([raw[g] for g in show])
    mins = vals.min(axis=0)
    maxs = vals.max(axis=0)
    spans = maxs - mins
    spans[spans == 0] = 1.0
    normed = (vals - mins) / spans

    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]   # close polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    cmap = plt.cm.tab10
    for idx, g in enumerate(show):
        values = normed[idx].tolist() + [normed[idx][0]]
        col = cmap(idx / max(len(show) - 1, 1))
        ax.plot(angles, values, linewidth=1.8, label=g, color=col)
        ax.fill(angles, values, alpha=0.08, color=col)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                       fontsize=7, color="grey")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              fontsize=9, framealpha=0.9, title="Group", title_fontsize=10)
    ax.set_title("SAFT-$\\gamma$ Mie group descriptors\n(min-max normalised)",
                 fontsize=13, fontweight="bold", pad=24)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Radar chart saved to {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    base_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), ".."))
    xml_path = os.path.join(base_dir, "database", "CCS_Mie_Databank_221020.xml")
    fig_dir  = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("  SAFT-gamma Mie group similarity  --  visualisation")
    print("=" * 60)

    # -- Load database & pair tables (same as saft_similarity.py) --
    groups, cross = load_database(xml_path)
    available = [g for g in GROUPS_OF_INTEREST if g in groups]
    missing   = [g for g in GROUPS_OF_INTEREST if g not in groups]
    if missing:
        print(f"  Groups not in database (skipped): {missing}")
    group_names = available
    print(f"  Using {len(group_names)} groups.")

    disp_table, delta_table, param_table = build_pair_tables(
        group_names, groups, cross, T=T_REF, eta=ETA_REF)

    # -- Distance matrix (cross-pair based) --
    D = build_distance_matrix(group_names, disp_table, delta_table, param_table)

    # Quick summary for key pairs
    idx_map = {g: i for i, g in enumerate(group_names)}
    key_pairs = [("NH2", "NH"), ("NH2", "N"), ("NH", "N"),
                 ("NH2_2nd", "NH_2nd"), ("NH2_2nd", "N_2nd"),
                 ("NH2", "CHOH"), ("CH2OH", "CHOH"),
                 ("CH3", "CH2")]
    print("\n  Cross-pair based distances (selected):")
    for g1, g2 in key_pairs:
        if g1 in idx_map and g2 in idx_map:
            d = D[idx_map[g1], idx_map[g2]]
            print(f"    {g1:>12s}  <->  {g2:<12s}  d = {d:.4f}")
    print()

    # -- Figures --
    plot_heatmap(D, group_names,
                 os.path.join(fig_dir, "group_distance_heatmap.png"))

    plot_mds(D, group_names,
             os.path.join(fig_dir, "group_mds_map.png"))

    plot_network(D, group_names,
                 os.path.join(fig_dir, "group_similarity_network.png"),
                 k_neighbours=4)

    plot_amine_distances(D, group_names,
                         os.path.join(fig_dir, "amine_hydroxyl_distances.png"))

    plot_radar(group_names, groups, disp_table, delta_table, param_table,
               os.path.join(fig_dir, "group_radar_descriptors.png"))

    print(f"\nAll figures saved to  {fig_dir}/")


if __name__ == "__main__":
    main()