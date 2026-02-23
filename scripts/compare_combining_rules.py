"""
compare_combining_rules.py
==========================
Compare SAFT-γ Mie combining rules against explicit cross-interaction
values stored in the XML database.

For every unlike pair (k, l) that has an explicit database entry, we
compute what the combining rule *would* predict from the self-interaction
parameters alone:

    σ_{kl}^CR  = (σ_{kk} + σ_{ll}) / 2               (arithmetic)
    λ_{kl}^CR  = 3 + √[(λ_{kk}-3)(λ_{ll}-3)]         (nonlinear)
    ε_{kl}^CR  = √(ε_{kk}·ε_{ll}) · (σ_{kk}³·σ_{ll}³)^½ / σ_{kl}³   (Berthelot+σ³)

Then we compare against the database values and plot parity diagrams.

Usage
-----
    python compare_combining_rules.py [--xml path/to/database.xml]
"""

from __future__ import annotations
import os, sys, math
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import saft_similarity_florian as ss


def collect_pairs(xml_path: str):
    """
    For every cross pair with explicit database values, compute the
    combining-rule prediction and return both.

    Returns
    -------
    list of dict, each with:
        pair, param, db_value, cr_value
    """
    groups, cross = ss.load_database(xml_path)
    available = [g for g in ss.GROUPS_OF_INTEREST if g in groups]

    records = []
    seen = set()

    for k in available:
        for l in available:
            if k == l:
                continue
            pair_key = tuple(sorted([k, l]))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            ci = cross.get((k, l))
            if ci is None:
                continue  # no database entry, nothing to compare

            gk, gl = groups[k], groups[l]

            # ── σ ──
            cr_sig = ss.combining_sigma(gk["sigma"], gl["sigma"])
            db_sig = ci["sigma"]
            if db_sig != 0.0:
                records.append({
                    "pair": f"{k} | {l}", "param": "σ",
                    "db": db_sig, "cr": cr_sig,
                    "unit": "m",
                })

            # ── λ_r ──
            cr_lr = ss.combining_lambda(gk["lambdaRepulsive"], gl["lambdaRepulsive"])
            db_lr = ci["lambdaRepulsive"]
            if db_lr != 0.0:
                records.append({
                    "pair": f"{k} | {l}", "param": "λ_r",
                    "db": db_lr, "cr": cr_lr,
                    "unit": "",
                })

            # ── λ_a ──
            cr_la = ss.combining_lambda(gk["lambdaAttractive"], gl["lambdaAttractive"])
            db_la = ci["lambdaAttractive"]
            if db_la != 0.0:
                records.append({
                    "pair": f"{k} | {l}", "param": "λ_a",
                    "db": db_la, "cr": cr_la,
                    "unit": "",
                })

            # ── ε ──
            sig_for_eps = db_sig if db_sig != 0.0 else cr_sig
            cr_eps = ss.combining_epsilon(gk["epsilon"], gl["epsilon"],
                                          gk["sigma"], gl["sigma"],
                                          sig_for_eps)
            db_eps = ci["epsilon"]
            if db_eps != 0.0:
                records.append({
                    "pair": f"{k} | {l}", "param": "ε",
                    "db": db_eps, "cr": cr_eps,
                    "unit": "K",
                })

    return records


def make_plots(records, out_dir):
    """Generate parity plots for each parameter."""
    os.makedirs(out_dir, exist_ok=True)

    params = ["ε", "σ", "λ_r", "λ_a"]
    titles = {
        "ε":   "ε_{kl} / k_B  (K)",
        "σ":   "σ_{kl}  (Å)",
        "λ_r": "λ^r_{kl}",
        "λ_a": "λ^a_{kl}",
    }
    scale = {"σ": 1e10, "ε": 1.0, "λ_r": 1.0, "λ_a": 1.0}  # σ: m → Å
    colors = {"ε": "#E74C3C", "σ": "#3498DB", "λ_r": "#2ECC71", "λ_a": "#F39C12"}

    # ── Individual parity plots ──
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    axes = axes.flatten()

    summary = {}

    for idx, param in enumerate(params):
        ax = axes[idx]
        subset = [r for r in records if r["param"] == param]
        if not subset:
            ax.set_title(f"{titles[param]}\n(no database entries)")
            continue

        s = scale[param]
        db_vals  = np.array([r["db"] * s for r in subset])
        cr_vals  = np.array([r["cr"] * s for r in subset])
        pair_names = [r["pair"] for r in subset]

        # Parity line
        lo = min(db_vals.min(), cr_vals.min()) * 0.95
        hi = max(db_vals.max(), cr_vals.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "--", color="#444", lw=1, zorder=1,
                label="perfect agreement")

        # Scatter
        ax.scatter(db_vals, cr_vals, c=colors[param], s=50, alpha=0.75,
                   edgecolors="white", linewidth=0.5, zorder=3)

        # Linear regression
        if len(db_vals) > 1:
            coeffs = np.polyfit(db_vals, cr_vals, 1)
            x_fit = np.linspace(lo, hi, 100)
            y_fit = np.polyval(coeffs, x_fit)
            ax.plot(x_fit, y_fit, "-", color=colors[param], lw=1.5, alpha=0.6,
                    label=f"fit: y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}")

            r_val, _ = pearsonr(db_vals, cr_vals)

            # Relative errors
            rel_err = np.abs(cr_vals - db_vals) / np.abs(db_vals) * 100
            mae = np.mean(np.abs(cr_vals - db_vals))

            summary[param] = {
                "n": len(subset),
                "R": r_val,
                "slope": coeffs[0],
                "intercept": coeffs[1],
                "MAE": mae,
                "mean_rel_err": np.mean(rel_err),
                "max_rel_err": np.max(rel_err),
            }

            ax.text(0.05, 0.92,
                    f"n = {len(subset)}\n"
                    f"R = {r_val:.4f}\n"
                    f"mean |err| = {np.mean(rel_err):.1f}%\n"
                    f"max  |err| = {np.max(rel_err):.1f}%",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white",
                              ec="#ccc", alpha=0.9))
        else:
            summary[param] = {"n": len(subset)}

        # Annotate outliers (>10% relative error)
        for i in range(len(db_vals)):
            if len(db_vals) > 1:
                rel = abs(cr_vals[i] - db_vals[i]) / abs(db_vals[i]) * 100
                if rel > 10:
                    ax.annotate(pair_names[i], (db_vals[i], cr_vals[i]),
                                fontsize=6, color="#666", xytext=(4, 4),
                                textcoords="offset points",
                                bbox=dict(boxstyle="round,pad=0.15",
                                          fc="white", ec="none", alpha=0.7))

        ax.set_xlabel(f"Database {titles[param]}", fontsize=10)
        ax.set_ylabel(f"Combining rule {titles[param]}", fontsize=10)
        ax.set_title(f"{titles[param]}", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax.set_aspect("equal")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    fig.suptitle("Combining rules vs database: dispersive cross parameters",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "combining_rules_parity.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Parity plots → {path}")

    # ── Relative error distribution ──
    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 10))
    axes2 = axes2.flatten()

    for idx, param in enumerate(params):
        ax = axes2[idx]
        subset = [r for r in records if r["param"] == param]
        if not subset:
            continue

        s = scale[param]
        db_vals = np.array([r["db"] * s for r in subset])
        cr_vals = np.array([r["cr"] * s for r in subset])
        rel_err = (cr_vals - db_vals) / np.abs(db_vals) * 100  # signed

        ax.hist(rel_err, bins=max(8, len(rel_err) // 2), color=colors[param],
                alpha=0.7, edgecolor="white", lw=0.5)
        ax.axvline(0, color="#444", lw=1, ls="--")
        ax.axvline(np.median(rel_err), color=colors[param], lw=2,
                   label=f"median = {np.median(rel_err):+.2f}%")
        ax.set_xlabel(f"Relative error  (CR − DB) / |DB|  (%)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"{titles[param]}  (n = {len(subset)})",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    fig2.suptitle("Combining-rule relative error distribution",
                  fontsize=14, fontweight="bold", y=1.01)
    fig2.tight_layout()
    path2 = os.path.join(out_dir, "combining_rules_errors.png")
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Error histograms → {path2}")

    return summary


def main():
    # Parse arguments
    xml_path = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--xml":
            xml_path = args[i + 1]
            i += 2
        else:
            i += 1

    if xml_path is None:
        xml_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..",
                         "database", "CCS_Mie_Databank_221020.xml"))

    print(f"Loading database: {xml_path}")
    records = collect_pairs(xml_path)
    print(f"Found {len(records)} parameter comparisons "
          f"across {len(set(r['pair'] for r in records))} cross pairs")

    # Print table
    print(f"\n{'Pair':<25s}  {'Param':>5s}  {'Database':>12s}  "
          f"{'Comb. rule':>12s}  {'Rel. err':>10s}")
    print("-" * 72)
    for r in sorted(records, key=lambda x: (x["param"], x["pair"])):
        s = 1e10 if r["param"] == "σ" else 1.0
        unit = "Å" if r["param"] == "σ" else ("K" if r["param"] == "ε" else "")
        rel = (r["cr"] - r["db"]) / abs(r["db"]) * 100 if r["db"] != 0 else 0
        print(f"{r['pair']:<25s}  {r['param']:>5s}  "
              f"{r['db']*s:12.4f}  {r['cr']*s:12.4f}  {rel:+9.2f}%")

    out_dir = os.path.join(os.path.dirname(__file__) or ".", "figures")
    out_dir = os.path.normpath(out_dir)
    summary = make_plots(records, out_dir)

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"{'Param':<8s}  {'n':>4s}  {'R':>8s}  {'slope':>8s}  "
          f"{'MAE':>10s}  {'<|err|>':>8s}  {'max|err|':>8s}")
    print(f"{'-'*60}")
    for param in ["ε", "σ", "λ_r", "λ_a"]:
        s = summary.get(param, {})
        if "R" in s:
            unit = " K" if param == "ε" else (" Å" if param == "σ" else "")
            print(f"{param:<8s}  {s['n']:4d}  {s['R']:8.4f}  {s['slope']:8.4f}  "
                  f"{s['MAE']:9.4f}{unit}  {s['mean_rel_err']:7.1f}%  "
                  f"{s['max_rel_err']:7.1f}%")
        elif "n" in s:
            print(f"{param:<8s}  {s['n']:4d}  (insufficient data)")


if __name__ == "__main__":
    main()