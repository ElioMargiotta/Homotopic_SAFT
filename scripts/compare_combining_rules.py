"""
compare_combining_rules.py
==========================
Compare SAFT-γ Mie combining rules against explicit cross-interaction
values stored in the XML database, for **both** dispersive and
associative parameters.

Dispersive combining rules
--------------------------
    σ_{kl}  = (σ_{kk} + σ_{ll}) / 2
    λ_{kl}  = 3 + √[(λ_{kk}-3)(λ_{ll}-3)]
    ε_{kl}  = √(ε_{kk}·ε_{ll}) · (σ_{kk}³·σ_{ll}³)^½ / σ_{kl}³

Association combining rules (CR-1)
-----------------------------------
    ε^{assoc}_{kl}  = √( ε^{assoc}_{kk} · ε^{assoc}_{ll} )
    K_{kl}          = [ (∛K_{kk} + ∛K_{ll}) / 2 ]³

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


# ═════════════════════════════════════════════════════════════════════
# Collect dispersive cross pairs
# ═════════════════════════════════════════════════════════════════════

def collect_dispersive(xml_path: str):
    """Compare CR vs DB for ε, σ, λ_r, λ_a."""
    groups, cross = ss.load_database(xml_path)
    available = [g for g in ss.GROUPS_OF_INTEREST if g in groups]

    records = []
    seen = set()
    for k in available:
        for l in available:
            if k == l:
                continue
            pk = tuple(sorted([k, l]))
            if pk in seen:
                continue
            seen.add(pk)

            ci = cross.get((k, l))
            if ci is None:
                continue

            gk, gl = groups[k], groups[l]

            cr_sig = ss.combining_sigma(gk["sigma"], gl["sigma"])
            if ci["sigma"] != 0.0:
                records.append({"pair": f"{k} | {l}", "param": "sigma",
                                "db": ci["sigma"], "cr": cr_sig})

            cr_lr = ss.combining_lambda(gk["lambdaRepulsive"], gl["lambdaRepulsive"])
            if ci["lambdaRepulsive"] != 0.0:
                records.append({"pair": f"{k} | {l}", "param": "lambda_r",
                                "db": ci["lambdaRepulsive"], "cr": cr_lr})

            cr_la = ss.combining_lambda(gk["lambdaAttractive"], gl["lambdaAttractive"])
            if ci["lambdaAttractive"] != 0.0:
                records.append({"pair": f"{k} | {l}", "param": "lambda_a",
                                "db": ci["lambdaAttractive"], "cr": cr_la})

            sig_for = ci["sigma"] if ci["sigma"] != 0.0 else cr_sig
            cr_eps = ss.combining_epsilon(gk["epsilon"], gl["epsilon"],
                                          gk["sigma"], gl["sigma"], sig_for)
            if ci["epsilon"] != 0.0:
                records.append({"pair": f"{k} | {l}", "param": "epsilon",
                                "db": ci["epsilon"], "cr": cr_eps})

    return records


# ═════════════════════════════════════════════════════════════════════
# Collect associative cross pairs
# ═════════════════════════════════════════════════════════════════════

def collect_associative(xml_path: str):
    """
    Compare CR vs DB for ε^assoc and K (bonding volume).

    For each cross pair (k,l) that has explicit association parameters
    in the database, compute what the CR-1 combining rule would give
    from the self-interaction parameters of groups k and l.
    """
    groups, cross = ss.load_database(xml_path)
    available = [g for g in ss.GROUPS_OF_INTEREST if g in groups]

    records = []
    seen = set()

    for k in available:
        for l in available:
            if k == l:
                continue
            pk = tuple(sorted([k, l]))
            if pk in seen:
                continue
            seen.add(pk)

            ci = cross.get((k, l))
            if ci is None:
                continue

            db_assoc = ci.get("association", [])
            if not db_assoc:
                continue

            # Build CR-1 fallback from self-interactions
            cr_assoc = ss._cr1_association_fallback(k, l, groups)

            # Match DB interactions to CR interactions by canonical site types
            for db_inter in db_assoc:
                db_s1 = db_inter["site1"]
                db_s2 = db_inter["site2"]
                db_ea = db_inter["epsilonAssoc"]
                db_bv = db_inter["bondingVolume"]

                if db_ea == 0.0 and db_bv == 0.0:
                    continue

                # Find matching CR interaction by canonical site types
                db_canon = (ss._canonical_site(db_s1), ss._canonical_site(db_s2))
                cr_match = None
                for cr_inter in cr_assoc:
                    cr_canon = (ss._canonical_site(cr_inter["site1"]),
                                ss._canonical_site(cr_inter["site2"]))
                    if cr_canon == db_canon or cr_canon == (db_canon[1], db_canon[0]):
                        cr_match = cr_inter
                        break

                if cr_match is None:
                    continue

                site_label = f"{db_s1}–{db_s2}"
                pair_label = f"{k} | {l}  ({site_label})"

                # ε^assoc
                if db_ea != 0.0 and cr_match["epsilonAssoc"] != 0.0:
                    records.append({
                        "pair": pair_label, "param": "eps_assoc",
                        "db": db_ea, "cr": cr_match["epsilonAssoc"],
                    })

                # K (bonding volume)
                if db_bv != 0.0 and cr_match["bondingVolume"] != 0.0:
                    records.append({
                        "pair": pair_label, "param": "bond_vol",
                        "db": db_bv, "cr": cr_match["bondingVolume"],
                    })

    return records


# ═════════════════════════════════════════════════════════════════════
# Generic parity + error histogram plotter
# ═════════════════════════════════════════════════════════════════════

def plot_parity_grid(records, param_list, config, suptitle, out_path):
    """
    Plot a grid of parity plots.

    config : dict[param] -> {title, xlabel_db, xlabel_cr, scale, color}
    """
    n = len(param_list)
    ncols = min(n, 2)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 6 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # Pad axes if needed
    while len(axes) < len(param_list):
        axes = list(axes) + [fig.add_subplot(nrows, ncols, len(axes) + 1)]

    for idx, param in enumerate(param_list):
        ax = axes[idx]
        c = config[param]
        subset = [r for r in records if r["param"] == param]

        if not subset:
            ax.set_title(f"{c['title']}\n(no database entries)", fontsize=12)
            ax.set_xlabel(c["xlabel_db"], fontsize=10)
            ax.set_ylabel(c["xlabel_cr"], fontsize=10)
            continue

        s = c["scale"]
        db = np.array([r["db"] * s for r in subset])
        cr = np.array([r["cr"] * s for r in subset])
        names = [r["pair"] for r in subset]

        lo = min(db.min(), cr.min()) * 0.90
        hi = max(db.max(), cr.max()) * 1.10
        if lo == hi:
            lo -= 1; hi += 1

        ax.plot([lo, hi], [lo, hi], "--", color="#444", lw=1, zorder=1,
                label=r"$y = x$")
        ax.scatter(db, cr, c=c["color"], s=50, alpha=0.75,
                   edgecolors="white", linewidth=0.5, zorder=3)

        if len(db) > 1:
            coeffs = np.polyfit(db, cr, 1)
            x_fit = np.linspace(lo, hi, 100)
            ax.plot(x_fit, np.polyval(coeffs, x_fit), "-", color=c["color"],
                    lw=1.5, alpha=0.6,
                    label=rf"fit: $y = {coeffs[0]:.3f}\,x + {coeffs[1]:.1f}$")

            r_val, _ = pearsonr(db, cr)
            rel_err = np.abs(cr - db) / np.maximum(np.abs(db), 1e-30) * 100

            ax.text(0.05, 0.92,
                    f"$n = {len(subset)}$\n"
                    f"$R = {r_val:.4f}$\n"
                    rf"$\langle|\mathrm{{err}}|\rangle = {np.mean(rel_err):.1f}\%$" + "\n"
                    rf"$\max|\mathrm{{err}}| = {np.max(rel_err):.1f}\%$",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white",
                              ec="#ccc", alpha=0.9))

            for i in range(len(db)):
                rel = abs(cr[i] - db[i]) / max(abs(db[i]), 1e-30) * 100
                if rel > 10:
                    ax.annotate(names[i], (db[i], cr[i]), fontsize=5.5,
                                color="#666", xytext=(4, 4),
                                textcoords="offset points",
                                bbox=dict(boxstyle="round,pad=0.15",
                                          fc="w", ec="none", alpha=0.7))

        ax.set_xlabel(c["xlabel_db"], fontsize=10)
        ax.set_ylabel(c["xlabel_cr"], fontsize=10)
        ax.set_title(c["title"], fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax.set_aspect("equal")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    # Hide unused axes
    for j in range(len(param_list), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Parity → {out_path}")


def plot_error_grid(records, param_list, config, suptitle, out_path):
    """Plot a grid of relative error histograms."""
    n = len(param_list)
    ncols = min(n, 2)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    while len(axes) < len(param_list):
        axes = list(axes) + [fig.add_subplot(nrows, ncols, len(axes) + 1)]

    for idx, param in enumerate(param_list):
        ax = axes[idx]
        c = config[param]
        subset = [r for r in records if r["param"] == param]

        if not subset:
            ax.set_title(f"{c['title']}\n(no database entries)", fontsize=11)
            continue

        s = c["scale"]
        db = np.array([r["db"] * s for r in subset])
        cr = np.array([r["cr"] * s for r in subset])
        rel = (cr - db) / np.maximum(np.abs(db), 1e-30) * 100

        # Exclude extreme outliers for readability
        mask = np.abs(rel) < 500
        rel_plot = rel[mask]
        n_excl = np.sum(~mask)

        nbins = max(8, len(rel_plot) // 2)
        ax.hist(rel_plot, bins=nbins, color=c["color"], alpha=0.7,
                edgecolor="white", lw=0.5)
        ax.axvline(0, color="#444", lw=1, ls="--")
        ax.axvline(np.median(rel_plot), color=c["color"], lw=2,
                   label=rf"median $= {np.median(rel_plot):+.1f}\%$")

        ax.set_xlabel(c["err_xlabel"], fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        title = f"{c['title']}  ($n = {len(subset)}$)"
        if n_excl > 0:
            title += f"\n({n_excl} outlier{'s' if n_excl > 1 else ''} $> 500\\%$ excluded)"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)

    for j in range(len(param_list), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Errors → {out_path}")


# ═════════════════════════════════════════════════════════════════════
# Print summary table
# ═════════════════════════════════════════════════════════════════════

def print_table(records, param_list, config, title):
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")
    print(f"{'Pair':<35s}  {'Param':>12s}  {'Database':>12s}  "
          f"{'Comb. rule':>12s}  {'Rel.err':>9s}")
    print("-" * 85)
    for r in sorted(records, key=lambda x: (x["param"], x["pair"])):
        c = config.get(r["param"], {})
        s = c.get("scale", 1.0)
        rel = (r["cr"] - r["db"]) / abs(r["db"]) * 100 if r["db"] != 0 else 0
        print(f"{r['pair']:<35s}  {r['param']:>12s}  "
              f"{r['db']*s:12.4f}  {r['cr']*s:12.4f}  {rel:+8.2f}%")


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    xml_path = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--xml":
            xml_path = args[i + 1]; i += 2
        else:
            i += 1

    if xml_path is None:
        xml_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..",
                         "database", "CCS_Mie_Databank_221020.xml"))

    out_dir = os.path.join(os.path.dirname(__file__) or ".", "figures")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading database: {xml_path}")

    # ── Dispersive ──
    disp_records = collect_dispersive(xml_path)
    print(f"\nDispersive: {len(disp_records)} comparisons across "
          f"{len(set(r['pair'] for r in disp_records))} pairs")

    disp_config = {
        "epsilon": {
            "title":     r"$\varepsilon_{kl}/k_{\mathrm{B}}$  (K)",
            "xlabel_db": r"Database $\varepsilon_{kl}/k_{\mathrm{B}}$ (K)",
            "xlabel_cr": r"Combining rule $\varepsilon_{kl}/k_{\mathrm{B}}$ (K)",
            "err_xlabel": r"$(\varepsilon^{\mathrm{CR}} - \varepsilon^{\mathrm{DB}}) / |\varepsilon^{\mathrm{DB}}|$  (%)",
            "scale": 1.0, "color": "#E74C3C",
        },
        "sigma": {
            "title":     r"$\sigma_{kl}$  (Å)",
            "xlabel_db": r"Database $\sigma_{kl}$ (Å)",
            "xlabel_cr": r"Combining rule $\sigma_{kl}$ (Å)",
            "err_xlabel": r"$(\sigma^{\mathrm{CR}} - \sigma^{\mathrm{DB}}) / |\sigma^{\mathrm{DB}}|$  (%)",
            "scale": 1e10, "color": "#3498DB",
        },
        "lambda_r": {
            "title":     r"$\lambda^{\mathrm{r}}_{kl}$",
            "xlabel_db": r"Database $\lambda^{\mathrm{r}}_{kl}$",
            "xlabel_cr": r"Combining rule $\lambda^{\mathrm{r}}_{kl}$",
            "err_xlabel": r"$(\lambda_r^{\mathrm{CR}} - \lambda_r^{\mathrm{DB}}) / |\lambda_r^{\mathrm{DB}}|$  (%)",
            "scale": 1.0, "color": "#2ECC71",
        },
        "lambda_a": {
            "title":     r"$\lambda^{\mathrm{a}}_{kl}$",
            "xlabel_db": r"Database $\lambda^{\mathrm{a}}_{kl}$",
            "xlabel_cr": r"Combining rule $\lambda^{\mathrm{a}}_{kl}$",
            "err_xlabel": r"$(\lambda_a^{\mathrm{CR}} - \lambda_a^{\mathrm{DB}}) / |\lambda_a^{\mathrm{DB}}|$  (%)",
            "scale": 1.0, "color": "#F39C12",
        },
    }

    disp_params = ["epsilon", "sigma", "lambda_r", "lambda_a"]
    print_table(disp_records, disp_params, disp_config, "Dispersive cross parameters")

    plot_parity_grid(disp_records, disp_params, disp_config,
                     r"Combining rules vs database — dispersive cross parameters",
                     os.path.join(out_dir, "combining_rules_dispersive_parity.png"))
    plot_error_grid(disp_records, disp_params, disp_config,
                    r"Combining-rule error — dispersive parameters",
                    os.path.join(out_dir, "combining_rules_dispersive_errors.png"))

    # ── Associative ──
    assoc_records = collect_associative(xml_path)
    print(f"\nAssociative: {len(assoc_records)} comparisons across "
          f"{len(set(r['pair'] for r in assoc_records))} site pairs")

    assoc_config = {
        "eps_assoc": {
            "title":     r"$\varepsilon^{\mathrm{assoc}}_{kl}/k_{\mathrm{B}}$  (K)",
            "xlabel_db": r"Database $\varepsilon^{\mathrm{assoc}}_{kl}/k_{\mathrm{B}}$ (K)",
            "xlabel_cr": r"CR-1 $\varepsilon^{\mathrm{assoc}}_{kl}/k_{\mathrm{B}}$ (K)",
            "err_xlabel": r"$(\varepsilon_{\mathrm{assoc}}^{\mathrm{CR}} - \varepsilon_{\mathrm{assoc}}^{\mathrm{DB}}) / |\varepsilon_{\mathrm{assoc}}^{\mathrm{DB}}|$  (%)",
            "scale": 1.0, "color": "#8E44AD",
        },
        "bond_vol": {
            "title":     r"$K_{kl}$  (bonding volume, m³)",
            "xlabel_db": r"Database $K_{kl}$ (m³)",
            "xlabel_cr": r"CR-1 $K_{kl}$ (m³)",
            "err_xlabel": r"$(K^{\mathrm{CR}} - K^{\mathrm{DB}}) / |K^{\mathrm{DB}}|$  (%)",
            "scale": 1.0, "color": "#E67E22",
        },
    }

    assoc_params = ["eps_assoc", "bond_vol"]
    print_table(assoc_records, assoc_params, assoc_config, "Associative cross parameters")

    plot_parity_grid(assoc_records, assoc_params, assoc_config,
                     r"Combining rules vs database — association cross parameters (CR-1)",
                     os.path.join(out_dir, "combining_rules_association_parity.png"))
    plot_error_grid(assoc_records, assoc_params, assoc_config,
                    r"Combining-rule error — association parameters (CR-1)",
                    os.path.join(out_dir, "combining_rules_association_errors.png"))

    print(f"\n  All figures → {out_dir}/")


if __name__ == "__main__":
    main()