"""
Compute the SAFT-γ Mie log-Euclidean distance between two molecules.

Fully self-contained: only needs ``saft_pair_tables.json`` (generated once
by ``saft_similarity.py``).  No XML database or other imports required.

Usage
-----
    python scripts/compute_distance.py <vec_A> <vec_B>

where each vector is a comma-separated list of group counts aligned with
the group order stored in ``saft_pair_tables.json``.

Examples
--------
    # MEA  = [0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
    # DMEA = [0,0,0,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
    python scripts/compute_distance.py 0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0  0,0,0,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0

You can also import the module and call ``compute_distance`` directly:

    from compute_distance import load_tables, compute_distance
    tables = load_tables()
    d, sig_a, sig_b, components = compute_distance([0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                                                    [0,0,0,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                                                    tables)
"""

from __future__ import annotations

import json
import math
import os
import sys
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Load pre-computed pair tables (everything from the JSON, no XML needed)
# ═══════════════════════════════════════════════════════════════════════════════

def load_tables(json_path: str | None = None) -> dict:
    """
    Load everything needed for distance computation from the JSON file.

    The JSON contains:
        D_kl_dispersion, Delta_kl_association, sigma3_kl  – pair tables
        group_metadata   – per-group nu, shapeFactor, sigma
        groups           – ordered list of group names
        weights          – {w_J, w_S, w_M, w_P, w_SH}
        S0               – association floor
        T_ref_K, eta_ref – reference conditions

    Returns a dict with keys:
        group_names, disp_table, delta_table, sigma3_table,
        group_meta, weights, S0, settings
    """
    if json_path is None:
        base = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
        json_path = os.path.join(base, "saft_pair_tables.json")

    with open(json_path, "r") as f:
        data = json.load(f)

    group_names: list[str] = data["groups"]

    # Re-key from "k|l" strings to (k, l) tuples
    disp_table:   dict[tuple[str, str], float] = {}
    delta_table:  dict[tuple[str, str], float] = {}
    sigma3_table: dict[tuple[str, str], float] = {}

    for key, val in data["D_kl_dispersion"].items():
        k, l = key.split("|")
        disp_table[(k, l)] = val

    for key, val in data["Delta_kl_association"].items():
        k, l = key.split("|")
        delta_table[(k, l)] = val

    for key, val in data["sigma3_kl"].items():
        k, l = key.split("|")
        sigma3_table[(k, l)] = val

    return {
        "group_names":   group_names,
        "disp_table":    disp_table,
        "delta_table":   delta_table,
        "sigma3_table":  sigma3_table,
        "group_meta":    data["group_metadata"],
        "weights":       data["weights"],
        "S0":            data["S0"],
        "settings":      {"T_ref_K": data["T_ref_K"],
                          "eta_ref": data["eta_ref"]},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Signature & distance (self-contained, using pre-loaded tables)
# ═══════════════════════════════════════════════════════════════════════════════

def _signature(vector, tables: dict) -> dict:
    """
    Compute the 5-component SAFT-γ Mie signature for a molecule vector.

    Returns {"D_bar", "A_bar", "m_total", "sigma3_avg", "shape_avg"}.
    """
    group_names  = tables["group_names"]
    group_meta   = tables["group_meta"]
    disp_table   = tables["disp_table"]
    delta_table  = tables["delta_table"]
    sigma3_table = tables["sigma3_table"]

    n = np.asarray(vector, dtype=float)
    G = len(group_names)

    # Segment fractions  x_{s,k} = n_k · ν_k · S_k / m
    weighted = np.zeros(G)
    for i in range(G):
        if n[i] == 0:
            continue
        gm = group_meta[group_names[i]]
        weighted[i] = n[i] * gm["nu"] * gm["shapeFactor"]

    m_i = weighted.sum()
    if m_i == 0.0:
        return {"D_bar": 0.0, "A_bar": 0.0, "m_total": 0.0,
                "sigma3_avg": 0.0, "shape_avg": 0.0}

    xs = weighted / m_i

    D_sum = 0.0
    A_sum = 0.0
    sig3_sum = 0.0

    for i in range(G):
        if xs[i] == 0.0:
            continue
        gi = group_names[i]
        for j in range(G):
            if xs[j] == 0.0:
                continue
            gj = group_names[j]
            w = xs[i] * xs[j]
            D_sum    += w * disp_table[(gi, gj)]
            A_sum    += w * delta_table[(gi, gj)]
            sig3_sum += w * sigma3_table[(gi, gj)]

    shape_sum = 0.0
    for i in range(G):
        if xs[i] == 0.0:
            continue
        shape_sum += xs[i] * group_meta[group_names[i]]["shapeFactor"]

    return {"D_bar": D_sum, "A_bar": A_sum, "m_total": m_i,
            "sigma3_avg": sig3_sum, "shape_avg": shape_sum}


def compute_distance(vec_a, vec_b, tables: dict | None = None,
                     weights: dict | None = None,
                     s0: float | None = None) -> tuple[float, dict, dict, dict]:
    """
    Compute the log-Euclidean distance between two molecule vectors.

    Parameters
    ----------
    vec_a, vec_b : list[int]
        Group-count vectors aligned with ``tables["group_names"]``.
    tables : dict, optional
        Output of ``load_tables()``. If None, loads from default JSON path.
    weights : dict, optional
        {"w_J", "w_S", "w_M", "w_P", "w_SH"}.  Defaults to values
        stored in the JSON.
    s0 : float, optional
        Association floor.  Defaults to S0 from the JSON.

    Returns
    -------
    distance   : float
    sig_a      : dict   – 5-component signature of molecule A
    sig_b      : dict   – 5-component signature of molecule B
    components : dict   – per-component log-ratio contributions
                          {"d_D", "d_A", "d_m", "d_sigma", "d_shape"}
    """
    if tables is None:
        tables = load_tables()

    if weights is None:
        weights = tables["weights"]
    if s0 is None:
        s0 = tables["S0"]

    sig_a = _signature(vec_a, tables)
    sig_b = _signature(vec_b, tables)

    D_a = max(abs(sig_a["D_bar"]), 1e-300)
    D_b = max(abs(sig_b["D_bar"]), 1e-300)
    dD  = math.log(D_a / D_b)

    dA = math.log((sig_a["A_bar"] + s0) / (sig_b["A_bar"] + s0))

    m_a = max(sig_a["m_total"], 1e-300)
    m_b = max(sig_b["m_total"], 1e-300)
    dm  = math.log(m_a / m_b)

    s3_a = max(sig_a["sigma3_avg"], 1e-300)
    s3_b = max(sig_b["sigma3_avg"], 1e-300)
    ds3  = math.log(s3_a / s3_b)

    sh_a = max(sig_a["shape_avg"], 1e-300)
    sh_b = max(sig_b["shape_avg"], 1e-300)
    dsh  = math.log(sh_a / sh_b)

    dist = math.sqrt(weights["w_J"]  * dD**2
                   + weights["w_S"]  * dA**2
                   + weights["w_M"]  * dm**2
                   + weights["w_P"]  * ds3**2
                   + weights["w_SH"] * dsh**2)

    components = {"d_D": dD, "d_A": dA, "d_m": dm,
                  "d_sigma": ds3, "d_shape": dsh}

    return dist, sig_a, sig_b, components


def distance(base_vector, target_vector) -> float:
    """
    Compute the SAFT-γ Mie distance between two molecule vectors.

    This is a simplified interface that loads tables automatically and
    returns only the distance value.

    Parameters
    ----------
    base_vector, target_vector : list[int]
        Group-count vectors aligned with the standard group order.

    Returns
    -------
    distance : float
        The log-Euclidean distance between the two molecules.
    """
    dist, _, _, _ = compute_distance(base_vector, target_vector)
    return dist


# ═══════════════════════════════════════════════════════════════════════════════
# Pretty-print helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _vec_label(vec, group_names):
    """Compact human-readable label for a group-count vector."""
    parts = []
    for i, count in enumerate(vec):
        if count > 0:
            parts.append(f"{count}×{group_names[i]}")
    return " + ".join(parts) if parts else "(empty)"


def _print_signature(label: str, sig: dict):
    print(f"\n  {label}:")
    print(f"    D_bar      = {sig['D_bar']:.6e}   (dispersion, J·m³)")
    print(f"    A_bar      = {sig['A_bar']:.6e}   (association, m³)")
    print(f"    m_total    = {sig['m_total']:.4f}           (chain length)")
    print(f"    sigma3_avg = {sig['sigma3_avg']:.6e}   (packing, m³)")
    print(f"    shape_avg  = {sig['shape_avg']:.6f}         (shape factor)")


def _print_components(comp: dict, weights: dict):
    print("\n  Log-ratio components (and weighted²):")
    names = [("d_D",     "w_J",  "Dispersion"),
             ("d_A",     "w_S",  "Association"),
             ("d_m",     "w_M",  "Chain length"),
             ("d_sigma", "w_P",  "Packing σ³"),
             ("d_shape", "w_SH", "Shape")]
    for key, wkey, label in names:
        val = comp[key]
        w   = weights[wkey]
        print(f"    {label:<16s}  d = {val:+.6f}   w·d² = {w * val**2:.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 3:
        print("Usage: python compute_distance.py <vec_A> <vec_B>")
        print()
        print("Each vector is a comma-separated list of group counts.")
        print("Group order (from saft_pair_tables.json):")
        tables = load_tables()
        for i, g in enumerate(tables["group_names"]):
            print(f"  [{i:2d}] {g}")
        print()
        print("Example (MEA vs DMEA):")
        print("  python scripts/compute_distance.py "
              "0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0  "
              "0,0,0,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0")
        return

    vec_a = [int(x) for x in sys.argv[1].split(",")]
    vec_b = [int(x) for x in sys.argv[2].split(",")]

    tables = load_tables()
    group_names = tables["group_names"]
    G = len(group_names)

    if len(vec_a) != G or len(vec_b) != G:
        print(f"Error: vectors must have {G} elements (got {len(vec_a)} and {len(vec_b)}).")
        print(f"Group order: {group_names}")
        sys.exit(1)

    weights = tables["weights"]
    s0      = tables["S0"]

    dist, sig_a, sig_b, comp = compute_distance(vec_a, vec_b, tables)

    print("=" * 60)
    print("  SAFT-γ Mie distance between two molecules")
    print("=" * 60)
    print(f"\n  Molecule A:  {_vec_label(vec_a, group_names)}")
    print(f"  Molecule B:  {_vec_label(vec_b, group_names)}")

    _print_signature("Signature A", sig_a)
    _print_signature("Signature B", sig_b)
    _print_components(comp, weights)

    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  Distance  D = {dist:.6f}            ║")
    print(f"  ╚══════════════════════════════════════╝")

    print(f"\n  Weights:  w_D={weights['w_J']}  w_A={weights['w_S']}  "
          f"w_m={weights['w_M']}  w_σ={weights['w_P']}  w_S={weights['w_SH']}  "
          f"S0={s0:.0e}")


if __name__ == "__main__":
    main()
