"""
Compute four distance metrics between two molecules from their
group-count vectors.

Distances
---------
1. **D_thermo** — SAFT-γ Mie thermodynamic distance (Euclidean in
   F_mono, F_chain, F_assoc).  Measures how differently two molecules
   interact energetically.

2. **D_struct** — SAFT-γ Mie structural distance (log-Euclidean in
   m, σ³, S̄).  Measures how differently two molecules are built.

3. **D_vec** — simple Euclidean distance between group-count vectors.
   Measures raw compositional difference.

4. **D_cos** — cosine distance between group-count vectors.  Measures
   angular difference in composition (size-invariant).

The SAFT distances (1–2) require the XML database to build pair tables
and compute full Helmholtz signatures.  The naive distances (3–4) need
only the group-count vectors.

Usage
-----
    python compute_distance.py <vec_A> <vec_B> [--xml path/to/database.xml]

where each vector is a comma-separated list of group counts aligned with
the group order in ``saft_similarity_florian.GROUPS_OF_INTEREST``.

Examples
--------
    # NCCO (MEA) vs CC(N)CO (2-amino-1-propanol)
    python compute_distance.py 0,0,0,0,1,0,0,1,0,1,0,0,0,0 \\
                                0,0,0,1,0,1,0,1,0,1,0,0,0,0

You can also import and call functions directly:

    from compute_distance import compute_all_distances, load_saft_tables
    tables = load_saft_tables("path/to/database.xml")
    result = compute_all_distances(vec_a, vec_b, tables)
"""

from __future__ import annotations

import math
import os
import sys

# Import SAFT-γ Mie machinery from saft_similarity_florian
import saft_similarity_florian as ss


# ═══════════════════════════════════════════════════════════════════════════════
# Load SAFT tables from XML (once)
# ═══════════════════════════════════════════════════════════════════════════════

def load_saft_tables(xml_path: str | None = None) -> dict:
    """
    Load the XML database and build all pair tables needed for SAFT
    distance computation.

    Parameters
    ----------
    xml_path : str, optional
        Path to ``CCS_Mie_Databank_221020.xml``.  If None, uses default
        relative path ``../database/CCS_Mie_Databank_221020.xml``.

    Returns
    -------
    dict with keys:
        group_names, groups, cross, a1_table, delta_table, param_table,
        T, eta, weights_thermo, weights_struct
    """
    if xml_path is None:
        xml_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..",
                         "database", "CCS_Mie_Databank_221020.xml"))

    groups, cross = ss.load_database(xml_path)

    available = [g for g in ss.GROUPS_OF_INTEREST if g in groups]
    group_names = available

    a1_table, delta_table, param_table = ss.build_pair_tables(
        group_names, groups, cross, T=ss.T_REF, eta=ss.ETA_REF)

    return {
        "group_names":     group_names,
        "groups":          groups,
        "cross":           cross,
        "a1_table":        a1_table,
        "delta_table":     delta_table,
        "param_table":     param_table,
        "T":               ss.T_REF,
        "eta":             ss.ETA_REF,
        "weights_thermo":  {"w_MONO": ss.W_MONO, "w_CHAIN": ss.W_CHAIN,
                            "w_ASSOC": ss.W_ASSOC},
        "weights_struct":  {"w_M": ss.W_M, "w_P": ss.W_P, "w_SH": ss.W_SH},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Distance functions
# ═══════════════════════════════════════════════════════════════════════════════

def distance_thermo(vec_a, vec_b, tables: dict) -> tuple[float, dict, dict]:
    """
    SAFT thermodynamic distance: Euclidean in (F_mono, F_chain, F_assoc).

    Returns (distance, signature_a, signature_b).
    """
    sig_a = ss.signature(vec_a, tables["group_names"], tables["groups"],
                         tables["a1_table"], tables["delta_table"],
                         tables["param_table"], tables["cross"],
                         tables["T"], tables["eta"])
    sig_b = ss.signature(vec_b, tables["group_names"], tables["groups"],
                         tables["a1_table"], tables["delta_table"],
                         tables["param_table"], tables["cross"],
                         tables["T"], tables["eta"])

    d = ss.euclidean_distance_thermo(sig_a, sig_b, tables["weights_thermo"])
    return d, sig_a, sig_b


def distance_struct(vec_a, vec_b, tables: dict) -> float:
    """
    SAFT structural distance: log-Euclidean in (m, σ³, S̄).

    Returns distance (signatures computed internally).
    """
    sig_a = ss.signature(vec_a, tables["group_names"], tables["groups"],
                         tables["a1_table"], tables["delta_table"],
                         tables["param_table"], tables["cross"],
                         tables["T"], tables["eta"])
    sig_b = ss.signature(vec_b, tables["group_names"], tables["groups"],
                         tables["a1_table"], tables["delta_table"],
                         tables["param_table"], tables["cross"],
                         tables["T"], tables["eta"])

    return ss.distance_structure(sig_a, sig_b, tables["weights_struct"])


def distance_euclidean(vec_a, vec_b) -> float:
    """
    Simple Euclidean distance between group-count vectors.

        D_vec = sqrt( sum_k (n_{k,a} - n_{k,b})^2 )
    """
    return ss.euclidean_distance_vector(vec_a, vec_b)


def distance_cosine(vec_a, vec_b) -> float:
    """
    Cosine distance between group-count vectors.

        D_cos = 1 - (a . b) / (|a| |b|)

    Size-invariant: identical group ratios → D_cos = 0.
    """
    return ss.cosine_distance_vector(vec_a, vec_b)


# ═══════════════════════════════════════════════════════════════════════════════
# Unified interface
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_distances(vec_a, vec_b, tables: dict) -> dict:
    """
    Compute all four distance metrics between two molecules.

    Parameters
    ----------
    vec_a, vec_b : list[int]
        Group-count vectors aligned with ``tables["group_names"]``.
    tables : dict
        Output of ``load_saft_tables()``.

    Returns
    -------
    dict with keys:
        d_thermo, d_struct, d_euclidean, d_cosine,
        sig_a, sig_b,
        components_thermo  (ΔF_mono, ΔF_chain, ΔF_assoc)
    """
    d_th, sig_a, sig_b = distance_thermo(vec_a, vec_b, tables)
    d_st = ss.distance_structure(sig_a, sig_b, tables["weights_struct"])
    d_eu = distance_euclidean(vec_a, vec_b)
    d_co = distance_cosine(vec_a, vec_b)

    components = {
        "dF_mono":  sig_a["F_mono"]  - sig_b["F_mono"],
        "dF_chain": sig_a["F_chain"] - sig_b["F_chain"],
        "dF_assoc": sig_a["F_assoc"] - sig_b["F_assoc"],
        "d_m":      math.log(max(sig_a["m_total"], 1e-300) /
                             max(sig_b["m_total"], 1e-300)),
        "d_sigma3": math.log(max(sig_a["sigma3_avg"], 1e-300) /
                             max(sig_b["sigma3_avg"], 1e-300)),
        "d_shape":  math.log(max(sig_a["shape_avg"], 1e-300) /
                             max(sig_b["shape_avg"], 1e-300)),
    }

    return {
        "d_thermo":    d_th,
        "d_struct":    d_st,
        "d_euclidean": d_eu,
        "d_cosine":    d_co,
        "sig_a":       sig_a,
        "sig_b":       sig_b,
        "components_thermo": components,
    }


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
    print(f"    F_mono     = {sig['F_mono']:12.6f}   (monomer A^mono/NkBT)")
    print(f"    F_chain    = {sig['F_chain']:12.6f}   (chain   A^chain/NkBT)")
    print(f"    F_assoc    = {sig['F_assoc']:12.6f}   (assoc   A^assoc/NkBT)")
    print(f"    m_total    = {sig['m_total']:12.4f}   (chain length)")
    print(f"    sigma3_avg = {sig['sigma3_avg']:12.6e}   (packing, m³)")
    print(f"    shape_avg  = {sig['shape_avg']:12.6f}   (shape factor)")


def _print_components(comp: dict):
    print("\n  Thermodynamic components (A − B):")
    print(f"    ΔF_mono  = {comp['dF_mono']:+12.6f}")
    print(f"    ΔF_chain = {comp['dF_chain']:+12.6f}")
    print(f"    ΔF_assoc = {comp['dF_assoc']:+12.6f}")
    print(f"\n  Structural components (log-ratio A/B):")
    print(f"    d_m      = {comp['d_m']:+12.6f}")
    print(f"    d_σ³     = {comp['d_sigma3']:+12.6f}")
    print(f"    d_shape  = {comp['d_shape']:+12.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Parse arguments
    xml_path = None
    args = sys.argv[1:]
    vecs = []
    i = 0
    while i < len(args):
        if args[i] == "--xml":
            xml_path = args[i + 1]
            i += 2
        else:
            vecs.append(args[i])
            i += 1

    if len(vecs) < 2:
        print("Usage: python compute_distance.py <vec_A> <vec_B> [--xml path/to/database.xml]")
        print()
        print("Each vector is a comma-separated list of group counts.")
        print(f"Group order ({len(ss.GROUPS_OF_INTEREST)} groups):")
        for idx, g in enumerate(ss.GROUPS_OF_INTEREST):
            print(f"  [{idx:2d}] {g}")
        print()
        print("Example (NCCO vs CC(N)CO):")
        print("  python compute_distance.py "
              "0,0,0,0,1,0,0,1,0,1,0,0,0,0  "
              "0,0,0,1,0,1,0,1,0,1,0,0,0,0")
        return

    vec_a = [int(x) for x in vecs[0].split(",")]
    vec_b = [int(x) for x in vecs[1].split(",")]

    # Load tables
    print("Loading SAFT pair tables...")
    tables = load_saft_tables(xml_path)
    group_names = tables["group_names"]
    G = len(group_names)

    if len(vec_a) != G or len(vec_b) != G:
        print(f"Error: vectors must have {G} elements "
              f"(got {len(vec_a)} and {len(vec_b)}).")
        print(f"Group order: {group_names}")
        sys.exit(1)

    # Compute all distances
    result = compute_all_distances(vec_a, vec_b, tables)

    # Display
    print("\n" + "=" * 64)
    print("  Distance between two molecules — 4 metrics")
    print("=" * 64)
    print(f"\n  Molecule A:  {_vec_label(vec_a, group_names)}")
    print(f"  Molecule B:  {_vec_label(vec_b, group_names)}")

    _print_signature("Signature A", result["sig_a"])
    _print_signature("Signature B", result["sig_b"])
    _print_components(result["components_thermo"])

    print("\n  ╔══════════════════════════════════════════════╗")
    print(f"  ║  D_thermo    = {result['d_thermo']:10.6f}                ║")
    print(f"  ║  D_struct    = {result['d_struct']:10.6f}                ║")
    print(f"  ║  D_euclidean = {result['d_euclidean']:10.6f}                ║")
    print(f"  ║  D_cosine    = {result['d_cosine']:10.6f}                ║")
    print("  ╚══════════════════════════════════════════════╝")

    print(f"\n  Settings:  T = {tables['T']} K,  η = {tables['eta']}")
    wt = tables["weights_thermo"]
    ws = tables["weights_struct"]
    print(f"  Weights thermo:  w_mono={wt['w_MONO']}  "
          f"w_chain={wt['w_CHAIN']}  w_assoc={wt['w_ASSOC']}")
    print(f"  Weights struct:  w_m={ws['w_M']}  "
          f"w_σ³={ws['w_P']}  w_shape={ws['w_SH']}")


if __name__ == "__main__":
    main()