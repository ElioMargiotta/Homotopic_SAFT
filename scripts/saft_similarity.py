"""
SAFT-γ Mie group-contribution similarity metric.

Ranks candidate molecules (represented only by group-count vectors) against a
TARGET molecule by computing physics-derived dispersion (J̄) and association (S̄)
signatures and using a log-Euclidean distance, **without** running a full EOS.

Algorithm
---------
1.  Parse the XML database for self- and cross-interaction parameters.
2.  For every unordered pair (k, l) of GROUPS_OF_INTEREST compute:
      • J_disp_kl  – integrated Mie dispersion-well strength
      • S_HB_kl    – hydrogen-bonding association strength at T_ref
3.  Given a group-count vector  n = [n_1, …, n_G]:
      • count unordered pairs  N_kl
      • average:  J̄ = Σ N_kl·J_kl / N_pairs ,  S̄ = Σ N_kl·S_kl / N_pairs
4.  Distance:
      D = sqrt( w_J·[ln(J̄_cand/J̄_targ)]² + w_S·[ln(S̄_cand/S̄_targ)]² )
    with an ε-floor for zeros.
"""

import xml.etree.ElementTree as ET
import numpy as np
import math
import json
from itertools import combinations_with_replacement

# ═══════════════════════════════════════════════════════════════════════
# 0. Constants & settings
# ═══════════════════════════════════════════════════════════════════════
T_REF = 298.15  # K

# Weights for the distance metric
W_J = 1.0
W_S = 1.0

# Epsilon floor to avoid log(0)
EPS_FLOOR = 1e-300

# Default λ_attractive when not provided in the cross block
LAMBDA_A_DEFAULT = 6.0


# ═══════════════════════════════════════════════════════════════════════
# 1. XML database parser
# ═══════════════════════════════════════════════════════════════════════

def _pf(text, default=0.0):
    """Parse a float from XML text, returning *default* on failure."""
    if text is None:
        return default
    try:
        return float(text.strip())
    except Exception:
        return default


def load_database(xml_path):
    """
    Parse ``database.xml`` and return two dicts.

    Returns
    -------
    groups : dict[str, dict]
        Per-group data including dispersion params, association site
        multiplicities, and self-association interaction list.
    cross  : dict[tuple[str,str], dict]
        Symmetric cross-interaction data keyed by (group1, group2).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # ── groups ──
    groups = {}
    for g in root.find("groups").findall("group"):
        name = g.attrib["name"]
        d = {}

        d["m"]           = _pf(g.findtext("numberOfSegments"), 0.0)
        d["shapeFactor"] = _pf(g.findtext("shapeFactor"), 0.0)

        # Self-interaction: dispersion
        si = g.find("selfInteraction")
        eps = sigma = lrep = latt = 0.0
        self_assoc = []
        if si is not None:
            disp = si.find("dispersion")
            if disp is not None:
                eps   = _pf(disp.findtext("epsilon"), 0.0)
                sigma = _pf(disp.findtext("sigma"), 0.0)
                lrep  = _pf(disp.findtext("lambdaRepulsive"), 0.0)
                latt  = _pf(disp.findtext("lambdaAttractive"), LAMBDA_A_DEFAULT)
            ab = si.find("association")
            if ab is not None:
                for inter in ab.findall("interaction"):
                    s1 = inter.attrib.get("site1", "")
                    s2 = inter.attrib.get("site2", "")
                    ea = _pf(inter.findtext("epsilonAssoc"), 0.0)
                    bv = _pf(inter.findtext("bondingVolume"), 0.0)
                    self_assoc.append({"site1": s1, "site2": s2,
                                       "epsilonAssoc": ea, "bondingVolume": bv})

        d["epsilon"]         = eps
        d["sigma"]           = sigma
        d["lambdaRepulsive"] = lrep
        d["lambdaAttractive"] = latt
        d["self_assoc"]      = self_assoc

        # Association-site multiplicities  {siteName: multiplicity}
        sites = {}
        as_el = g.find("associationSites")
        if as_el is not None:
            for sm in as_el.findall("siteMultiplicity"):
                sname = sm.attrib.get("name", "")
                mult  = _pf(sm.text, 0.0)
                sites[sname] = mult
        d["sites"] = sites

        groups[name] = d

    # ── cross-interactions ──
    cross = {}
    ci_el = root.find("crossInteractions")
    if ci_el is not None:
        for ci in ci_el.findall("crossInteraction"):
            g1 = ci.attrib["group1"]
            g2 = ci.attrib["group2"]

            disp = ci.find("dispersion")
            c_eps  = _pf(disp.findtext("epsilon"), 0.0) if disp is not None else 0.0
            c_lrep = _pf(disp.findtext("lambdaRepulsive"), 0.0) if disp is not None else 0.0
            c_latt = _pf(disp.findtext("lambdaAttractive"), 0.0) if disp is not None else 0.0
            c_sig  = _pf(disp.findtext("sigma"), 0.0) if disp is not None else 0.0

            assoc_list = []
            ab = ci.find("association")
            if ab is not None:
                for inter in ab.findall("interaction"):
                    s1 = inter.attrib.get("site1", "")
                    s2 = inter.attrib.get("site2", "")
                    ea = _pf(inter.findtext("epsilonAssoc"), 0.0)
                    bv = _pf(inter.findtext("bondingVolume"), 0.0)
                    assoc_list.append({"site1": s1, "site2": s2,
                                       "epsilonAssoc": ea, "bondingVolume": bv})

            info = {
                "epsilon": c_eps,
                "sigma": c_sig,
                "lambdaRepulsive": c_lrep,
                "lambdaAttractive": c_latt,
                "association": assoc_list,
            }
            cross[(g1, g2)] = info
            # mirror with swapped site labels
            info_swapped = dict(info)
            assoc_swapped = []
            for inter in assoc_list:
                assoc_swapped.append({
                    "site1": inter["site2"],
                    "site2": inter["site1"],
                    "epsilonAssoc": inter["epsilonAssoc"],
                    "bondingVolume": inter["bondingVolume"],
                })
            info_swapped["association"] = assoc_swapped
            cross[(g2, g1)] = info_swapped

    return groups, cross


# ═══════════════════════════════════════════════════════════════════════
# 2. Combining rules  (used when cross params are missing or zero)
# ═══════════════════════════════════════════════════════════════════════

def _combining_sigma(sig_k, sig_l):
    """Arithmetic mean combining rule for σ."""
    return (sig_k + sig_l) / 2.0


def _combining_lambda_rep(lr_k, lr_l):
    """Arithmetic mean combining rule for λ_rep."""
    return (lr_k + lr_l) / 2.0


def _combining_epsilon(eps_k, eps_l):
    """Geometric mean combining rule for ε (configurable)."""
    return math.sqrt(abs(eps_k * eps_l))


def get_pair_params(k, l, groups, cross):
    """
    Return (epsilon, sigma, lambdaR, lambdaA) for the pair (k, l),
    applying combining rules where explicit cross values are absent.
    """
    gk, gl = groups[k], groups[l]

    if k == l:
        # Self pair – use self-interaction directly
        return (gk["epsilon"], gk["sigma"],
                gk["lambdaRepulsive"], gk["lambdaAttractive"])

    ci = cross.get((k, l))

    # epsilon
    if ci is not None and ci["epsilon"] != 0.0:
        eps = ci["epsilon"]
    else:
        eps = _combining_epsilon(gk["epsilon"], gl["epsilon"])

    # sigma
    if ci is not None and ci["sigma"] != 0.0:
        sig = ci["sigma"]
    else:
        sig = _combining_sigma(gk["sigma"], gl["sigma"])

    # lambdaRepulsive
    if ci is not None and ci["lambdaRepulsive"] != 0.0:
        lr = ci["lambdaRepulsive"]
    else:
        lr = _combining_lambda_rep(gk["lambdaRepulsive"], gl["lambdaRepulsive"])

    # lambdaAttractive
    if ci is not None and ci["lambdaAttractive"] != 0.0:
        la = ci["lambdaAttractive"]
    else:
        la = LAMBDA_A_DEFAULT

    return eps, sig, lr, la


# ═══════════════════════════════════════════════════════════════════════
# 3. Derived pair scalars:  J_disp  &  S_HB
# ═══════════════════════════════════════════════════════════════════════

def mie_prefactor(lr, la):
    """
    C_kl = (λR / (λR - λA)) · (λR / λA)^(λA / (λR - λA))

    Returns 0.0 when the exponent is degenerate (λR == λA).
    """
    if lr <= la:
        return 0.0
    diff = lr - la
    return (lr / diff) * (lr / la) ** (la / diff)


def J_disp(eps, sig, lr, la):
    """
    Integrated Mie-dispersion well strength:

        J = 4π · C · ε · σ³ · [1/(λA-3) − 1/(λR-3)]

    Returns 0.0 when parameters are non-physical (λ ≤ 3).
    """
    if lr <= 3.0 or la <= 3.0 or sig == 0.0:
        return 0.0
    C = mie_prefactor(lr, la)
    return 4.0 * math.pi * C * eps * sig**3 * (1.0/(la - 3.0) - 1.0/(lr - 3.0))


def S_HB_site_pair(eps_assoc, bond_vol, T):
    """
    Association strength for one site-pair:
        S = K · (exp(εᵃ / T) − 1)
    where K ≡ bondingVolume.
    """
    if bond_vol == 0.0 or eps_assoc == 0.0:
        return 0.0
    return bond_vol * (math.exp(eps_assoc / T) - 1.0)


def S_HB_pair(k, l, groups, cross, T=T_REF):
    """
    Total association strength between groups k and l:

        S_HB_kl = Σ_{a ∈ k} Σ_{b ∈ l}  m_{k,a} · m_{l,b} · S_ab(T)

    where the sum runs over every association interaction entry that
    matches sites on group k and group l, weighted by site multiplicities.
    """
    sites_k = groups[k]["sites"]
    sites_l = groups[l]["sites"]

    if not sites_k or not sites_l:
        return 0.0

    # Collect relevant association interactions
    if k == l:
        assoc_list = groups[k]["self_assoc"]
    else:
        ci = cross.get((k, l))
        assoc_list = ci["association"] if ci is not None else []

    total = 0.0
    for inter in assoc_list:
        s1 = inter["site1"]
        s2 = inter["site2"]
        ea = inter["epsilonAssoc"]
        bv = inter["bondingVolume"]

        # site1 belongs to the first group in the pair key, site2 to the second.
        # For self-pairs k==l this is the same group.
        # Need to handle both orderings because cross dict may store
        # the interaction with group-order swapped.
        m1 = sites_k.get(s1, 0.0)
        m2 = sites_l.get(s2, 0.0)

        total += m1 * m2 * S_HB_site_pair(ea, bv, T)

    return total


# ═══════════════════════════════════════════════════════════════════════
# 4. Pre-compute pair tables for GROUPS_OF_INTEREST
# ═══════════════════════════════════════════════════════════════════════

def build_pair_tables(group_names, groups, cross, T=T_REF):
    """
    Compute J_disp_kl and S_HB_kl for every unordered pair (k, l)
    among *group_names*.

    Returns
    -------
    J_table : dict[(str,str), float]
    S_table : dict[(str,str), float]
        Both keyed by (k, l) with k <= l (lexicographic) for unique storage,
        but retrievable in either order via helper.
    """
    J_table = {}
    S_table = {}

    for k, l in combinations_with_replacement(group_names, 2):
        eps, sig, lr, la = get_pair_params(k, l, groups, cross)
        j_val = J_disp(eps, sig, lr, la)
        s_val = S_HB_pair(k, l, groups, cross, T)

        # Store in both orderings for easy lookup
        J_table[(k, l)] = j_val
        J_table[(l, k)] = j_val
        S_table[(k, l)] = s_val
        S_table[(l, k)] = s_val

    return J_table, S_table


# ═══════════════════════════════════════════════════════════════════════
# 5. Signature of a group-count vector
# ═══════════════════════════════════════════════════════════════════════

def signature(vector, group_names, J_table, S_table):
    """
    Compute the thermodynamic signature {J̄, S̄} of a molecule
    described by its group-count *vector*.

    Parameters
    ----------
    vector : array-like of int
        Counts aligned with *group_names*.
    group_names : list[str]
    J_table, S_table : pair-table dicts from ``build_pair_tables``.

    Returns
    -------
    dict  {"J_bar": float, "S_bar": float}
    """
    n = np.asarray(vector, dtype=float)
    N = n.sum()
    if N < 2:
        return {"J_bar": 0.0, "S_bar": 0.0}

    N_pairs = N * (N - 1) / 2.0
    G = len(group_names)

    J_sum = 0.0
    S_sum = 0.0
    for i in range(G):
        if n[i] == 0:
            continue
        for j in range(i, G):
            if n[j] == 0:
                continue
            # Number of unordered pairs
            if i == j:
                N_kl = n[i] * (n[i] - 1) / 2.0
            else:
                N_kl = n[i] * n[j]

            if N_kl == 0:
                continue

            gi = group_names[i]
            gj = group_names[j]
            J_sum += N_kl * J_table[(gi, gj)]
            S_sum += N_kl * S_table[(gi, gj)]

    J_bar = J_sum / N_pairs
    S_bar = S_sum / N_pairs
    return {"J_bar": J_bar, "S_bar": S_bar}


# ═══════════════════════════════════════════════════════════════════════
# 6. Distance & ranking
# ═══════════════════════════════════════════════════════════════════════

def log_euclidean_distance(sig_cand, sig_targ, w_J=W_J, w_S=W_S):
    """
    D = sqrt( w_J·[ln(J̄_c / J̄_t)]² + w_S·[ln(S̄_c / S̄_t)]² )

    If either S̄ is zero, fall back to absolute difference on S̄.
    """
    J_c = max(sig_cand["J_bar"], EPS_FLOOR)
    J_t = max(sig_targ["J_bar"], EPS_FLOOR)

    dJ = math.log(J_c / J_t)

    S0 = 1e-27  # tune once

    S_c = sig_cand["S_bar"]
    S_t = sig_targ["S_bar"]

    if S_t == 0.0 or S_c == 0.0:
        # Fallback: use absolute difference scaled by a characteristic value
        # so the metric stays meaningful even when one molecule has no HB.
        dS = math.log((S_c + S0) / (S_t + S0))
    else:
        dS = math.log(S_c / S_t)

    return math.sqrt(w_J * dJ**2 + w_S * dS**2)


def rank_candidates(target_vector, candidate_vectors, group_names,
                    J_table, S_table, w_J=W_J, w_S=W_S):
    """
    Rank *candidate_vectors* by proximity to *target_vector*.

    Returns
    -------
    list[dict]  sorted by distance (ascending).
        Each dict: {candidate_index, candidate_vector, signature, distance}
    """
    sig_targ = signature(target_vector, group_names, J_table, S_table)

    results = []
    for idx, cv in enumerate(candidate_vectors):
        sig_c = signature(cv, group_names, J_table, S_table)
        d = log_euclidean_distance(sig_c, sig_targ, w_J, w_S)
        results.append({
            "candidate_index": idx,
            "candidate_vector": list(cv),
            "signature": sig_c,
            "distance": d,
        })

    results.sort(key=lambda r: r["distance"])
    return results, sig_targ


# ═══════════════════════════════════════════════════════════════════════
# 7. Helper: read compound vectors from the XML <compounds> section
# ═══════════════════════════════════════════════════════════════════════

def load_compounds(xml_path, group_names):
    """
    Read the ``<compounds>`` section and return a dict of
    {compound_name: group-count vector} aligned with *group_names*.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    group_idx = {g: i for i, g in enumerate(group_names)}
    G = len(group_names)

    compounds = {}
    comp_el = root.find("compounds")
    if comp_el is None:
        return compounds

    for c in comp_el.findall("compound"):
        name = c.attrib["name"]
        vec = [0] * G
        gm_el = c.find("groupMultiplicities")
        if gm_el is None:
            continue
        for gm in gm_el.findall("groupMultiplicity"):
            ref = gm.attrib.get("ref", "")
            count = int(float(gm.text.strip()))
            if ref in group_idx:
                vec[group_idx[ref]] += count
        compounds[name] = vec

    return compounds


# ═══════════════════════════════════════════════════════════════════════
# 8. Example / main
# ═══════════════════════════════════════════════════════════════════════

GROUPS_OF_INTEREST = [
    "NH2_2nd",
    "NH_2nd",
    "N_2nd",
    "CH3",
    "CH2",
    "CH",
    "C",
    "CH2OH",
    "CH2OH_Short",
    "NH2",
    "NH",
    "N",
    "OH",
    "OH_Short",
    "cCH2",
    "cCH",
    "cNH",
    "cN",
    "cCHNH",
    "cCHN",
]


def print_pair_tables(J_table, S_table, group_names):
    """Pretty-print the pair tables as matrices."""
    G = len(group_names)
    print("\n╔══════════════════════════════════════════════════╗")
    print("║        J_disp pair table (selected entries)      ║")
    print("╚══════════════════════════════════════════════════╝")
    # Header
    header = f"{'':>14s}" + "".join(f"{g:>14s}" for g in group_names)
    print(header)
    for i, gi in enumerate(group_names):
        row = f"{gi:>14s}"
        for j, gj in enumerate(group_names):
            row += f"{J_table[(gi, gj)]:14.6e}"
        print(row)

    print("\n╔══════════════════════════════════════════════════╗")
    print("║        S_HB pair table  (selected entries)       ║")
    print("╚══════════════════════════════════════════════════╝")
    print(header)
    for i, gi in enumerate(group_names):
        row = f"{gi:>14s}"
        for j, gj in enumerate(group_names):
            row += f"{S_table[(gi, gj)]:14.6e}"
        print(row)


def main():
    import os

    xml_path = os.path.join(os.path.dirname(__file__), "..", "database", "database.xml")
    xml_path = os.path.normpath(xml_path)

    print(f"Loading database from: {xml_path}")
    groups, cross = load_database(xml_path)

    # Check which groups of interest are present in the database
    available = [g for g in GROUPS_OF_INTEREST if g in groups]
    missing   = [g for g in GROUPS_OF_INTEREST if g not in groups]
    if missing:
        print(f"⚠  Groups not found in database (skipped): {missing}")
    group_names = available

    print(f"Using {len(group_names)} groups: {group_names}\n")

    # ── Build pair tables ──
    J_table, S_table = build_pair_tables(group_names, groups, cross, T=T_REF)

    # ── Show a compact view of the pair tables ──
    print_pair_tables(J_table, S_table, group_names)

    # ── Load compounds from the XML ──
    compounds = load_compounds(xml_path, group_names)
    print(f"\nLoaded {len(compounds)} compounds from database.")

    if not compounds:
        print("No compounds found – exiting.")
        return

    # ── Pick a target (e.g. MEA) and rank candidates ──
    target_name = "MEA"
    if target_name not in compounds:
        # Fall back to first compound
        target_name = next(iter(compounds))
    target_vec = compounds[target_name]

    print(f"\n═══ TARGET: {target_name}  vector = {target_vec} ═══")

    # Build candidate list (all compounds except target)
    cand_names = [n for n in compounds if n != target_name]
    cand_vecs  = [compounds[n] for n in cand_names]

    ranking, sig_target = rank_candidates(
        target_vec, cand_vecs, group_names, J_table, S_table,
        w_J=W_J, w_S=W_S,
    )

    print(f"\nTarget signature:  J̄ = {sig_target['J_bar']:.6e},  S̄ = {sig_target['S_bar']:.6e}\n")
    print(f"{'Rank':>4s}  {'Compound':<40s}  {'J̄':>14s}  {'S̄':>14s}  {'Distance':>12s}")
    print("─" * 90)
    for rank, entry in enumerate(ranking, 1):
        cname = cand_names[entry["candidate_index"]]
        sig   = entry["signature"]
        dist  = entry["distance"]
        print(f"{rank:4d}  {cname:<40s}  {sig['J_bar']:14.6e}  {sig['S_bar']:14.6e}  {dist:12.6f}")

    # ── Export pair tables as JSON-serializable dicts ──
    J_json = {f"{k[0]}|{k[1]}": v for k, v in J_table.items()}
    S_json = {f"{k[0]}|{k[1]}": v for k, v in S_table.items()}

    out_path = os.path.join(os.path.dirname(__file__), "..", "saft_pair_tables.json")
    out_path = os.path.normpath(out_path)
    with open(out_path, "w") as f:
        json.dump({"J_disp": J_json, "S_HB": S_json,
                    "groups": group_names, "T_ref_K": T_REF}, f, indent=2)
    print(f"\nPair tables saved to {out_path}")

    # ── Export ranking as JSON ──
    ranking_out = []
    for entry in ranking:
        ranking_out.append({
            "compound": cand_names[entry["candidate_index"]],
            "candidate_vector": entry["candidate_vector"],
            "signature": entry["signature"],
            "distance": entry["distance"],
        })

    rank_path = os.path.join(os.path.dirname(__file__), "..", f"ranking_vs_{target_name}.json")
    rank_path = os.path.normpath(rank_path)
    with open(rank_path, "w") as f:
        json.dump({
            "target": target_name,
            "target_vector": target_vec,
            "target_signature": sig_target,
            "settings": {"T_ref_K": T_REF, "w_J": W_J, "w_S": W_S},
            "ranking": ranking_out,
        }, f, indent=2)
    print(f"Ranking saved to {rank_path}")


if __name__ == "__main__":
    main()
