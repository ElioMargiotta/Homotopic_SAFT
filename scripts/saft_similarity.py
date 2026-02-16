"""
SAFT-γ Mie group-contribution similarity metric  –  Haslam et al. formalism.

Composition-agnostic ranking of candidate molecules by thermodynamic similarity
to a TARGET molecule, using **only** SAFT-γ Mie group parameters (self + cross)
from an XML database.  No full Helmholtz-energy evaluation is performed.

Theoretical basis  (Haslam, Galindo, Jackson et al.)
-----------------------------------------------------
1. Dispersion proxy  –  proportional to the first-order monomer perturbation
   term a₁ of SAFT-γ Mie, evaluated at a reference packing fraction.
2. Association proxy  –  Wertheim TPT1 association strength Δ_{kl}^{ab},
   including the Mayer-f function F, the bonding volume K, and a
   radial-distribution-function integral I(T, ρ).
3. Combining rules  –  strictly follow SAFT-γ Mie:
     σ_{kl}  = (σ_{kk} + σ_{ll}) / 2                       (arithmetic)
     λ_{kl}  = 3 + √[(λ_{kk}-3)(λ_{ll}-3)]                 (nonlinear)
     ε_{kl}  = √(ε_{kk} ε_{ll}) · (σ_{kk}³ σ_{ll}³)^½ / σ_{kl}³
4. Molecule signatures use SAFT segment fractions x_s,k rather than
   intramolecular pair combinatorics.

References
----------
[1] Papaioannou et al., J. Chem. Phys. 140, 054107 (2014).
[2] Dufal et al., J. Chem. Eng. Data 59, 3272 (2014).
[3] Haslam et al. (SAFT-γ Mie review / group-contribution framework).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import numpy as np
import math
import json
from itertools import combinations_with_replacement

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 0 — Constants & configurable settings
# ═══════════════════════════════════════════════════════════════════════════════

T_REF     = 298.15      # Reference temperature  [K]
RHO_REF   = 10000.0     # Reference number density [mol/m³] — liquid-like order
ETA_REF   = 0.40        # Reference packing fraction for a1 evaluation  (≈ liquid)

W_J = 1.0               # Weight for dispersion distance component
W_S = 1.0               # Weight for association distance component
W_M = 1.0               # Weight for chain-length distance component

S0  = 1e-28             # Association floor [m³] to regularise log(0)

LAMBDA_A_DEFAULT = 6.0  # Default attractive exponent when not specified


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — XML database parser
# ═══════════════════════════════════════════════════════════════════════════════

def _pf(text: str | None, default: float = 0.0) -> float:
    """Parse a float from XML text; return *default* on ``None`` or failure."""
    if text is None:
        return default
    try:
        return float(text.strip())
    except Exception:
        return default


def load_database(xml_path: str):
    """
    Parse ``database.xml`` into group and cross-interaction dictionaries.

    Returns
    -------
    groups : dict[str, dict]
        Per-group data: dispersion params, shape factor, number of segments,
        association site multiplicities, self-association interaction list.
    cross  : dict[tuple[str,str], dict]
        Symmetric cross-interaction data keyed by (group1, group2).
        The mirrored entry (g2, g1) swaps site labels so that ``site1``
        always refers to the *first* group in the key.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # ── groups ──────────────────────────────────────────────────────────────
    groups: dict[str, dict] = {}
    for g in root.find("groups").findall("group"):
        name = g.attrib["name"]
        d: dict = {}

        # Segment number ν_k and shape factor S_k
        d["nu"]          = _pf(g.findtext("numberOfSegments"), 0.0)
        d["shapeFactor"] = _pf(g.findtext("shapeFactor"), 0.0)

        # Self-interaction dispersion
        si = g.find("selfInteraction")
        eps = sigma = lrep = latt = 0.0
        self_assoc: list[dict] = []
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
                    self_assoc.append({
                        "site1":        inter.attrib.get("site1", ""),
                        "site2":        inter.attrib.get("site2", ""),
                        "epsilonAssoc": _pf(inter.findtext("epsilonAssoc"), 0.0),
                        "bondingVolume": _pf(inter.findtext("bondingVolume"), 0.0),
                    })

        d["epsilon"]          = eps
        d["sigma"]            = sigma
        d["lambdaRepulsive"]  = lrep
        d["lambdaAttractive"] = latt
        d["self_assoc"]       = self_assoc

        # Association-site multiplicities  {siteName: multiplicity}
        sites: dict[str, float] = {}
        as_el = g.find("associationSites")
        if as_el is not None:
            for sm in as_el.findall("siteMultiplicity"):
                sites[sm.attrib.get("name", "")] = _pf(sm.text, 0.0)
        d["sites"] = sites

        groups[name] = d

    # ── cross-interactions ──────────────────────────────────────────────────
    cross: dict[tuple[str, str], dict] = {}
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

            assoc_list: list[dict] = []
            ab = ci.find("association")
            if ab is not None:
                for inter in ab.findall("interaction"):
                    assoc_list.append({
                        "site1":        inter.attrib.get("site1", ""),
                        "site2":        inter.attrib.get("site2", ""),
                        "epsilonAssoc": _pf(inter.findtext("epsilonAssoc"), 0.0),
                        "bondingVolume": _pf(inter.findtext("bondingVolume"), 0.0),
                    })

            info = {
                "epsilon":          c_eps,
                "sigma":            c_sig,
                "lambdaRepulsive":  c_lrep,
                "lambdaAttractive": c_latt,
                "association":      assoc_list,
            }
            cross[(g1, g2)] = info

            # Mirror: swap site labels so site1 ↔ first key group
            assoc_swapped = [
                {"site1": a["site2"], "site2": a["site1"],
                 "epsilonAssoc": a["epsilonAssoc"],
                 "bondingVolume": a["bondingVolume"]}
                for a in assoc_list
            ]
            cross[(g2, g1)] = {
                "epsilon":          c_eps,
                "sigma":            c_sig,
                "lambdaRepulsive":  c_lrep,
                "lambdaAttractive": c_latt,
                "association":      assoc_swapped,
            }

    return groups, cross


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — SAFT-γ Mie combining rules  (Refs [1–3])
# ═══════════════════════════════════════════════════════════════════════════════

def combining_sigma(sig_kk: float, sig_ll: float) -> float:
    """
    Eq. (σ combining):  σ_{kl} = (σ_{kk} + σ_{ll}) / 2

    Arithmetic mean — standard in SAFT-γ Mie (Ref [1], Eq. 24).
    """
    return (sig_kk + sig_ll) / 2.0


def combining_lambda(lam_kk: float, lam_ll: float) -> float:
    """
    Eq. (λ combining):  λ_{kl} = 3 + √[(λ_{kk} - 3)(λ_{ll} - 3)]

    Nonlinear combining rule for both λ_r and λ_a (Ref [1], Eq. 23).
    Returns LAMBDA_A_DEFAULT (6.0) if both inputs equal 6.0 identically.
    """
    arg = (lam_kk - 3.0) * (lam_ll - 3.0)
    if arg < 0.0:
        return 3.0  # Safeguard for non-physical input
    return 3.0 + math.sqrt(arg)


def combining_epsilon(eps_kk: float, eps_ll: float,
                      sig_kk: float, sig_ll: float,
                      sig_kl: float) -> float:
    """
    Eq. (ε combining):
        ε_{kl} = √(ε_{kk} · ε_{ll}) · (σ_{kk}³ · σ_{ll}³)^{1/2} / σ_{kl}³

    Modified Berthelot rule with σ³ correction (Ref [1], Eq. 25).
    Ensures consistent energy scaling when segment sizes differ.
    """
    if sig_kl == 0.0:
        return 0.0
    return (math.sqrt(abs(eps_kk * eps_ll))
            * math.sqrt(sig_kk**3 * sig_ll**3)
            / sig_kl**3)


def get_pair_params(k: str, l: str, groups: dict, cross: dict):
    """
    Resolve (ε_{kl}, σ_{kl}, λ^r_{kl}, λ^a_{kl}) for the group pair (k, l).

    Priority:
    1. Self-pair (k == l) → use self-interaction directly.
    2. Explicit cross entry exists → use database value for each parameter
       that is non-zero; fall back to combining rule otherwise.
    3. No cross entry → full combining rules.

    Returns
    -------
    tuple  (epsilon, sigma, lambdaR, lambdaA)
    """
    gk, gl = groups[k], groups[l]

    if k == l:
        return (gk["epsilon"], gk["sigma"],
                gk["lambdaRepulsive"], gk["lambdaAttractive"])

    ci = cross.get((k, l))

    # σ  — arithmetic combining rule  (Ref [1], Eq. 24)
    # The database rarely stores cross σ; when it does, honour it.
    if ci is not None and ci["sigma"] != 0.0:
        sig = ci["sigma"]
    else:
        sig = combining_sigma(gk["sigma"], gl["sigma"])

    # λ_r  — nonlinear combining rule  (Ref [1], Eq. 23)
    if ci is not None and ci["lambdaRepulsive"] != 0.0:
        lr = ci["lambdaRepulsive"]
    else:
        lr = combining_lambda(gk["lambdaRepulsive"], gl["lambdaRepulsive"])

    # λ_a  — nonlinear combining rule  (Ref [1], Eq. 23)
    if ci is not None and ci["lambdaAttractive"] != 0.0:
        la = ci["lambdaAttractive"]
    else:
        la = combining_lambda(gk["lambdaAttractive"], gl["lambdaAttractive"])

    # ε  — modified Berthelot with σ³ correction  (Ref [1], Eq. 25)
    if ci is not None and ci["epsilon"] != 0.0:
        eps = ci["epsilon"]
    else:
        eps = combining_epsilon(gk["epsilon"], gl["epsilon"],
                                gk["sigma"], gl["sigma"], sig)

    return eps, sig, lr, la


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3a — Dispersion proxy  (a₁-like cohesion measure)
# ═══════════════════════════════════════════════════════════════════════════════

def mie_prefactor(lr: float, la: float) -> float:
    """
    Mie potential prefactor (Ref [1], Eq. 2):

        C_{kl} = λ^r / (λ^r − λ^a) · (λ^r / λ^a)^{λ^a/(λ^r − λ^a)}

    Returns 0.0 if λ^r ≤ λ^a (degenerate).
    """
    if lr <= la:
        return 0.0
    diff = lr - la
    return (lr / diff) * (lr / la) ** (la / diff)


def _sutherland_a1(lam: float, eta: float) -> float:
    """
    Effective first-order perturbation integral for a Sutherland potential
    of exponent λ, evaluated at packing fraction η.

    Uses the parameterisation from the SAFT-γ Mie paper (Ref [1], Appendix):
        a₁^S(η; λ) ≈ −12η · [ c₁/(λ-3) + c₂(η)/(λ-4) + ... ]

    For a *proxy* (not full EOS) we use the leading-order contact-value
    expression that captures the dominant η and λ dependence:

        a₁^S ≈ − 1/(λ−3) · (1 − η/2) / (1 − η)³

    This is the mean-field / van-der-Waals-1-fluid approximation to the
    Sutherland-λ perturbation integral.

    Ref [1] Eq. (A2) in the low-density / mean-field limit.
    """
    if lam <= 3.0:
        return 0.0
    return -1.0 / (lam - 3.0) * (1.0 - eta / 2.0) / (1.0 - eta) ** 3


def dispersion_a1_proxy(eps: float, sig: float, lr: float, la: float,
                        eta: float = ETA_REF) -> float:
    """
    SAFT-consistent dispersion proxy for the group pair (k, l).

    Proportional to the first-order monomer perturbation term:

        D_{kl} = C_{kl} · ε_{kl} · σ_{kl}³
                 · [ a₁^S(η; λ^a) − a₁^S(η; λ^r) ]

    where a₁^S is the Sutherland-λ integral at packing fraction η.

    The overall sign is positive (attractive), because
    a₁^S(λ^a) < 0 and |a₁^S(λ^a)| > |a₁^S(λ^r)| for λ^a < λ^r.

    We retain the factor σ_{kl}³ so that the proxy carries correct
    dimensions of [energy · volume].

    Ref [1] Eqs. (19)–(20).
    """
    if sig == 0.0 or lr <= 3.0 or la <= 3.0:
        return 0.0

    C = mie_prefactor(lr, la)
    # a1_S returns negative values; the difference (attractive − repulsive)
    # gives the net attraction (positive for well-behaved Mie potentials).
    a1s_att = _sutherland_a1(la, eta)
    a1s_rep = _sutherland_a1(lr, eta)

    return C * eps * sig**3 * (a1s_att - a1s_rep)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3b — Association proxy  (Wertheim TPT1)
# ═══════════════════════════════════════════════════════════════════════════════

def _g_hs_contact(eta: float) -> float:
    """
    Hard-sphere radial distribution function at contact (Carnahan-Starling):

        g^{HS}(σ; η) = (1 − η/2) / (1 − η)³

    Ref [1] Eq. (A7) / standard CS expression.
    """
    if eta >= 1.0:
        return 1e30  # Safeguard
    return (1.0 - eta / 2.0) / (1.0 - eta) ** 3


def _I_assoc(sig_kl: float, eta: float) -> float:
    """
    Association kernel I_{kl}(T, ρ) in the SAFT-γ Mie framework.

    In the full theory (Ref [1], Eq. 37 and Appendix):
        Δ_{kl,ab} = F_{kl,ab} · K_{kl,ab} · I_{kl}

    where I_{kl} ∝ g^{Mie}(σ_{kl}) captures the probability of two
    segments being at bonding distance.  The Mie g is approximated via
    a high-temperature perturbation expansion around the hard-sphere
    reference.  For our *proxy* (no full EOS), we use the leading term:

        I_{kl} ≈ g^{HS}(σ_{kl}; η)

    This is the dominant contribution; higher-order a₁/a₂ corrections
    are state-dependent and require the full Helmholtz machinery.

    The packing fraction η is evaluated at a tuneable liquid-like
    reference point (ETA_REF ≈ 0.4).
    """
    return _g_hs_contact(eta)


def mayer_f(eps_assoc: float, T: float) -> float:
    """
    Mayer-f function for association:

        F_{kl,ab} = exp(ε^{assoc}_{kl,ab} / T) − 1

    Ref [1] Eq. (36).
    """
    if eps_assoc == 0.0:
        return 0.0
    return math.exp(eps_assoc / T) - 1.0


def delta_site_pair(eps_assoc: float, bond_vol: float,
                    sig_kl: float, T: float, eta: float) -> float:
    """
    Association strength for a single site-site interaction:

        Δ_{kl,ab} = F_{kl,ab} · K_{kl,ab} · I_{kl}(T, ρ)

    where
        F = exp(ε^{assoc}/T) − 1          (Mayer-f function)
        K = bondingVolume                   (bonding volume, m³)
        I ≈ g^{HS}(σ_{kl}; η)             (RDF at contact)

    Ref [1] Eqs. (36)–(37).
    """
    if bond_vol == 0.0 or eps_assoc == 0.0:
        return 0.0
    F = mayer_f(eps_assoc, T)
    K = bond_vol
    I = _I_assoc(sig_kl, eta)
    return F * K * I


def delta_pair(k: str, l: str, groups: dict, cross: dict,
               sig_kl: float, T: float = T_REF,
               eta: float = ETA_REF) -> float:
    """
    Total association strength between groups k and l, summed over
    all site-site interactions weighted by site multiplicities:

        Δ_{kl}(T, ρ) = Σ_{a ∈ k} Σ_{b ∈ l}  m_{k,a} · m_{l,b} · Δ_{kl,ab}

    Ref [1] Eq. (38) / standard Wertheim summation.

    Parameters
    ----------
    k, l      : group names
    sig_kl    : cross σ (needed for I_{kl})
    T         : temperature [K]
    eta       : packing fraction
    """
    sites_k = groups[k]["sites"]
    sites_l = groups[l]["sites"]

    if not sites_k or not sites_l:
        return 0.0

    # Select appropriate association interaction list
    if k == l:
        assoc_list = groups[k]["self_assoc"]
    else:
        ci = cross.get((k, l))
        assoc_list = ci["association"] if ci is not None else []

    total = 0.0
    for inter in assoc_list:
        s1 = inter["site1"]    # site on group k
        s2 = inter["site2"]    # site on group l
        ea = inter["epsilonAssoc"]
        bv = inter["bondingVolume"]

        m1 = sites_k.get(s1, 0.0)
        m2 = sites_l.get(s2, 0.0)

        total += m1 * m2 * delta_site_pair(ea, bv, sig_kl, T, eta)

    return total


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — Pre-compute pair tables for GROUPS_OF_INTEREST
# ═══════════════════════════════════════════════════════════════════════════════

def build_pair_tables(group_names: list[str], groups: dict, cross: dict,
                      T: float = T_REF, eta: float = ETA_REF):
    """
    Compute pair-level dispersion and association proxies for every
    unordered pair (k, l) among *group_names*.

    Returns
    -------
    disp_table  : dict[(str,str), float]   D_{kl} = a₁-proxy
    delta_table : dict[(str,str), float]   Δ_{kl}(T, η)
    param_table : dict[(str,str), dict]    {eps, sig, lr, la} resolved params
    """
    disp_table:  dict[tuple[str, str], float] = {}
    delta_table: dict[tuple[str, str], float] = {}
    param_table: dict[tuple[str, str], dict]  = {}

    for k, l in combinations_with_replacement(group_names, 2):
        eps, sig, lr, la = get_pair_params(k, l, groups, cross)

        d_val = dispersion_a1_proxy(eps, sig, lr, la, eta)
        a_val = delta_pair(k, l, groups, cross, sig, T, eta)

        # Store in both orderings for O(1) lookup
        for key in [(k, l), (l, k)]:
            disp_table[key]  = d_val
            delta_table[key] = a_val
            param_table[key] = {"epsilon": eps, "sigma": sig,
                                "lambdaR": lr, "lambdaA": la}

    return disp_table, delta_table, param_table


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — Segment-fraction molecule signature
# ═══════════════════════════════════════════════════════════════════════════════

def segment_fractions(vector, group_names: list[str], groups: dict):
    """
    Compute SAFT segment fractions for a molecule.

    In SAFT-γ Mie the total chain length of molecule i is:

        m_i = Σ_k  n_k · ν_k · S_k

    where n_k is the number of group k, ν_k its number of identical
    segments, and S_k the shape factor.  The segment fraction is:

        x_{s,k} = n_k · ν_k · S_k  /  m_i

    Ref [1] Eqs. (7)–(8).

    Returns
    -------
    xs   : np.ndarray, shape (G,)   segment fractions (sum to 1)
    m_i  : float                     total segment number
    """
    n = np.asarray(vector, dtype=float)
    G = len(group_names)

    # Weighted segment counts:  n_k * ν_k * S_k
    weighted = np.zeros(G)
    for i in range(G):
        if n[i] == 0:
            continue
        gd = groups[group_names[i]]
        weighted[i] = n[i] * gd["nu"] * gd["shapeFactor"]

    m_i = weighted.sum()
    if m_i == 0.0:
        return np.zeros(G), 0.0

    xs = weighted / m_i
    return xs, m_i


def signature(vector, group_names: list[str], groups: dict,
              disp_table: dict, delta_table: dict):
    """
    Compute the SAFT-consistent thermodynamic signature of a molecule.

    Dispersion signature  (proportional to a₁ of the whole molecule):

        D̄ = Σ_k Σ_l  x_{s,k} · x_{s,l} · D_{kl}

    Association signature:

        Ā = Σ_k Σ_l  x_{s,k} · x_{s,l} · Δ_{kl}

    Ref [1] Eqs. (19), (38); the double sum uses segment fractions
    exactly as in the monomer Helmholtz contribution.

    Returns
    -------
    dict  {"D_bar": float, "A_bar": float, "m_total": float}
    """
    xs, m_i = segment_fractions(vector, group_names, groups)

    if m_i == 0.0:
        return {"D_bar": 0.0, "A_bar": 0.0, "m_total": 0.0}

    G = len(group_names)
    D_sum = 0.0
    A_sum = 0.0

    for i in range(G):
        if xs[i] == 0.0:
            continue
        gi = group_names[i]
        for j in range(G):
            if xs[j] == 0.0:
                continue
            gj = group_names[j]
            w = xs[i] * xs[j]
            D_sum += w * disp_table[(gi, gj)]
            A_sum += w * delta_table[(gi, gj)]

    return {"D_bar": D_sum, "A_bar": A_sum, "m_total": m_i}


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — Distance metric & ranking
# ═══════════════════════════════════════════════════════════════════════════════

def log_euclidean_distance(sig_cand: dict, sig_targ: dict,
                           w_J: float = W_J, w_S: float = W_S,
                           w_M: float = W_M, s0: float = S0) -> float:
    """
    Scale-consistent log-relative distance in 3-D signature space:

        d_D = ln( |D̄_c| / |D̄_t| )
        d_A = ln( (Ā_c + S₀) / (Ā_t + S₀) )
        d_m = ln( m_c / m_t )
        D   = √( w_D · d_D² + w_A · d_A² + w_m · d_m² )

    The floor S₀ ≈ 1e-28 m³ prevents singularities when association
    is absent for one or both molecules.  Including the total
    chain length m distinguishes molecules with identical group
    *types* but different *sizes* (e.g. cyclobutane vs cyclohexane).
    """
    D_c = max(abs(sig_cand["D_bar"]), 1e-300)
    D_t = max(abs(sig_targ["D_bar"]), 1e-300)
    dD  = math.log(D_c / D_t)

    A_c = sig_cand["A_bar"]
    A_t = sig_targ["A_bar"]
    dA  = math.log((A_c + s0) / (A_t + s0))

    m_c = max(sig_cand["m_total"], 1e-300)
    m_t = max(sig_targ["m_total"], 1e-300)
    dm  = math.log(m_c / m_t)

    return math.sqrt(w_J * dD**2 + w_S * dA**2 + w_M * dm**2)


def rank_candidates(target_vector, candidate_vectors,
                    group_names: list[str], groups: dict,
                    disp_table: dict, delta_table: dict,
                    w_J: float = W_J, w_S: float = W_S,
                    w_M: float = W_M):
    """
    Rank *candidate_vectors* by proximity to *target_vector*.

    Returns
    -------
    ranking   : list[dict]  sorted by ascending distance
    sig_targ  : dict        target signature
    """
    sig_targ = signature(target_vector, group_names, groups,
                         disp_table, delta_table)

    results = []
    for idx, cv in enumerate(candidate_vectors):
        sig_c = signature(cv, group_names, groups, disp_table, delta_table)
        d = log_euclidean_distance(sig_c, sig_targ, w_J, w_S, w_M)
        results.append({
            "candidate_index":  idx,
            "candidate_vector": list(cv),
            "signature":        sig_c,
            "distance":         d,
        })

    results.sort(key=lambda r: r["distance"])
    return results, sig_targ


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — Utility: load compounds from <compounds> section
# ═══════════════════════════════════════════════════════════════════════════════

def load_compounds(xml_path: str, group_names: list[str]) -> dict[str, list[int]]:
    """
    Read ``<compounds>`` and return {name: group-count vector} aligned
    with *group_names*.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    group_idx = {g: i for i, g in enumerate(group_names)}
    G = len(group_names)

    compounds: dict[str, list[int]] = {}
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


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 8 — Groups of interest & main driver
# ═══════════════════════════════════════════════════════════════════════════════

GROUPS_OF_INTEREST = [
    "NH2_2nd", "NH_2nd", "N_2nd",
    "CH3", "CH2", "CH", "C",
    "CH2OH", "CH2OH_Short",
    "NH2", "NH", "N",
    "OH", "OH_Short",
    "cCH2", "cCH",
    "cNH", "cN",
    "cCHNH", "cCHN",
]


def _print_matrix(table: dict, group_names: list[str], title: str):
    """Pretty-print a pair table as a matrix."""
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")
    header = f"{'':>14s}" + "".join(f"{g:>14s}" for g in group_names)
    print(header)
    for gi in group_names:
        row = f"{gi:>14s}"
        for gj in group_names:
            row += f"{table[(gi, gj)]:14.6e}"
        print(row)


def main():
    import os

    xml_path = os.path.join(os.path.dirname(__file__), "..", "database", "database.xml")
    xml_path = os.path.normpath(xml_path)

    print(f"Loading database from: {xml_path}")
    print(f"T_ref = {T_REF} K,  η_ref = {ETA_REF}")
    groups, cross = load_database(xml_path)

    # Filter to available groups
    available = [g for g in GROUPS_OF_INTEREST if g in groups]
    missing   = [g for g in GROUPS_OF_INTEREST if g not in groups]
    if missing:
        print(f"⚠  Groups not in database (skipped): {missing}")
    group_names = available
    print(f"Using {len(group_names)} groups.\n")

    # ── Build pair tables ──
    disp_table, delta_table, param_table = build_pair_tables(
        group_names, groups, cross, T=T_REF, eta=ETA_REF)

    # ── Print resolved cross-parameter table (ε, σ, λr, λa) ──
    print("\n" + "═"*76)
    print("  Resolved pair parameters (ε_kl, σ_kl, λr_kl, λa_kl)")
    print("═"*76)
    print(f"{'Pair':<30s} {'ε (K)':>12s} {'σ (m)':>14s} {'λ_r':>10s} {'λ_a':>10s}")
    print("─"*76)
    for k, l in combinations_with_replacement(group_names, 2):
        p = param_table[(k, l)]
        print(f"{k+' | '+l:<30s} {p['epsilon']:12.4f} {p['sigma']:14.6e} "
              f"{p['lambdaR']:10.4f} {p['lambdaA']:10.4f}")

    # ── Print dispersion & association pair tables ──
    _print_matrix(disp_table, group_names, "D_{kl}  (a1-proxy dispersion, J·m³)")
    _print_matrix(delta_table, group_names, "Δ_{kl}  (association strength, m³)")

    # ── Load compounds ──
    compounds = load_compounds(xml_path, group_names)
    print(f"\nLoaded {len(compounds)} compounds from database.")
    if not compounds:
        print("No compounds found – exiting.")
        return

    # ── Target and ranking ──
    target_name = "MEA"
    if target_name not in compounds:
        target_name = next(iter(compounds))
    target_vec = compounds[target_name]

    print(f"\n{'═'*60}")
    print(f"  TARGET: {target_name}")
    print(f"  Vector: {target_vec}")
    print(f"{'═'*60}")

    cand_names = [n for n in compounds if n != target_name]
    cand_vecs  = [compounds[n] for n in cand_names]

    ranking, sig_target = rank_candidates(
        target_vec, cand_vecs, group_names, groups,
        disp_table, delta_table, w_J=W_J, w_S=W_S, w_M=W_M)

    print(f"\nTarget signature:  D̄ = {sig_target['D_bar']:.6e},  "
          f"Ā = {sig_target['A_bar']:.6e},  "
          f"m = {sig_target['m_total']:.4f}\n")

    print(f"{'Rank':>4s}  {'Compound':<40s}  {'D̄':>14s}  {'Ā':>14s}  "
          f"{'m':>8s}  {'Distance':>12s}")
    print("─" * 100)
    for rank, entry in enumerate(ranking, 1):
        cname = cand_names[entry["candidate_index"]]
        sig   = entry["signature"]
        dist  = entry["distance"]
        print(f"{rank:4d}  {cname:<40s}  {sig['D_bar']:14.6e}  "
              f"{sig['A_bar']:14.6e}  {sig['m_total']:8.4f}  {dist:12.6f}")

    # ── Export pair tables ──
    D_json = {f"{k[0]}|{k[1]}": v for k, v in disp_table.items()}
    A_json = {f"{k[0]}|{k[1]}": v for k, v in delta_table.items()}

    out_path = os.path.join(os.path.dirname(__file__), "..", "saft_pair_tables.json")
    out_path = os.path.normpath(out_path)
    with open(out_path, "w") as f:
        json.dump({
            "D_kl_dispersion": D_json,
            "Delta_kl_association": A_json,
            "groups": group_names,
            "T_ref_K": T_REF,
            "eta_ref": ETA_REF,
        }, f, indent=2)
    print(f"\nPair tables saved to {out_path}")

    # ── Export ranking ──
    ranking_out = [{
        "compound": cand_names[e["candidate_index"]],
        "candidate_vector": e["candidate_vector"],
        "signature": e["signature"],
        "distance": e["distance"],
    } for e in ranking]

    rank_path = os.path.join(os.path.dirname(__file__), "..",
                             f"ranking_vs_{target_name}.json")
    rank_path = os.path.normpath(rank_path)
    with open(rank_path, "w") as f:
        json.dump({
            "target": target_name,
            "target_vector": target_vec,
            "target_signature": sig_target,
            "settings": {"T_ref_K": T_REF, "eta_ref": ETA_REF,
                         "w_J": W_J, "w_S": W_S, "w_M": W_M, "S0": S0},
            "ranking": ranking_out,
        }, f, indent=2)
    print(f"Ranking saved to {rank_path}")


if __name__ == "__main__":
    main()
