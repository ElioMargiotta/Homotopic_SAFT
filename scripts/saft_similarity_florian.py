"""
SAFT-γ Mie group-contribution similarity metric  –  Haslam et al. formalism.

Composition-agnostic ranking of candidate molecules by thermodynamic similarity
to a TARGET molecule, using **only** SAFT-γ Mie group parameters (self + cross)
from an XML database.  No full Helmholtz-energy evaluation is performed.

Theoretical basis  (Haslam, Galindo, Jackson et al.)
-----------------------------------------------------
1. Monomer free-energy proxy  –  A^mono/NkBT ≈ m·a^HS + (m/kBT)·a₁
   where a^HS is the Carnahan–Starling hard-sphere free energy per segment,
   and a₁ is the first-order perturbation term (Barker–Henderson) with
   the Sutherland integral evaluated at reference packing fraction via
   the SAFT-γ Mie effective-packing-fraction parameterisation.  The
   effective hard-sphere diameter d_{kk} is computed by Gauss–Legendre
   quadrature of the Barker–Henderson integral.
   Second-order (a₂) and higher terms are omitted.
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
[4] Lafitte et al., J. Chem. Phys. 139, 154504 (2013).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import numpy as np
import math
import json
import csv
from itertools import combinations_with_replacement

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 0 — Constants & configurable settings
# ═══════════════════════════════════════════════════════════════════════════════

KB        = 1.380649e-23  # Boltzmann constant  [J/K]
T_REF     = 298.15        # Reference temperature  [K]
ETA_REF   = 0.40          # Reference packing fraction  (≈ liquid state)

W_MONO  = 1.0            # Weight for monomer free-energy distance component
W_CHAIN = 1.0            # Weight for chain free-energy distance component
W_ASSOC = 1.0            # Weight for association free-energy distance component

W_M     = 1.0            # Weight for chain-length structural component
W_P     = 1.0            # Weight for packing (sigma^3) structural component
W_SH    = 1.0            # Weight for shape-factor structural component

LAMBDA_A_DEFAULT = 6.0   # Default attractive exponent when not specified

# Effective packing-fraction parameterisation coefficients
# (Papaioannou et al. [1] Eq. 27, from Lafitte et al. [4] Table)
# ζ_eff(ζ_x; λ) = c1·ζ_x + c2·ζ_x² + c3·ζ_x³ + c4·ζ_x⁴
# where (c1,c2,c3,c4) = M · (1, 1/λ, 1/λ², 1/λ³)^T
_ZETA_EFF_MATRIX = np.array([
    [ 0.81096,   1.7888, -37.578,  92.284],
    [ 1.0205,  -19.341,  151.26, -463.50],
    [-1.9057,   22.845, -228.14,  973.92],
    [ 1.0885,   -6.1962, 106.98, -677.64],
])


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
                    s1 = inter.attrib.get("site1", "")
                    s2 = inter.attrib.get("site2", "")
                    # Cross interactions require complementary site types;
                    # skip pairs where both sites have the same canonical type.
                    if not _is_valid_cross_site_pair(s1, s2):
                        continue
                    assoc_list.append({
                        "site1":        s1,
                        "site2":        s2,
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
# MODULE 2 — SAFT-γ Mie combining rules  (Refs [1–3], incl. association)
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


# Site-type aliases: map variant names to a canonical type so that
# groups using different labels for the same physical site (e.g. "e1" vs "e")
# can be matched when applying combining rules.
_SITE_CANONICAL = {
    "e":  "e",
    "e1": "e",
    "e2": "e",
    "H":  "H",
    "a1": "a",
    "a2": "a",
}


def _canonical_site(name: str) -> str:
    """Return canonical site type for *name*, or *name* itself if unknown."""
    return _SITE_CANONICAL.get(name, name)


def _is_valid_cross_site_pair(site1: str, site2: str) -> bool:
    """Return True if *site1* and *site2* have **different** canonical types.

    Cross association interactions only exist between complementary sites
    (e.g. an electron-donor 'e' with a hydrogen-donor 'H').  Two sites
    of the same canonical type (e.g. 'e' with 'e1', or 'H' with 'H')
    cannot form a cross-association bond.
    """
    return _canonical_site(site1) != _canonical_site(site2)


def combining_eps_assoc(ea_kk: float, ea_ll: float) -> float:
    """
    SAFT-γ Mie combining rule for cross association energy:

        ε^{assoc}_{kl,ab} = √( ε^{assoc}_{kk,aa} · ε^{assoc}_{ll,bb} )

    Geometric mean — consistent with the standard SAFT-γ Mie framework.
    Symmetric: ε^{assoc}_{kl,ab} = ε^{assoc}_{lk,ba}.
    """
    return math.sqrt(abs(ea_kk * ea_ll))


def combining_bond_vol(kv_kk: float, kv_ll: float) -> float:
    """
    SAFT-γ Mie combining rule for cross bonding volume:

        K_{kl,ab} = [ ( ∛K_{kk,aa} + ∛K_{ll,bb} ) / 2 ]³

    Cubic mean of cube roots — accounts for the volumetric nature of
    the bonding parameter.
    Symmetric: K_{kl,ab} = K_{lk,ba}.
    """
    if kv_kk == 0.0 and kv_ll == 0.0:
        return 0.0
    cbrt_k = math.copysign(abs(kv_kk) ** (1.0 / 3.0), kv_kk)
    cbrt_l = math.copysign(abs(kv_ll) ** (1.0 / 3.0), kv_ll)
    avg = (cbrt_k + cbrt_l) / 2.0
    return avg ** 3


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
# MODULE 3a — Monomer free-energy proxy  (a^HS + a₁ perturbation)
#
# The monomer Helmholtz contribution is (Papaioannou Eq. 9, truncated):
#
#   A^mono / NkBT  ≈  m_i · a^HS  +  (m_i / kBT) · a₁
#
# where a^HS is the hard-sphere free energy per segment (BMCSL, Eq. 12),
# and a₁ is the first-order mean-attractive perturbation per segment
# (Eq. 17–18), with pair contributions a_{1,kl} (Eq. 19).
#
# Higher-order terms (a₂, a₃) are omitted — they are corrections to a₁.
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


def _effective_hs_diameter(sig: float, eps: float, lr: float, la: float,
                            T: float) -> float:
    """
    Barker–Henderson effective hard-sphere diameter (Ref [1], Eq. 10):

        d_{kk} = ∫₀^{σ} [1 − exp(−u^Mie(r)/kBT)] dr

    Evaluated by 5-point Gauss–Legendre quadrature on [0, σ], following
    Paricaud (2006) as recommended in the SAFT-VR Mie framework.

    For cross pairs, d_{kl} = (d_{kk} + d_{ll}) / 2  (arithmetic mean).
    """
    if sig == 0.0 or eps == 0.0 or lr <= la:
        return sig  # fall back to σ when potential is degenerate

    C = mie_prefactor(lr, la)

    # 5-point Gauss–Legendre nodes and weights on [0, 1]
    nodes = np.array([
        0.04691007703067,
        0.23076534494716,
        0.50000000000000,
        0.76923465505284,
        0.95308992296933,
    ])
    weights = np.array([
        0.11846344252810,
        0.23931433524968,
        0.28444444444444,
        0.23931433524968,
        0.11846344252810,
    ])

    # Map to [0, σ]:  r = σ · t,  dr = σ · dt
    # Note: eps is in Kelvin (= ε/kB from the database), so
    #   u^Mie(r) / kBT  =  C · (ε/kB) · [ (σ/r)^λr − (σ/r)^λa ] / T
    d = 0.0
    for t, w in zip(nodes, weights):
        if t == 0.0:
            d += w  # integrand → 1 as r → 0  (potential → +∞)
            continue
        ratio = 1.0 / t   # σ / r  where r = σ·t
        beta_u = C * eps * (ratio**lr - ratio**la) / T  # u^Mie / kBT
        if beta_u > 500.0:
            integrand = 1.0   # exp(−u/kBT) → 0 for large repulsion
        elif beta_u < -500.0:
            integrand = 0.0
        else:
            integrand = 1.0 - math.exp(-beta_u)
        d += w * integrand

    return sig * d


def _zeta_eff_coeffs(lam: float) -> tuple[float, float, float, float]:
    """
    Effective packing-fraction polynomial coefficients for exponent λ.

    (c1, c2, c3, c4) = M · (1, 1/λ, 1/λ², 1/λ³)^T

    Ref [1] Eq. 27.
    """
    lam_vec = np.array([1.0, 1.0/lam, 1.0/lam**2, 1.0/lam**3])
    c = _ZETA_EFF_MATRIX @ lam_vec
    return tuple(c)


def _zeta_eff(zeta_x: float, lam: float) -> float:
    """
    Effective packing fraction ζ_eff(ζ_x; λ) (Ref [1] Eq. 26):

        ζ_eff = c1·ζ_x + c2·ζ_x² + c3·ζ_x³ + c4·ζ_x⁴
    """
    c1, c2, c3, c4 = _zeta_eff_coeffs(lam)
    z = zeta_x
    return c1*z + c2*z**2 + c3*z**3 + c4*z**4


def _sutherland_a1s(eps_kl: float, d_kl: float, lam: float,
                    rho_s: float, zeta_x: float) -> float:
    """
    Sutherland first-order perturbation integral per segment
    (Ref [1] Eq. 25):

        a₁ˢ(ρ_s; λ) = −2π ρ_s · (ε_{kl} d³_{kl}) / (λ − 3)
                       · (1 − ζ_eff/2) / (1 − ζ_eff)³

    Note: ε_{kl} here is in Kelvin (database convention); the factor kB
    is applied at the molecule level when forming A₁/NkBT = m·a₁/kBT.

    The effective packing fraction ζ_eff is obtained from the polynomial
    parameterisation (Eq. 26–27) evaluated at the vdW one-fluid ζ_x.

    Returns a₁ˢ in units of [K·m³]  (energy-volume per segment).
    """
    if lam <= 3.0 or d_kl == 0.0:
        return 0.0
    ze = _zeta_eff(zeta_x, lam)
    if ze >= 1.0:
        ze = 0.9999  # safeguard
    cs = (1.0 - ze / 2.0) / (1.0 - ze)**3
    return -2.0 * math.pi * rho_s * eps_kl * d_kl**3 / (lam - 3.0) * cs


def _B_kl(eps_kl: float, d_kl: float, lam: float,
           rho_s: float, zeta_x: float) -> float:
    """
    Residual of the first-order perturbation beyond the Sutherland
    integral (Ref [1] Eq. 20):

        B_{kl}(ρ_s; λ) = 2π ρ_s d³_{kl} ε_{kl}
                         · [ (1 − ζ_x/2)/(1−ζ_x)³ · I(λ)
                            − 9ζ_x(1+ζ_x) / (2(1−ζ_x)³) · J(λ) ]

    where I(λ) and J(λ) depend on x₀ = σ/d  (Ref [1] Eqs. 23–24).
    Since this function is called for a specific pair, we pass x0 via
    the pair-level wrapper.

    For the *proxy* we omit the B term (it is a correction that partly
    cancels between attractive and repulsive branches). Setting B=0 is
    equivalent to the mean-field / Sutherland-only approximation and is
    consistent with the proxy philosophy.

    Returns 0.0.
    """
    return 0.0


def _a1_pair(eps_kl: float, sig_kl: float, d_kl: float,
             lr: float, la: float,
             rho_s: float, zeta_x: float) -> float:
    """
    First-order perturbation pair term a_{1,kl} (Ref [1] Eq. 19):

        a_{1,kl} = C_{kl} · [ x₀^{λᵃ} · (a₁ˢ(λᵃ) + B(λᵃ))
                              − x₀^{λʳ} · (a₁ˢ(λʳ) + B(λʳ)) ]

    where x₀ = σ_{kl} / d_{kl}.

    Returns a_{1,kl} in units of [K·m³].
    """
    if sig_kl == 0.0 or d_kl == 0.0 or lr <= la:
        return 0.0

    C = mie_prefactor(lr, la)
    x0 = sig_kl / d_kl

    a1s_a = _sutherland_a1s(eps_kl, d_kl, la, rho_s, zeta_x)
    a1s_r = _sutherland_a1s(eps_kl, d_kl, lr, rho_s, zeta_x)
    B_a   = _B_kl(eps_kl, d_kl, la, rho_s, zeta_x)
    B_r   = _B_kl(eps_kl, d_kl, lr, rho_s, zeta_x)

    return C * (x0**la * (a1s_a + B_a) - x0**lr * (a1s_r + B_r))


def _a_hs_pure(eta: float) -> float:
    """
    Hard-sphere Helmholtz free energy per segment for a pure fluid
    (Carnahan–Starling, Ref [1] Eq. 12 reduced to the one-component case):

        a^HS = (4η − 3η²) / (1 − η)²

    This is the standard CS expression A^HS/(N_seg · kBT).
    For the multi-component BMCSL form used in the full theory, we would
    need all ζ_m moment densities.  In the proxy we use the one-fluid
    approximation with packing fraction η = ζ_x (Eq. 22).

    Returns dimensionless a^HS.
    """
    if eta >= 1.0:
        eta = 0.9999
    if eta <= 0.0:
        return 0.0
    return (4.0 * eta - 3.0 * eta**2) / (1.0 - eta)**2


def compute_monomer_proxy_pair(eps: float, sig: float, lr: float, la: float,
                                T: float = T_REF, eta: float = ETA_REF):
    """
    Compute the pair-level ingredients for the monomer free-energy proxy.

    Returns
    -------
    d_kl      : effective HS diameter [m]
    a1_kl     : first-order perturbation per segment [K·m³]

    These are stored in pair tables.  The molecule-level monomer proxy
    is assembled in ``signature()`` as:

        F^mono_i / NkBT  =  m_i · a^HS(η)  +  (m_i / kBT) · a₁

    where a₁ = Σ_k Σ_l x_{s,k} x_{s,l} a_{1,kl}.
    """
    # Effective HS diameter (self-pair: Gauss–Legendre; cross: arithmetic)
    d = _effective_hs_diameter(sig, eps, lr, la, T)

    # Segment number density from packing fraction:
    #   η = (π/6) ρ_s d³   →   ρ_s = 6η / (π d³)
    if d == 0.0:
        return 0.0, 0.0
    rho_s = 6.0 * eta / (math.pi * d**3)
    zeta_x = eta  # one-fluid approximation at reference state

    a1 = _a1_pair(eps, sig, d, lr, la, rho_s, zeta_x)

    return d, a1


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3c — Chain free energy  (Wertheim TPT1)
#
# The chain contribution to A/NkBT is (Papaioannou Eq. 46):
#
#   A^chain / NkBT  =  -(m_i - 1) · ln g^Mie(σ̄_ii; ζ_x)
#
# where g^Mie is the RDF of the hypothetical one-fluid Mie system
# evaluated at the effective molecular diameter σ̄_ii and packing
# fraction ζ_x.
#
# For the proxy, we approximate g^Mie by its leading (zeroth-order)
# term g^HS_d (the HS RDF at σ̄ evaluated with HS diameter d̄), using
# the Boublík recipe (Eq. 48–52).  This is consistent with truncating
# the monomer term at first order.
#
# The molecular-averaged parameters (σ̄, d̄, ε̄, λ̄) are obtained from
# the segment-fraction-weighted mixing rules (Eqs. 40–45):
#
#   σ̄³  = Σ_k Σ_l  z_k z_l σ³_kl       (vdW one-fluid)
#   d̄³  = Σ_k Σ_l  z_k z_l d³_kl
#   ε̄   = Σ_k Σ_l  z_k z_l ε_kl σ³_kl / σ̄³
#   λ̄   = Σ_k Σ_l  z_k z_l λ_kl
#
# where z_{k,i} ∝ x_{s,k} σ³_{kk} / Σ_l x_{s,l} σ³_{ll}  (Eq. 41).
# ═══════════════════════════════════════════════════════════════════════════════

def _g_HS_boublik(x0: float, zeta_x: float) -> float:
    """
    Hard-sphere radial distribution function at distance σ̄, evaluated
    for a one-fluid system of HS diameter d̄ at packing fraction ζ_x.

    Uses Boublík's exponential form (Ref [1] Eq. 48):

        g^HS_d(σ̄) = exp( k0 + k1·x̄0 + k2·x̄0² + k3·x̄0³ )

    where x̄0 = σ̄/d̄  (≥ 1, typically ∈ [1.00, 1.05]).

    Coefficients k0..k3 are functions of ζ_x (Eqs. 49–52).

    Parameters
    ----------
    x0     : σ̄ / d̄  (must be ≥ 1)
    zeta_x : vdW one-fluid packing fraction

    Returns
    -------
    float : g^HS_d(σ̄; ζ_x)
    """
    if zeta_x >= 1.0:
        zeta_x = 0.9999
    if zeta_x <= 0.0:
        return 1.0

    z  = zeta_x
    z2 = z * z
    z3 = z2 * z
    z4 = z3 * z
    denom3 = (1.0 - z) ** 3
    denom2 = (1.0 - z) ** 2

    # Eq. 49
    k0 = -math.log(1.0 - z) + (42.0*z - 39.0*z2 + 9.0*z3 - 2.0*z4) / (6.0 * denom3)
    # Eq. 50
    k1 = (z4 + 6.0*z2 - 12.0*z) / (2.0 * denom3)
    # Eq. 51
    k2 = -3.0 * z2 / (8.0 * denom2)
    # Eq. 52
    k3 = (-z4 + 3.0*z2 + 3.0*z) / (6.0 * denom3)

    exponent = k0 + k1 * x0 + k2 * x0**2 + k3 * x0**3

    # Guard against overflow
    if exponent > 500.0:
        return math.exp(500.0)
    return math.exp(exponent)


def chain_free_energy(m_i: float, xs: np.ndarray,
                      group_names: list[str], groups: dict,
                      param_table: dict,
                      eta: float = ETA_REF) -> float:
    """
    Chain contribution to the Helmholtz free energy (Ref [1] Eq. 46):

        A^chain / NkBT  =  -(m_i - 1) · ln g^Mie(σ̄; ζ_x)

    where g^Mie ≈ g^HS_d(σ̄; ζ_x) at zeroth order (Boublík, Eq. 48).

    The molecular-averaged parameters σ̄ and d̄ are obtained from the
    vdW one-fluid mixing rules (Eqs. 40–45).

    Parameters
    ----------
    m_i         : total chain length (Σ n_k ν_k S_k)
    xs          : segment fractions, shape (G,)
    group_names : list of group names
    groups      : group data dict
    param_table : resolved pair parameters (sigma, d_kl, etc.)
    eta         : reference packing fraction

    Returns
    -------
    float : A^chain / NkBT  (dimensionless, typically negative)
    """
    if m_i <= 1.0:
        return 0.0  # single-segment molecule has no chain contribution

    G = len(group_names)

    # ── Molecular-averaged σ̄³ and d̄³ (vdW one-fluid, Eqs. 40, 42) ──
    # Using segment fractions z_k directly (Eq. 41 simplification):
    # In the full theory z_k involves σ³_kk weighting, but for the proxy
    # we use z_k = x_{s,k} as a consistent first-order approximation.
    # This is exact for homonuclear chains.

    sig3_bar = 0.0
    d3_bar   = 0.0

    for i in range(G):
        if xs[i] == 0.0:
            continue
        gi = group_names[i]
        for j in range(G):
            if xs[j] == 0.0:
                continue
            gj = group_names[j]
            w = xs[i] * xs[j]
            p = param_table[(gi, gj)]
            sig3_bar += w * p["sigma"] ** 3
            d3_bar   += w * p["d_kl"] ** 3

    if sig3_bar <= 0.0 or d3_bar <= 0.0:
        return 0.0

    # x̄₀ = σ̄ / d̄
    sig_bar = sig3_bar ** (1.0 / 3.0)
    d_bar   = d3_bar ** (1.0 / 3.0)
    x0_bar  = sig_bar / d_bar  if d_bar > 0.0 else 1.0

    # Clamp x0 to valid range [1, √2]
    if x0_bar < 1.0:
        x0_bar = 1.0
    if x0_bar > 1.4142:
        x0_bar = 1.4142

    # g^HS at σ̄ (Boublík, Eq. 48)
    g_hs = _g_HS_boublik(x0_bar, eta)

    if g_hs <= 0.0:
        return 0.0

    return -(m_i - 1.0) * math.log(g_hs)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3b — Association proxy  (Wertheim TPT1)
# ═══════════════════════════════════════════════════════════════════════════════

def _I_assoc(sig_kl: float, d_kl: float, eta: float) -> float:
    """
    Association kernel I_{kl}(T, ρ) in the SAFT-γ Mie framework.

    In the full theory (Ref [1], Eq. 67 and Appendix), the association
    integral involves g^{Mie} over a range of distances.  For the proxy
    we use the leading (zeroth-order) term:

        I_{kl} ≈ g^{HS}_d(σ_{kl}; η)

    evaluated with the **Boublík exponential form** (Eq. 48):

        g^{HS}_d(σ) = exp( k0 + k1·x0 + k2·x0² + k3·x0³ )

    where x0 = σ_{kl} / d_{kl}.  This is the same RDF used in the
    chain term, ensuring consistency across all Helmholtz contributions.

    The old Carnahan-Starling contact expression g^{CS} = (1-η/2)/(1-η)³
    is only valid at x0 = 1 (exact contact).  For association, the
    bonding sites are positioned at σ (not d), so x0 > 1 in general
    and the Boublík form is required.
    """
    if d_kl <= 0.0 or sig_kl <= 0.0:
        return 1.0  # fallback: ideal gas RDF

    x0 = sig_kl / d_kl
    # Clamp to valid Boublík range [1, √2]
    if x0 < 1.0:
        x0 = 1.0
    if x0 > 1.4142:
        x0 = 1.4142

    return _g_HS_boublik(x0, eta)


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
                    sig_kl: float, d_kl: float,
                    T: float, eta: float) -> float:
    """
    Association strength for a single site-site interaction:

        Δ_{kl,ab} = F_{kl,ab} · K_{kl,ab} · g^{HS}_d(σ_{kl}; η)

    where
        F = exp(ε^{assoc}/T) − 1          (Mayer-f function)
        K = bondingVolume                   (bonding volume, m³)
        g^{HS}_d = Boublík RDF at σ_{kl}  (Eq. 48, with x0 = σ/d)

    Ref [1] Eqs. (36)–(37), (48)–(52).
    """
    if bond_vol == 0.0 or eps_assoc == 0.0:
        return 0.0
    F = mayer_f(eps_assoc, T)
    K = bond_vol
    I = _I_assoc(sig_kl, d_kl, eta)
    return F * K * I


def delta_pair(k: str, l: str, groups: dict, cross: dict,
               sig_kl: float, d_kl: float,
               T: float = T_REF, eta: float = ETA_REF) -> float:
    """
    Total association strength between groups k and l, summed over
    all site-site interactions weighted by site multiplicities:

        Δ_{kl}(T, ρ) = Σ_{a ∈ k} Σ_{b ∈ l}  m_{k,a} · m_{l,b} · Δ_{kl,ab}

    Ref [1] Eq. (38) / standard Wertheim summation.

    Parameters
    ----------
    k, l      : group names
    sig_kl    : cross σ_{kl} [m]
    d_kl      : effective HS diameter d_{kl} [m]
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
        ci = cross.get((k, l)) or cross.get((l, k))
        if ci is not None:
            assoc_list = ci["association"]
        else:
            assoc_list = []

    # ── Association combining-rule fallback ────────────────────────────
    if not assoc_list and k != l:
        assoc_list = _cr1_association_fallback(k, l, groups)

    total = 0.0
    for inter in assoc_list:
        s1 = inter["site1"]    # site on group k
        s2 = inter["site2"]    # site on group l
        ea = inter["epsilonAssoc"]
        bv = inter["bondingVolume"]

        # Cross interactions require complementary (different) site types
        if k != l and not _is_valid_cross_site_pair(s1, s2):
            continue

        m1 = sites_k.get(s1, 0.0)
        m2 = sites_l.get(s2, 0.0)

        total += m1 * m2 * delta_site_pair(ea, bv, sig_kl, d_kl, T, eta)

    return total


def _cr1_association_fallback(k: str, l: str,
                              groups: dict) -> list[dict]:
    """
    Build a synthetic association interaction list for the cross pair
    (k, l) using the SAFT-γ Mie combining rules applied to compatible
    self-association site pairs.

    For every self-interaction  (a, b) on group k  and  (a', b') on
    group l, if the canonical types of (a, b) match (a', b') — in
    either order — a combined interaction is generated:

        ε^{assoc}_{kl} = √( ε^{assoc}_{kk} · ε^{assoc}_{ll} )   (geometric mean)
        K_{kl}         = [ (∛K_{kk} + ∛K_{ll}) / 2 ]³            (cubic mean of cube roots)

    The returned interaction uses the **actual site names** of group k
    (site1) and group l (site2) so that the multiplicity lookup in
    ``delta_pair`` works correctly.

    Returns an empty list if no compatible site pairs are found.
    """
    self_k = groups[k]["self_assoc"]
    self_l = groups[l]["self_assoc"]
    if not self_k or not self_l:
        return []

    # Index self-interactions by canonical site-type pair
    def _canon_key(s1: str, s2: str) -> tuple[str, str]:
        return (_canonical_site(s1), _canonical_site(s2))

    idx_k: dict[tuple[str, str], dict] = {}
    for inter in self_k:
        key = _canon_key(inter["site1"], inter["site2"])
        idx_k[key] = inter
        idx_k[(key[1], key[0])] = inter   # both orderings

    result: list[dict] = []
    used: set[tuple[str, str]] = set()   # avoid duplicate site combos
    for il in self_l:
        key_l = _canon_key(il["site1"], il["site2"])
        for try_key in [key_l, (key_l[1], key_l[0])]:
            if try_key in idx_k:
                ik = idx_k[try_key]
                # Map canonical back to actual site names on groups k and l
                actual_k_s1 = ik["site1"]   # site on k matching try_key[0]
                actual_l_s2 = il["site2"]   # site on l matching try_key[1]
                combo = (actual_k_s1, actual_l_s2)
                if combo in used:
                    continue
                # Cross interactions require different canonical site types
                if not _is_valid_cross_site_pair(actual_k_s1, actual_l_s2):
                    continue
                used.add(combo)

                e_k = ik["epsilonAssoc"]
                e_l = il["epsilonAssoc"]
                bv_k = ik["bondingVolume"]
                bv_l = il["bondingVolume"]
                eps_cross = math.sqrt(e_k * e_l) if e_k * e_l >= 0 else 0.0
                bv_cross = ((bv_k**(1.0/3.0) + bv_l**(1.0/3.0)) / 2.0)**3

                result.append({
                    "site1":         actual_k_s1,
                    "site2":         actual_l_s2,
                    "epsilonAssoc":  eps_cross,
                    "bondingVolume": bv_cross,
                })
                break

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3d — Association free energy  (Wertheim TPT1, Eq. 64–65)
#
# The association Helmholtz free energy per molecule is:
#
#   A^assoc / NkBT = Σ_k ν_{k,i} Σ_a n_{k,a} [ ln X_{k,a} − X_{k,a}/2 + 1/2 ]
#
# where X_{k,a} is the fraction of molecules *not* bonded at site a on
# group k, obtained from the mass-action equation:
#
#   X_{k,a} = 1 / (1 + ρ_s Σ_l Σ_b  n'_{l,b} X_{l,b} Δ_{kl,ab})
#
# Here n'_{l,b} = ν_{l,i} · n_{l,b} is the total count of site b on
# group l across the whole molecule, and ρ_s is the segment number
# density at the reference state.
#
# For a **pure component** the outer sum over components j collapses,
# and the iteration involves only site–site pairs within or between
# the groups of one molecule.
# ═══════════════════════════════════════════════════════════════════════════════

def _get_site_site_delta(k: str, site_a: str,
                         l: str, site_b: str,
                         groups: dict, cross: dict,
                         sig_kl: float, d_kl: float,
                         T: float, eta: float) -> float:
    """
    Return Δ_{kl,ab} for a specific (group_k, site_a)–(group_l, site_b) pair.

    Looks up the association interaction from self_assoc (if k==l) or
    cross-interaction data, falling back to combining rules.
    Uses the Boublík g^HS(x0; η) with x0 = σ_{kl}/d_{kl}.
    """
    # Get interaction list for this group pair
    if k == l:
        assoc_list = groups[k]["self_assoc"]
    else:
        ci = cross.get((k, l)) or cross.get((l, k))
        if ci is not None:
            assoc_list = ci["association"]
        else:
            assoc_list = _cr1_association_fallback(k, l, groups)

    # Cross interactions require complementary (different) site types
    if k != l and not _is_valid_cross_site_pair(site_a, site_b):
        return 0.0

    # Find the specific site-site interaction
    ca = _canonical_site(site_a)
    cb = _canonical_site(site_b)

    for inter in assoc_list:
        cs1 = _canonical_site(inter["site1"])
        cs2 = _canonical_site(inter["site2"])
        if (cs1 == ca and cs2 == cb) or (cs1 == cb and cs2 == ca):
            ea = inter["epsilonAssoc"]
            bv = inter["bondingVolume"]
            if ea == 0.0 or bv == 0.0:
                continue
            return delta_site_pair(ea, bv, sig_kl, d_kl, T, eta)

    return 0.0


def assoc_free_energy(vector, group_names: list[str], groups: dict,
                      cross: dict, param_table: dict,
                      T: float = T_REF, eta: float = ETA_REF,
                      max_iter: int = 200, tol: float = 1e-10) -> float:
    """
    Association contribution to A/NkBT for a pure molecule (Eq. 64–65).

    1. Enumerates all (group, site_type) pairs present in the molecule
       with their total site counts.
    2. Builds the Δ_{kl,ab} matrix for all active site pairs.
    3. Solves the mass-action equations iteratively for X_{k,a}.
    4. Evaluates A^assoc / NkBT.

    Parameters
    ----------
    vector      : group-count vector  [n_1, n_2, ..., n_G]
    group_names : ordered list of group names
    groups      : group database
    cross       : cross-interaction database
    param_table : pair parameters (for sigma_kl)
    T, eta      : reference state
    max_iter    : maximum iterations for mass-action solver
    tol         : convergence tolerance on X

    Returns
    -------
    float : A^assoc / NkBT  (dimensionless, ≤ 0 for associating molecules,
            = 0 for non-associating molecules)
    """
    n = np.asarray(vector, dtype=float)
    G = len(group_names)

    # ── Step 0: compute total chain length m_i for density conversion ──
    m_i = 0.0
    for i in range(G):
        if n[i] == 0:
            continue
        gd = groups[group_names[i]]
        m_i += n[i] * gd["nu"] * gd["shapeFactor"]

    # ── Step 1: enumerate all (group, site) with total counts ──
    # site_info: list of (group_name, site_name, total_count)
    # where total_count = n_k · n_{k,a}
    # (n_k = number of group in molecule, n_{k,a} = site multiplicity)
    site_info: list[tuple[str, str, float]] = []
    for i in range(G):
        if n[i] == 0:
            continue
        gname = group_names[i]
        gd = groups[gname]
        nk = n[i]          # number of this group in molecule
        for site_name, n_ka in gd["sites"].items():
            if n_ka > 0:
                # Total sites of type a from group k in the molecule:
                # ν_{k,i} · n_{k,a} in Papaioannou Eq. 65, where
                # ν_{k,i} = n_k is the group count in molecule i.
                total = nk * n_ka
                site_info.append((gname, site_name, total))

    if not site_info:
        return 0.0  # non-associating molecule

    NS = len(site_info)

    # ── Step 2: estimate molecular number density ρ from reference packing ──
    # Papaioannou Eq. 65 uses the molecular number density ρ (not ρ_s).
    # From ρ_s = 6η/(πd³) and ρ = ρ_s / m_i:
    d3_sum = 0.0
    d3_count = 0
    for i in range(G):
        if n[i] == 0:
            continue
        gi = group_names[i]
        p = param_table.get((gi, gi))
        if p is not None and p["d_kl"] > 0:
            d3_sum += p["d_kl"] ** 3
            d3_count += 1
    if d3_count > 0 and d3_sum > 0 and m_i > 0.0:
        d3_avg = d3_sum / d3_count
        rho_s = 6.0 * eta / (math.pi * d3_avg)
        rho_mol = rho_s / m_i   # molecular number density [1/m³]
    else:
        return 0.0

    # ── Step 3: build Δ matrix for all site pairs ──
    # delta_mat[i][j] = Δ_{kl,ab} for site_info[i] and site_info[j]
    delta_mat = np.zeros((NS, NS))
    for i in range(NS):
        gi, si, _ = site_info[i]
        for j in range(NS):
            gj, sj, _ = site_info[j]
            sig_kl = param_table.get((gi, gj), {}).get("sigma", 0.0)
            d_kl   = param_table.get((gi, gj), {}).get("d_kl", sig_kl)
            delta_mat[i][j] = _get_site_site_delta(
                gi, si, gj, sj, groups, cross, sig_kl, d_kl, T, eta)

    # Check if any association exists
    if np.all(delta_mat == 0.0):
        return 0.0

    # ── Step 4: solve mass-action for X  (Papaioannou Eq. 65) ──
    # X_{k,a} = 1 / (1 + ρ Σ_{l,b} ν_{l,i} n_{l,b} X_{l,b} Δ_{kl,ab})
    # where ρ is the molecular number density (not the segment density ρ_s).
    X = np.ones(NS)  # initial guess: fully unbonded
    n_counts = np.array([si[2] for si in site_info])  # total site counts

    for iteration in range(max_iter):
        X_new = np.zeros(NS)
        for i in range(NS):
            denom = 1.0
            for j in range(NS):
                denom += rho_mol * n_counts[j] * X[j] * delta_mat[i][j]
            X_new[i] = 1.0 / denom

        # Check convergence
        if np.max(np.abs(X_new - X)) < tol:
            X = X_new
            break
        X = X_new

    # ── Step 5: evaluate A^assoc / NkBT  (Eq. 64) ──
    # A^assoc/NkBT = Σ_k ν_{k,i} Σ_a n_{k,a} [ln X_{k,a} − X_{k,a}/2 + 1/2]
    # In our site_info, each entry already carries the total count
    # (n_k · n_{k,a}), and since ν_k is already absorbed into the group
    # parameterisation (via shape factor), we use the total count directly.
    F_assoc = 0.0
    for i in range(NS):
        _, _, count = site_info[i]
        Xi = max(X[i], 1e-300)  # guard against log(0)
        F_assoc += count * (math.log(Xi) - Xi / 2.0 + 0.5)

    return F_assoc


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — Pre-compute pair tables for GROUPS_OF_INTEREST
# ═══════════════════════════════════════════════════════════════════════════════

def build_pair_tables(group_names: list[str], groups: dict, cross: dict,
                      T: float = T_REF, eta: float = ETA_REF):
    """
    Compute pair-level monomer and association proxies for every
    unordered pair (k, l) among *group_names*.

    Returns
    -------
    a1_table    : dict[(str,str), float]   a_{1,kl}  [K·m³]
    delta_table : dict[(str,str), float]   Δ_{kl}(T, η)  [m³]
    param_table : dict[(str,str), dict]    {eps, sig, d_kl, lr, la} resolved
    """
    a1_table:    dict[tuple[str, str], float] = {}
    delta_table: dict[tuple[str, str], float] = {}
    param_table: dict[tuple[str, str], dict]  = {}

    # Pre-compute self d_{kk} for all groups (needed for cross d_{kl})
    d_self: dict[str, float] = {}
    for g in group_names:
        gd = groups[g]
        d_self[g] = _effective_hs_diameter(
            gd["sigma"], gd["epsilon"],
            gd["lambdaRepulsive"], gd["lambdaAttractive"], T)

    for k, l in combinations_with_replacement(group_names, 2):
        eps, sig, lr, la = get_pair_params(k, l, groups, cross)

        # Effective HS diameter: self → Gauss–Legendre, cross → arithmetic
        if k == l:
            d_kl = d_self[k]
        else:
            d_kl = (d_self[k] + d_self[l]) / 2.0

        # Segment number density from reference packing fraction
        if d_kl > 0.0:
            rho_s = 6.0 * eta / (math.pi * d_kl**3)
        else:
            rho_s = 0.0
        zeta_x = eta  # one-fluid approximation

        a1_val = _a1_pair(eps, sig, d_kl, lr, la, rho_s, zeta_x)
        a_val  = delta_pair(k, l, groups, cross, sig, d_kl, T, eta)

        # Store in both orderings for O(1) lookup
        for key in [(k, l), (l, k)]:
            a1_table[key]    = a1_val
            delta_table[key] = a_val
            param_table[key] = {"epsilon": eps, "sigma": sig,
                                "d_kl": d_kl,
                                "lambdaR": lr, "lambdaA": la}

    return a1_table, delta_table, param_table


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
              a1_table: dict, delta_table: dict,
              param_table: dict, cross: dict,
              T: float = T_REF, eta: float = ETA_REF):
    """
    Compute the SAFT-consistent thermodynamic signature of a molecule.

    Six components — three dimensionless free-energy terms + three
    auxiliary descriptors:

    1. **Monomer free energy** F^mono / (NkBT) — dimensionless:
           a1 = Σ_k Σ_l  x_{s,k} · x_{s,l} · a₁,kl       [K]
           F^mono = m_i · a^HS(η) + m_i · a1 / T            [—]
       Papaioannou Eqs. 9, 11, 17.

    2. **Chain free energy** F^chain / (NkBT) — dimensionless:
           F^chain = -(m_i - 1) · ln g^HS(σ̄; η)            [—]
       Papaioannou Eq. 46, with g^Mie ≈ g^HS (Boublík, Eq. 48).

    3. **Association free energy** F^assoc / (NkBT) — dimensionless:
           F^assoc = Σ_k ν_{k,i} Σ_a n_{k,a} [ ln X_{k,a} - X_{k,a}/2 + 1/2 ]
       Papaioannou Eq. 64, with X_{k,a} from iterative solution of
       the mass-action equations (Eq. 65).

    4. **Chain length**:
           m = Σ_k n_k · ν_k · S_k                          [—]

    5. **Packing proxy** (segment-averaged excluded volume):
           σ̄³ = Σ_k Σ_l  x_{s,k} · x_{s,l} · σ_{kl}³     [m³]

    6. **Shape average** (segment-fraction-weighted shape factor):
           S̄ = Σ_k  x_{s,k} · S_k                          [—]

    Ref [1] Eqs. (7)–(8), (9), (17)–(19), (46)–(52), (64)–(65).

    Returns
    -------
    dict  {"F_mono", "F_chain", "F_assoc", "m_total", "sigma3_avg", "shape_avg"}
    """
    xs, m_i = segment_fractions(vector, group_names, groups)

    if m_i == 0.0:
        return {"F_mono": 0.0, "F_chain": 0.0, "F_assoc": 0.0,
                "m_total": 0.0, "sigma3_avg": 0.0, "shape_avg": 0.0}

    G = len(group_names)

    # ── Double sums: a₁, σ³ ──
    a1_sum   = 0.0
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
            a1_sum   += w * a1_table[(gi, gj)]
            sig3_sum += w * param_table[(gi, gj)]["sigma"] ** 3

    # ── Hard-sphere free energy per segment ──
    a_hs = _a_hs_pure(eta)

    # ── Monomer free energy: A^mono / NkBT  (dimensionless) ──
    F_mono = m_i * a_hs + m_i * a1_sum / T

    # ── Chain free energy: A^chain / NkBT  (dimensionless) ──
    F_chain = chain_free_energy(m_i, xs, group_names, groups, param_table, eta)

    # ── Association free energy: A^assoc / NkBT  (dimensionless, ≤ 0) ──
    F_assoc = assoc_free_energy(vector, group_names, groups, cross,
                                param_table, T, eta)

    # Shape average: single sum  S̄ = Σ_k x_{s,k} · S_k
    shape_sum = 0.0
    for i in range(G):
        if xs[i] == 0.0:
            continue
        shape_sum += xs[i] * groups[group_names[i]]["shapeFactor"]

    return {"F_mono": F_mono, "F_chain": F_chain, "F_assoc": F_assoc,
            "m_total": m_i, "sigma3_avg": sig3_sum, "shape_avg": shape_sum}


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — Distance metric & ranking
# ═══════════════════════════════════════════════════════════════════════════════

def euclidean_distance_thermo(sig_cand: dict, sig_targ: dict,
                              weights: dict | None = None) -> float:
    """
    Thermodynamic distance in 3-D free-energy signature space.

    Measures how differently two molecules interact energetically,
    using the three SAFT Helmholtz contributions (A/NkBT).  Since all
    three components are dimensionless and on similar scales (~0–10),
    a plain weighted Euclidean distance is used:

        Δ_mono  = F^mono_c  − F^mono_t
        Δ_chain = F^chain_c − F^chain_t
        Δ_assoc = F^assoc_c − F^assoc_t

        D_thermo = √( w_mono·Δ_mono² + w_chain·Δ_chain² + w_assoc·Δ_assoc² )

    Parameters
    ----------
    weights : dict  {"w_MONO", "w_CHAIN", "w_ASSOC"}
        If None, defaults to module-level constants.
    """
    if weights is None:
        weights = {"w_MONO": W_MONO, "w_CHAIN": W_CHAIN, "w_ASSOC": W_ASSOC}

    dFm = sig_cand["F_mono"]  - sig_targ["F_mono"]
    dFc = sig_cand["F_chain"] - sig_targ["F_chain"]
    dFa = sig_cand["F_assoc"] - sig_targ["F_assoc"]

    return math.sqrt(weights["w_MONO"]  * dFm**2
                   + weights["w_CHAIN"] * dFc**2
                   + weights["w_ASSOC"] * dFa**2)


def distance_structure(sig_cand: dict, sig_targ: dict,
                       weights: dict | None = None) -> float:
    """
    Structural distance in 3-D molecular-architecture space.

    Measures how differently two molecules are built, independent of
    their energetics:

        d_m  = ln( m_c / m_t )        chain length  (backbone size)
        d_σ  = ln( σ̄³_c / σ̄³_t )     packing proxy (segment size)
        d_sh = ln( S̄_c / S̄_t )       shape average (fused topology)

        D_struct = √( w_m·d_m² + w_σ·d_σ² + w_sh·d_sh² )

    These three components capture complementary structural aspects:

    * **m** — total effective chain length (Σ n_k ν_k S_k); distinguishes
      short molecules (MEA, m ≈ 2) from long chains (DETA, m ≈ 5).
    * **σ̄³** — segment-averaged excluded volume; captures size
      differences between molecules whose segments have different
      diameters (e.g. cyclic vs linear groups).
    * **S̄** — segment-fraction-weighted shape factor; reflects chain
      topology — low S̄ means highly fused (compact) segments, high S̄
      means more linear connectivity.

    Parameters
    ----------
    weights : dict  {"w_M", "w_P", "w_SH"}
        If None, defaults to module-level constants (equal weights).
    """
    if weights is None:
        weights = {"w_M": W_M, "w_P": W_P, "w_SH": W_SH}

    m_c = max(sig_cand["m_total"], 1e-300)
    m_t = max(sig_targ["m_total"], 1e-300)
    dm  = math.log(m_c / m_t)

    s3_c = max(sig_cand["sigma3_avg"], 1e-300)
    s3_t = max(sig_targ["sigma3_avg"], 1e-300)
    ds3  = math.log(s3_c / s3_t)

    sh_c = max(sig_cand["shape_avg"], 1e-300)
    sh_t = max(sig_targ["shape_avg"], 1e-300)
    dsh  = math.log(sh_c / sh_t)

    return math.sqrt(weights["w_M"]  * dm**2
                   + weights["w_P"]  * ds3**2
                   + weights["w_SH"] * dsh**2)


def euclidean_distance_vector(vec_cand, vec_targ) -> float:
    """
    Simple Euclidean distance between two group-count vectors.

    This is a purely compositional metric — it measures how different
    the group multiplicities are, with no thermodynamic weighting:

        D_vec = sqrt( sum_k (n_{k,c} - n_{k,t})^2 )

    Parameters
    ----------
    vec_cand, vec_targ : list[int]
        Group-count vectors of equal length.

    Returns
    -------
    float  —  Euclidean distance in group-count space.
    """
    return math.sqrt(sum((c - t) ** 2 for c, t in zip(vec_cand, vec_targ)))


def cosine_distance_vector(vec_cand, vec_targ) -> float:
    """
    Cosine distance between two group-count vectors.

    Measures the angular difference between compositions, ignoring
    magnitude (molecule size).  Two molecules with the same group
    *ratios* but different sizes have cosine distance = 0.

        D_cos = 1 − (a · b) / (|a| |b|)

    Returns 0 when vectors are parallel, 1 when orthogonal.

    Parameters
    ----------
    vec_cand, vec_targ : list[int]
        Group-count vectors of equal length.

    Returns
    -------
    float  —  Cosine distance in [0, 1].
    """
    dot = sum(c * t for c, t in zip(vec_cand, vec_targ))
    norm_c = math.sqrt(sum(c * c for c in vec_cand))
    norm_t = math.sqrt(sum(t * t for t in vec_targ))
    if norm_c < 1e-300 or norm_t < 1e-300:
        return 1.0
    cos_sim = dot / (norm_c * norm_t)
    # Clamp for numerical safety
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return 1.0 - cos_sim


def _inverse_variance_weights(signatures: list[dict]) -> dict:
    """
    Compute inverse-variance weights for thermodynamic components.

    For each signature component, the variance across candidates is
    computed and the weight is set to 1/var.  This normalises the
    Euclidean distance so that each feature contributes equally on
    average, regardless of its natural scale.

    Falls back to 1.0 if variance is zero (all candidates identical
    in that component).
    """
    keys = ["F_mono", "F_chain", "F_assoc"]
    weight_names = ["w_MONO", "w_CHAIN", "w_ASSOC"]

    weights = {}
    for key, wname in zip(keys, weight_names):
        vals = [sig[key] for sig in signatures]
        if len(vals) < 2:
            weights[wname] = 1.0
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean)**2 for v in vals) / len(vals)
        weights[wname] = 1.0 / var if var > 1e-30 else 1.0

    return weights


def _inverse_variance_weights_structure(signatures: list[dict]) -> dict:
    """
    Compute inverse-variance weights for structural components.

    Same approach as ``_inverse_variance_weights`` but for the three
    structural descriptors (m, σ³, S̄).
    """
    keys = ["m_total", "sigma3_avg", "shape_avg"]
    weight_names = ["w_M", "w_P", "w_SH"]

    weights = {}
    for key, wname in zip(keys, weight_names):
        vals = []
        for sig in signatures:
            vals.append(math.log(max(sig[key], 1e-300)))
        if len(vals) < 2:
            weights[wname] = 1.0
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean)**2 for v in vals) / len(vals)
        weights[wname] = 1.0 / var if var > 1e-30 else 1.0

    return weights


def rank_candidates(target_vector, candidate_vectors,
                    group_names: list[str], groups: dict,
                    a1_table: dict, delta_table: dict,
                    param_table: dict, cross: dict,
                    T: float = T_REF, eta: float = ETA_REF,
                    auto_weights: bool = False):
    """
    Rank *candidate_vectors* by proximity to *target_vector*.

    Computes two independent distance metrics for each candidate:

    * **D_thermo** — thermodynamic distance (monomer + chain + association
      free-energy contributions).  This is the primary ranking criterion.
    * **D_struct** — structural distance (chain length + packing + shape).
      Reported alongside but does not influence the ranking order.

    Parameters
    ----------
    a1_table      : pair-level first-order perturbation a_{1,kl} [K]
    delta_table   : pair-level association strength Δ_{kl} [m³]
    param_table   : resolved pair parameters (for sigma3_avg, d_kl)
    cross         : cross-interaction dict (for association site lookups)
    auto_weights  : if True, use inverse-variance weights computed across
                    the candidate set; otherwise use module-level W_* constants.

    Returns
    -------
    ranking      : list[dict]  sorted by ascending D_thermo
    sig_targ     : dict        target signature
    weights_th   : dict        thermodynamic weights used
    weights_st   : dict        structural weights used
    """
    sig_targ = signature(target_vector, group_names, groups,
                         a1_table, delta_table, param_table, cross, T, eta)

    # Pre-compute all candidate signatures
    cand_sigs = []
    for cv in candidate_vectors:
        cand_sigs.append(signature(cv, group_names, groups,
                                   a1_table, delta_table, param_table, cross, T, eta))

    # Determine weights
    if auto_weights:
        weights_th = _inverse_variance_weights(cand_sigs)
        weights_st = _inverse_variance_weights_structure(cand_sigs)
    else:
        weights_th = {"w_MONO": W_MONO, "w_CHAIN": W_CHAIN, "w_ASSOC": W_ASSOC}
        weights_st = {"w_M": W_M, "w_P": W_P, "w_SH": W_SH}

    results = []
    for idx, sig_c in enumerate(cand_sigs):
        d_th = euclidean_distance_thermo(sig_c, sig_targ, weights_th)
        d_st = distance_structure(sig_c, sig_targ, weights_st)
        d_vec = euclidean_distance_vector(candidate_vectors[idx], target_vector)
        d_cos = cosine_distance_vector(candidate_vectors[idx], target_vector)
        results.append({
            "candidate_index":  idx,
            "candidate_vector": list(candidate_vectors[idx]),
            "signature":        sig_c,
            "distance":         d_th,
            "distance_struct":  d_st,
            "distance_vector":  d_vec,
            "distance_cosine":  d_cos,
        })

    results.sort(key=lambda r: r["distance"])
    return results, sig_targ, weights_th, weights_st


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — Utility: load compounds from CSV
# ═══════════════════════════════════════════════════════════════════════════════

def load_compounds(csv_path: str, group_names: list[str]) -> dict[str, list[int]]:
    """
    Read solvent_space.csv and return {smiles: group-count vector} aligned
    with *group_names*.
    """
    import csv
    import ast

    compounds: dict[str, list[int]] = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            smiles = row[0]
            vector_str = row[1]
            vec = [int(float(x)) for x in ast.literal_eval(vector_str)]
            compounds[smiles] = vec

    return compounds


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 8 — Groups of interest & main driver
# ═══════════════════════════════════════════════════════════════════════════════

GROUPS_OF_INTEREST = [
    "NH2_2nd", "NH_2nd", "N_2nd",
    "CH3", "CH2", "CH", "C",
    "CH2OH", "CH2OH_Short",
    "NH2", "NH", "N",
    "CHOH","OH_Short",
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

    xml_path = os.path.join(os.path.dirname(__file__), "..", "database", "CCS_Mie_Databank_221020.xml")
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
    a1_table, delta_table, param_table = build_pair_tables(
        group_names, groups, cross, T=T_REF, eta=ETA_REF)

    # ── Print resolved cross-parameter table (ε, σ, d, λr, λa) ──
    print("\n" + "═"*88)
    print("  Resolved pair parameters (ε_kl, σ_kl, d_kl, λr_kl, λa_kl)")
    print("═"*88)
    print(f"{'Pair':<30s} {'ε (K)':>12s} {'σ (m)':>14s} {'d (m)':>14s} {'λ_r':>10s} {'λ_a':>10s}")
    print("─"*88)
    for k, l in combinations_with_replacement(group_names, 2):
        p = param_table[(k, l)]
        print(f"{k+' | '+l:<30s} {p['epsilon']:12.4f} {p['sigma']:14.6e} "
              f"{p['d_kl']:14.6e} {p['lambdaR']:10.4f} {p['lambdaA']:10.4f}")

    # ── Print a1 & association pair tables ──
    _print_matrix(a1_table, group_names, "a_{1,kl}  (first-order perturbation per segment, K)")
    _print_matrix(delta_table, group_names, "Δ_{kl}  (association strength, m³)")

    # ── Load compounds ──
    csv_path = os.path.join(os.path.dirname(__file__), "..", "database/solvent_space.csv")
    compounds = load_compounds(csv_path, group_names)
    print(f"\nLoaded {len(compounds)} compounds from CSV.")
    if not compounds:
        print("No compounds found – exiting.")
        return

    # ── Target and ranking ──
    target_name = "NCCO"
    if target_name not in compounds:
        target_name = next(iter(compounds))
    target_vec = compounds[target_name]

    print(f"\n{'═'*60}")
    print(f"  TARGET: {target_name}")
    print(f"  Vector: {target_vec}")
    print(f"{'═'*60}")

    cand_names = [n for n in compounds if n != target_name]
    cand_vecs  = [compounds[n] for n in cand_names]

    ranking, sig_target, used_weights_th, used_weights_st = rank_candidates(
        target_vec, cand_vecs, group_names, groups,
        a1_table, delta_table, param_table, cross,
        T=T_REF, eta=ETA_REF, auto_weights=False)

    print(f"\nTarget signature:")
    print(f"  F_mono     = {sig_target['F_mono']:.6f}   (monomer A^mono/NkBT, dimensionless)")
    print(f"  F_chain    = {sig_target['F_chain']:.6f}   (chain   A^chain/NkBT, dimensionless)")
    print(f"  F_assoc    = {sig_target['F_assoc']:.6f}   (assoc   A^assoc/NkBT, dimensionless)")
    print(f"  m          = {sig_target['m_total']:.4f}           (chain length)")
    print(f"  sigma3_avg = {sig_target['sigma3_avg']:.6e}   (packing, m3)")
    print(f"  shape_avg  = {sig_target['shape_avg']:.6f}         (shape factor)")
    print(f"\nThermodynamic weights:")
    for wk, wv in used_weights_th.items():
        print(f"  {wk:>8s} = {wv:.6f}")
    print(f"Structural weights:")
    for wk, wv in used_weights_st.items():
        print(f"  {wk:>8s} = {wv:.6f}")

    print(f"\n{'Rank':>4s}  {'Compound':<35s}  {'F_mono':>10s}  {'F_chain':>10s}  {'F_assoc':>10s}  "
          f"{'m':>7s}  {'sig3_avg':>13s}  {'shape':>7s}  {'D_thermo':>10s}  {'D_struct':>10s}  {'D_vector':>10s}")
    print("\u2500" * 160)
    for rank, entry in enumerate(ranking, 1):
        cname = cand_names[entry["candidate_index"]]
        sig   = entry["signature"]
        d_th  = entry["distance"]
        d_st  = entry["distance_struct"]
        d_vec = entry["distance_vector"]
        print(f"{rank:4d}  {cname:<35s}  {sig['F_mono']:10.4f}  {sig['F_chain']:10.4f}  "
              f"{sig['F_assoc']:10.4f}  {sig['m_total']:7.4f}  "
              f"{sig['sigma3_avg']:13.6e}  {sig['shape_avg']:7.4f}  {d_th:10.6f}  {d_st:10.6f}  {d_vec:10.6f}")

    # ── Export pair tables ──
    a1_json = {f"{k[0]}|{k[1]}": v for k, v in a1_table.items()}
    A_json  = {f"{k[0]}|{k[1]}": v for k, v in delta_table.items()}

    # σ³_kl pair table
    sig3_json = {}
    for (k, l), pdict in param_table.items():
        sig3_json[f"{k}|{l}"] = pdict["sigma"] ** 3

    # Per-group metadata
    group_meta = {}
    for gname in group_names:
        gd = groups[gname]
        group_meta[gname] = {
            "nu":          gd["nu"],
            "shapeFactor": gd["shapeFactor"],
            "sigma":       gd["sigma"],
        }

    out_path = os.path.join(os.path.dirname(__file__), "..", "saft_pair_tables_florian.json")
    out_path = os.path.normpath(out_path)
    with open(out_path, "w") as f:
        json.dump({
            "a1_kl_perturbation": a1_json,
            "Delta_kl_association": A_json,
            "sigma3_kl": sig3_json,
            "groups": group_names,
            "group_metadata": group_meta,
            "T_ref_K": T_REF,
            "eta_ref": ETA_REF,
            "weights": {"w_MONO": W_MONO, "w_CHAIN": W_CHAIN, "w_ASSOC": W_ASSOC},
            "weights_structure": {"w_M": W_M, "w_P": W_P, "w_SH": W_SH},
        }, f, indent=2)
    print(f"\nPair tables saved to {out_path}")

    # ── Export ranking ──
    ranking_out = [{
        "compound": cand_names[e["candidate_index"]],
        "candidate_vector": e["candidate_vector"],
        "signature": e["signature"],
        "distance_thermo": e["distance"],
        "distance_struct": e["distance_struct"],
        "distance_vector": e["distance_vector"],
    } for e in ranking]

    rank_path = os.path.join(os.path.dirname(__file__), "..",
                             f"ranking_vs_{target_name}.json")
    rank_path = os.path.normpath(rank_path)
    with open(rank_path, "w") as f:
        json.dump({
            "target": target_name,
            "target_vector": target_vec,
            "target_signature": sig_target,
            "settings": {"T_ref_K": T_REF, "eta_ref": ETA_REF},
            "weights_thermo": used_weights_th,
            "weights_struct": used_weights_st,
            "ranking": ranking_out,
        }, f, indent=2)
    print(f"Ranking saved to {rank_path}")

    # ── Export CSV tables (appendix-ready) ──
    csv_dir = os.path.join(os.path.dirname(__file__), "..", "tables")
    csv_dir = os.path.normpath(csv_dir)
    os.makedirs(csv_dir, exist_ok=True)
    export_csv_tables(group_names, groups, cross, param_table,
                      a1_table, delta_table, csv_dir, T=T_REF, eta=ETA_REF)
    print(f"CSV tables saved to {csv_dir}/")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 9 — CSV export (appendix-ready parameter tables)
# ═══════════════════════════════════════════════════════════════════════════════

def export_csv_tables(group_names: list[str], groups: dict, cross: dict,
                      param_table: dict, a1_table: dict, delta_table: dict,
                      out_dir: str, T: float = T_REF, eta: float = ETA_REF):
    """
    Export all extracted / computed parameters as CSV files suitable for
    inclusion in a paper appendix.

    Files produced
    --------------
    table_self_parameters.csv
        Self-interaction (like-pair) group parameters: ν_k, S_k,
        ε_{kk}/k_B, σ_{kk}, λ^r_{kk}, λ^a_{kk}, plus association
        site information.

    table_cross_dispersion.csv
        Unlike-pair dispersion parameters for every (k,l) pair:
        ε_{kl}/k_B, σ_{kl}, d_{kl}, λ^r_{kl}, λ^a_{kl}, and the
        source (database / combining rule) for each parameter.

    table_association.csv
        Association parameters for every site-site interaction that
        has nonzero strength: groups, site labels, ε^HB_{kl,ab}/k_B,
        K_{kl,ab}, and source (database / CR-1 fallback).

    table_computed_a1_delta.csv
        Computed pair-level quantities: a_{1,kl}, Δ_{kl}, with the
        Mie prefactor C_{kl} and effective HS diameter d_{kl}.
    """
    import os

    M_TO_ANGSTROM = 1e10     # m → Å
    M3_TO_A3      = 1e30     # m³ → ų

    # ──────────────────────────────────────────────────────────────────────
    # Table 1: Self-interaction parameters
    # ──────────────────────────────────────────────────────────────────────
    path = os.path.join(out_dir, "table_self_parameters_florian.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Group", "nu_k", "S_k",
                     "eps_kk/kB [K]", "sigma_kk [A]",
                     "lambda_r_kk", "lambda_a_kk",
                     "assoc_sites"])
        for gname in group_names:
            gd = groups[gname]
            # Format association site summary
            site_strs = []
            for sname, mult in gd["sites"].items():
                if mult > 0:
                    site_strs.append(f"{sname}x{int(mult)}")
            sites_summary = "; ".join(site_strs) if site_strs else "—"

            w.writerow([
                gname,
                f"{gd['nu']:.4f}",
                f"{gd['shapeFactor']:.4f}",
                f"{gd['epsilon']:.4f}",
                f"{gd['sigma'] * M_TO_ANGSTROM:.4f}",
                f"{gd['lambdaRepulsive']:.4f}",
                f"{gd['lambdaAttractive']:.4f}",
                sites_summary,
            ])
    print(f"  → {path}")

    # ──────────────────────────────────────────────────────────────────────
    # Table 2: Cross-interaction dispersion parameters
    # ──────────────────────────────────────────────────────────────────────
    path = os.path.join(out_dir, "table_cross_dispersion_florian.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Group_k", "Group_l",
                     "eps_kl/kB [K]", "sigma_kl [A]", "d_kl [A]",
                     "lambda_r_kl", "lambda_a_kl",
                     "C_kl (Mie prefactor)",
                     "source_eps", "source_sigma", "source_lambdaR", "source_lambdaA"])

        for k, l in combinations_with_replacement(group_names, 2):
            p = param_table[(k, l)]
            eps_kl = p["epsilon"]
            sig_kl = p["sigma"]
            d_kl   = p["d_kl"]
            lr     = p["lambdaR"]
            la     = p["lambdaA"]
            C_kl   = mie_prefactor(lr, la) if lr > la else 0.0

            # Determine source for each parameter
            ci = cross.get((k, l))
            if k == l:
                src_e = src_s = src_lr = src_la = "self"
            else:
                src_e  = "database" if (ci and ci["epsilon"] != 0.0) else "CR"
                src_s  = "database" if (ci and ci["sigma"] != 0.0) else "CR"
                src_lr = "database" if (ci and ci["lambdaRepulsive"] != 0.0) else "CR"
                src_la = "database" if (ci and ci["lambdaAttractive"] != 0.0) else "CR"

            w.writerow([
                k, l,
                f"{eps_kl:.4f}",
                f"{sig_kl * M_TO_ANGSTROM:.4f}",
                f"{d_kl * M_TO_ANGSTROM:.4f}",
                f"{lr:.4f}",
                f"{la:.4f}",
                f"{C_kl:.6f}",
                src_e, src_s, src_lr, src_la,
            ])
    print(f"  → {path}")

    # ──────────────────────────────────────────────────────────────────────
    # Table 3: Association parameters (site-site interactions)
    # ──────────────────────────────────────────────────────────────────────
    path = os.path.join(out_dir, "table_association_florian.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Group_k", "Site_a", "Group_l", "Site_b",
                     "eps_HB_kl_ab/kB [K]", "K_kl_ab [A^3]",
                     "source"])

        seen: set[tuple] = set()
        for k, l in combinations_with_replacement(group_names, 2):
            sites_k = groups[k]["sites"]
            sites_l = groups[l]["sites"]
            if not sites_k or not sites_l:
                continue

            # Get association interaction list for this pair
            if k == l:
                assoc_list = groups[k]["self_assoc"]
                source = "self"
            else:
                ci = cross.get((k, l)) or cross.get((l, k))
                if ci is not None and ci["association"]:
                    assoc_list = ci["association"]
                    source = "database"
                else:
                    assoc_list = _cr1_association_fallback(k, l, groups)
                    source = "CR-1" if assoc_list else ""

            for inter in assoc_list:
                s1 = inter["site1"]
                s2 = inter["site2"]
                ea = inter["epsilonAssoc"]
                bv = inter["bondingVolume"]
                if ea == 0.0 and bv == 0.0:
                    continue

                # Avoid duplicate rows
                row_key = (k, s1, l, s2)
                if row_key in seen:
                    continue
                seen.add(row_key)

                w.writerow([
                    k, s1, l, s2,
                    f"{ea:.4f}",
                    f"{bv * M3_TO_A3:.6f}",
                    source,
                ])
    print(f"  → {path}")

    # ──────────────────────────────────────────────────────────────────────
    # Table 4: Computed pair quantities (a1_kl, Delta_kl)
    # ──────────────────────────────────────────────────────────────────────
    path = os.path.join(out_dir, "table_computed_a1_delta_florian.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Group_k", "Group_l",
                     "a1_kl [K.m^3]", "Delta_kl [m^3]",
                     "eps_kl/kB [K]", "sigma_kl [A]", "d_kl [A]",
                     "lambda_r_kl", "lambda_a_kl"])

        for k, l in combinations_with_replacement(group_names, 2):
            a1  = a1_table[(k, l)]
            dlt = delta_table[(k, l)]
            p   = param_table[(k, l)]

            w.writerow([
                k, l,
                f"{a1:.6e}",
                f"{dlt:.6e}",
                f"{p['epsilon']:.4f}",
                f"{p['sigma'] * M_TO_ANGSTROM:.4f}",
                f"{p['d_kl'] * M_TO_ANGSTROM:.4f}",
                f"{p['lambdaR']:.4f}",
                f"{p['lambdaA']:.4f}",
            ])
    print(f"  → {path}")

    # ── Export parameter tables as CSV (appendix-ready) ──
    _export_parameter_csvs(group_names, groups, cross, param_table,
                           a1_table, delta_table, T_REF, ETA_REF,
                           os.path.dirname(__file__))


def _export_parameter_csvs(group_names: list[str], groups: dict,
                           cross: dict, param_table: dict,
                           a1_table: dict, delta_table: dict,
                           T: float, eta: float, base_dir: str):
    """
    Export all resolved SAFT-γ Mie parameters as publication-ready CSV files.

    Produces three files (mirroring Perdomo et al. 2023 Tables 4–5):

      1. ``group_self_parameters.csv``  — per-group self-interaction parameters
      2. ``dispersion_parameters.csv``  — unlike-group dispersion interactions
      3. ``association_parameters.csv`` — site-site association interactions
    """
    import os

    out_dir = os.path.normpath(os.path.join(base_dir, ".."))

    # ── CSV 1 — Group self-interaction parameters ──
    self_path = os.path.join(out_dir, "group_self_parameters_florian.csv")
    with open(self_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "group",
            "nu (segments)", "S_k (shape factor)",
            "epsilon_kk/kB [K]", "sigma_kk [Ang]", "sigma_kk [m]",
            "lambda_r", "lambda_a",
            "d_kk [Ang]", "d_kk [m]",
            "C_kk (Mie prefactor)",
            "association_sites",
        ])
        for g in group_names:
            gd = groups[g]
            sig_A = gd["sigma"] * 1e10
            p = param_table.get((g, g), {})
            d_m  = p.get("d_kl", gd["sigma"])
            d_A  = d_m * 1e10
            lr = gd["lambdaRepulsive"]
            la = gd["lambdaAttractive"]
            C = mie_prefactor(lr, la)
            site_str = "; ".join(
                f"{sn}={int(sv)}" for sn, sv in gd["sites"].items() if sv > 0
            )
            if not site_str:
                site_str = "none"
            w.writerow([
                g,
                f"{gd['nu']:.4f}", f"{gd['shapeFactor']:.6f}",
                f"{gd['epsilon']:.4f}", f"{sig_A:.6f}", f"{gd['sigma']:.6e}",
                f"{lr:.4f}", f"{la:.4f}",
                f"{d_A:.6f}", f"{d_m:.6e}",
                f"{C:.6f}",
                site_str,
            ])
    print(f"Self-interaction parameters  -> {self_path}")

    # ── CSV 2 — Unlike-group dispersion parameters ──
    disp_path = os.path.join(out_dir, "dispersion_parameters_florian.csv")
    with open(disp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "group_k", "group_l",
            "epsilon_kl/kB [K]", "sigma_kl [Ang]", "sigma_kl [m]",
            "lambda_r_kl", "lambda_a_kl",
            "d_kl [Ang]", "d_kl [m]",
            "C_kl (Mie prefactor)",
            "a1_kl [K.m3]",
            "source_eps", "source_sigma", "source_lambda_r", "source_lambda_a",
        ])

        for k, l in combinations_with_replacement(group_names, 2):
            p = param_table[(k, l)]
            eps = p["epsilon"]
            sig = p["sigma"]
            d   = p["d_kl"]
            lr  = p["lambdaR"]
            la  = p["lambdaA"]
            C   = mie_prefactor(lr, la)
            a1  = a1_table[(k, l)]

            if k == l:
                src_e = src_s = src_lr = src_la = "self"
            else:
                ci = cross.get((k, l)) or cross.get((l, k))
                src_e  = "database" if (ci and ci["epsilon"] != 0.0) else "CR"
                src_s  = "database" if (ci and ci["sigma"] != 0.0) else "CR"
                src_lr = "database" if (ci and ci["lambdaRepulsive"] != 0.0) else "CR"
                src_la = "database" if (ci and ci["lambdaAttractive"] != 0.0) else "CR"

            w.writerow([
                k, l,
                f"{eps:.4f}", f"{sig*1e10:.6f}", f"{sig:.6e}",
                f"{lr:.4f}", f"{la:.4f}",
                f"{d*1e10:.6f}", f"{d:.6e}",
                f"{C:.6f}",
                f"{a1:.6e}",
                src_e, src_s, src_lr, src_la,
            ])
    print(f"Dispersion parameters       -> {disp_path}")

    # ── CSV 3 — Association parameters (site-site) ──
    assoc_path = os.path.join(out_dir, "association_parameters_florian.csv")
    with open(assoc_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "group_k", "group_l",
            "site_a (on k)", "site_b (on l)",
            "epsilon_HB_kl_ab/kB [K]", "K_kl_ab [Ang3]", "K_kl_ab [m3]",
            "F_kl_ab (Mayer-f at T_ref)",
            "g_HS (Boublik at eta_ref)",
            "Delta_kl_ab [m3]",
            "Delta_kl_total [m3]",
            "source",
        ])

        for k, l in combinations_with_replacement(group_names, 2):
            sites_k = groups[k]["sites"]
            sites_l = groups[l]["sites"]
            if not sites_k or not sites_l:
                continue

            if k == l:
                assoc_list = groups[k]["self_assoc"]
                source = "self"
            else:
                ci = cross.get((k, l)) or cross.get((l, k))
                if ci is not None and ci["association"]:
                    assoc_list = ci["association"]
                    source = "database"
                else:
                    assoc_list = _cr1_association_fallback(k, l, groups)
                    source = "CR-1" if assoc_list else "none"

            if not assoc_list:
                continue

            sig_kl = param_table[(k, l)]["sigma"]
            d_kl   = param_table[(k, l)]["d_kl"]
            delta_total = delta_table.get((k, l), 0.0)

            for inter in assoc_list:
                s1 = inter["site1"]
                s2 = inter["site2"]
                ea = inter["epsilonAssoc"]
                bv = inter["bondingVolume"]

                if ea == 0.0 or bv == 0.0:
                    continue

                # Cross interactions require different canonical site types
                if k != l and not _is_valid_cross_site_pair(s1, s2):
                    continue

                F = mayer_f(ea, T)
                I = _I_assoc(sig_kl, d_kl, eta)
                delta_ab = delta_site_pair(ea, bv, sig_kl, d_kl, T, eta)

                w.writerow([
                    k, l,
                    s1, s2,
                    f"{ea:.4f}", f"{bv*1e30:.6f}", f"{bv:.6e}",
                    f"{F:.6f}",
                    f"{I:.6f}",
                    f"{delta_ab:.6e}",
                    f"{delta_total:.6e}",
                    source,
                ])
    print(f"Association parameters       -> {assoc_path}")


if __name__ == "__main__":
    main()