# SAFT-γ Mie Group-Contribution Similarity — Complete Technical Reference

> Composition-agnostic similarity metrics for molecules represented as SAFT-γ Mie group-contribution vectors.
> Three complementary scripts implement (1) physics-based molecule ranking, (2) comparison against naive structural metrics, and (3) validation of combining rules against explicit database values.

---

## Table of Contents

1. [Repository layout](#1-repository-layout)
2. [Groups of interest](#2-groups-of-interest)
3. [Method — `saft_similarity_florian.py`](#3-method--saft_similarity_florianpy)
   - [3.1 Overview](#31-overview)
   - [3.2 Module 1 — XML database parser](#32-module-1--xml-database-parser)
   - [3.3 Module 2 — Combining rules](#33-module-2--combining-rules)
   - [3.4 Module 3a — Mie potential & Barker–Henderson diameter](#34-module-3a--mie-potential--barkerhenderson-diameter)
   - [3.5 Module 3b — First-order monomer perturbation $a_1$](#35-module-3b--first-order-monomer-perturbation-a_1)
   - [3.6 Module 3c — Chain free energy (Wertheim TPT1)](#36-module-3c--chain-free-energy-wertheim-tpt1)
   - [3.7 Module 3d — Association free energy (Wertheim TPT1)](#37-module-3d--association-free-energy-wertheim-tpt1)
   - [3.8 Module 4 — Pre-computed pair tables](#38-module-4--pre-computed-pair-tables)
   - [3.9 Module 5 — Molecular signature](#39-module-5--molecular-signature)
   - [3.10 Module 6 — Distance metrics and ranking](#310-module-6--distance-metrics-and-ranking)
   - [3.11 Global approximations and assumptions](#311-global-approximations-and-assumptions)
4. [Distance comparison study — `plot_distance_comparaison.py`](#4-distance-comparison-study--plot_distance_comparaisonpy)
   - [4.1 Metrics defined](#41-metrics-defined)
   - [4.2 Figures and results](#42-figures-and-results)
5. [Combining rules validation — `compare_combining_rules.py`](#5-combining-rules-validation--compare_combining_rulespy)
   - [5.1 Dispersive parameters](#51-dispersive-parameters)
   - [5.2 Association parameters (CR-1)](#52-association-parameters-cr-1)
   - [5.3 Figures and results](#53-figures-and-results)
6. [Parameter tables (CSV outputs)](#6-parameter-tables-csv-outputs)
7. [Running the scripts](#7-running-the-scripts)
8. [Dependencies](#8-dependencies)
9. [References](#9-references)

---

## 1. Repository layout

```
├── CCS_Mie_Databank_221020.xml          # SAFT-γ Mie group-contribution XML database
│
├── saft_similarity_florian.py           # Main physics-based similarity engine
├── plot_distance_comparaison.py         # Compare SAFT distances vs naive group-count distances
├── compare_combining_rules.py           # Validate combining rules against explicit DB values
│
├── solvent_space.csv                    # Candidate solvent molecules (SMILES + group vectors)
│
├── table_self_parameters_florian.csv    # Per-group self-interaction parameters (output)
├── table_cross_dispersion_florian.csv   # Cross-pair dispersion parameters (output)
├── table_association_florian.csv        # Site-site association parameters (output)
├── table_computed_a1_delta_florian.csv  # Computed a₁ and Δ values for all pairs (output)
│
├── fig_euclidean_scatter.png            # Scatter: D_struct / D_SAFT vs D_vec
├── fig_euclidean_residuals.png          # Residual histograms vs D_vec
├── fig_euclidean_rank_rank.png          # Rank-rank correlation vs D_vec
├── fig_cosine_scatter.png               # Scatter: D_struct / D_SAFT vs D_cos
├── fig_cosine_residuals.png             # Residual histograms vs D_cos
├── fig_cosine_rank_rank.png             # Rank-rank correlation vs D_cos
├── fig_euclidean_vs_cosine.png          # 4-panel Euclidean vs Cosine comparison
│
├── combining_rules_dispersive_parity.png   # Parity plots: CR vs DB for ε, σ, λ_r, λ_a
├── combining_rules_dispersive_errors.png   # Error histograms for dispersive CR
├── combining_rules_association_parity.png  # Parity plots: CR-1 vs DB for ε^assoc, K
└── combining_rules_association_errors.png  # Error histograms for association CR-1
```

---

## 2. Groups of interest

All scripts operate on the same 14 functional groups drawn from `CCS_Mie_Databank_221020.xml`:

| # | Group | Family | Notes |
|---|-------|--------|-------|
| 1 | `NH2` | Primary amine | Standard parameterisation |
| 2 | `NH2_2nd` | Primary amine | Re-parameterised for secondary-amine environment |
| 3 | `NH` | Secondary amine | |
| 4 | `NH_2nd` | Secondary amine | Re-parameterised variant |
| 5 | `N` | Tertiary amine | |
| 6 | `N_2nd` | Tertiary amine | Re-parameterised variant |
| 7 | `OH` | Hydroxyl | Standard; longer-chain context |
| 8 | `OH_Short` | Hydroxyl | Re-parameterised for short-chain context |
| 9 | `CH3` | Alkyl | Terminal methyl |
| 10 | `CH2` | Alkyl | Methylene |
| 11 | `CH2OH` | Hydroxymethyl | Combined CH₂+OH group |
| 12 | `CH2OH_Short` | Hydroxymethyl | Short-chain variant |
| 13 | `cNH` | Cyclic amine | N–H in a ring |
| 14 | `cCHNH` | Cyclic amine | CH–NH in a ring (e.g., piperazine) |

`_2nd` and `_Short` variants represent the same chemical moiety re-fitted for a different molecular context (chain length or substitution pattern) to improve transferability.

---

## 3. Method — `saft_similarity_florian.py`

### 3.1 Overview

The script computes, for each molecule in a candidate library, a **6-component thermodynamic signature** derived from the SAFT-γ Mie group-contribution equation of state. Molecules are then ranked against a target by Euclidean distance in this signature space, without solving the full equation of state. The reference state is fixed at T = 298.15 K and η = 0.40 (a liquid-like packing fraction).

The residual Helmholtz energy decomposes as (Papaioannou et al. [1], Eq. 1):

$$\frac{A^{\mathrm{res}}}{Nk_BT} = \frac{A^{\mathrm{mono}}}{Nk_BT} + \frac{A^{\mathrm{chain}}}{Nk_BT} + \frac{A^{\mathrm{assoc}}}{Nk_BT}$$

Each of these three contributions is evaluated using a reduced (proxy) formulation that is tractable without composition and density iteration:

| Contribution | Physical meaning | Proxy formula | Symbol |
|---|---|---|---|
| Monomer (HS + 1st-order perturbation) | Segment–segment repulsion and dispersion | $m_i\,a^{HS}(\eta) + m_i\,a_1/T$ | $F^{\mathrm{mono}}$ |
| Chain (Wertheim TPT1) | Connectivity of bonded segments | $-(m_i-1)\ln g^{HS}(\bar\sigma;\eta)$ | $F^{\mathrm{chain}}$ |
| Association (Wertheim TPT1) | Hydrogen bonding | $\sum_k\nu_k\sum_a n_{k,a}[\ln X_{k,a}-X_{k,a}/2+1/2]$ | $F^{\mathrm{assoc}}$ |

Three structural descriptors complement the free-energy terms:

| Descriptor | Physical meaning | Formula |
|---|---|---|
| $m$ | Effective chain length | $\sum_k n_k\nu_kS_k$ |
| $\bar\sigma^3$ | Segment-averaged excluded volume | $\sum_k\sum_l x_{s,k}x_{s,l}\sigma_{kl}^3$ |
| $\bar S$ | Shape-weighted topology | $\sum_k x_{s,k}S_k$ |

---

### 3.2 Module 1 — XML database parser

**Function:** `load_database(xml_path)`

Parses `CCS_Mie_Databank_221020.xml` into two Python dictionaries:

**`groups`** — per-group self-interaction data:
- $\nu_k$: number of identical Mie segments per group occurrence (`<numberOfSegments>`)
- $S_k$: shape factor (`<shapeFactor>`); accounts for fused or branched segment topology
- $\varepsilon_{kk}/k_B$ [K]: dispersion well depth
- $\sigma_{kk}$ [m]: segment diameter
- $\lambda^r_{kk}$, $\lambda^a_{kk}$: repulsive and attractive Mie exponents
- Self-association interactions: list of `{site1, site2, epsilonAssoc, bondingVolume}` entries
- Association site multiplicities: `{siteName: multiplicity}` — number of identical sites of each type per group

**`cross`** — explicit unlike-pair cross-interaction data, keyed by `(group1, group2)`:
- $\varepsilon_{kl}/k_B$ [K], $\sigma_{kl}$ [m], $\lambda^r_{kl}$, $\lambda^a_{kl}$ for dispersion
- Association: list of site-site interactions with $\varepsilon^{\mathrm{assoc}}_{kl,ab}/k_B$ [K] and $K_{kl,ab}$ [m³]

Each cross entry is also stored in the reverse direction `(group2, group1)` with site labels swapped, so that `site1` always refers to the first group in the key. This symmetry is exploited throughout `delta_pair` and `_get_site_site_delta`.

**Fallback:** When `lambdaAttractive` is absent from the XML, the default value $\lambda^a = 6.0$ (standard London attraction) is used (`LAMBDA_A_DEFAULT`).

---

### 3.3 Module 2 — Combining rules

**Source:** Papaioannou et al. [1], Eqs. 23–25; standard SAFT association CR-1.

When the database does not contain an explicit cross-interaction entry for a parameter, the following combining rules are applied hierarchically. For each scalar, if the database value is non-zero it takes precedence; otherwise the rule is applied.

#### Dispersion

**Segment diameter — arithmetic mean** ([1] Eq. 24):

$$\sigma_{kl} = \frac{\sigma_{kk} + \sigma_{ll}}{2}$$

Rationale: segment diameters are length-scale additive in the hard-sphere picture. The arithmetic mean is exact for two hard spheres of different radii.

**Mie exponents — nonlinear combining rule** ([1] Eq. 23):

$$\lambda_{kl} = 3 + \sqrt{(\lambda_{kk} - 3)(\lambda_{ll} - 3)}$$

Applied independently to $\lambda^r$ and $\lambda^a$. The shift by 3 reflects the fact that the Mie potential only diverges faster than $r^{-3}$ for $\lambda > 3$; the geometric mean acts on the "excess" above this lower bound. Returns 3.0 if the argument is negative (non-physical input safeguard).

**Dispersion well depth — modified Berthelot with volume correction** ([1] Eq. 25):

$$\varepsilon_{kl} = \sqrt{\varepsilon_{kk}\,\varepsilon_{ll}}\;\frac{\sqrt{\sigma_{kk}^3\,\sigma_{ll}^3}}{\sigma_{kl}^3}$$

The geometric mean (standard Berthelot) is corrected by the ratio of geometric-mean volume to cross-pair volume. This ensures that the well depth scales consistently with the strength of the effective dispersion interaction when segment sizes differ significantly.

#### Association — CR-1 fallback

Applied when no explicit cross-association entry exists in the database and both groups carry association sites.

**Association energy — geometric mean:**

$$\varepsilon^{\mathrm{assoc}}_{kl,ab} = \sqrt{\varepsilon^{\mathrm{assoc}}_{kk,aa}\;\varepsilon^{\mathrm{assoc}}_{ll,bb}}$$

**Bonding volume — cubic mean of cube roots:**

$$K_{kl,ab} = \left[\frac{K_{kk,aa}^{1/3} + K_{ll,bb}^{1/3}}{2}\right]^3$$

This form is used because $K$ represents a physical volume; the arithmetic mean of the linear extent (cube root) is more physically natural than a direct arithmetic mean of the volumes.

**Site-type compatibility:** Cross-association interactions are only formed between sites with **different** canonical types. The canonical types are `e` (electron donor, aggregating `e`, `e1`, `e2`) and `H` (hydrogen donor), and `a` (aggregating `a1`, `a2`). A site pair `(e, H)` is valid; `(e, e1)` or `(H, H)` is not, and yields $\Delta = 0$.

**Priority logic in `get_pair_params`:**
1. Self-pair ($k = k$): use self-interaction directly.
2. Explicit cross entry with non-zero value: use database value.
3. Explicit cross entry with zero value, or no cross entry: apply combining rule.

---

### 3.4 Module 3a — Mie potential & Barker–Henderson diameter

**Source:** [1] Eqs. 2, 10; [4] Sec. II.

#### Mie potential

The pair interaction between segments $k$ and $l$ is:

$$u^{\mathrm{Mie}}_{kl}(r) = C_{kl}\,\varepsilon_{kl}\left[\left(\frac{\sigma_{kl}}{r}\right)^{\lambda^r_{kl}} - \left(\frac{\sigma_{kl}}{r}\right)^{\lambda^a_{kl}}\right]$$

The prefactor $C_{kl}$ ensures the well depth equals $\varepsilon_{kl}$ at the minimum ([1] Eq. 2):

$$C_{kl} = \frac{\lambda^r_{kl}}{\lambda^r_{kl} - \lambda^a_{kl}}\left(\frac{\lambda^r_{kl}}{\lambda^a_{kl}}\right)^{\lambda^a_{kl}/(\lambda^r_{kl} - \lambda^a_{kl})}$$

Returns 0 if $\lambda^r \le \lambda^a$ (degenerate).

#### Barker–Henderson effective hard-sphere diameter

The Barker–Henderson (BH) perturbation theory replaces the real soft-core potential by an effective hard sphere whose diameter $d_{kk}$ equals the range over which the potential is repulsive ([1] Eq. 10):

$$d_{kk} = \int_0^{\sigma_{kk}}\left[1 - \exp\!\left(-\frac{u^{\mathrm{Mie}}_{kk}(r)}{k_BT}\right)\right]dr$$

This integral is evaluated by **5-point Gauss–Legendre quadrature** on $[0, \sigma]$ using nodes/weights from Paricaud (2006), recommended in the SAFT-VR Mie framework. Since $\varepsilon_{kk}$ is stored in Kelvin (= $\varepsilon_{kk}/k_B$), the integrand is:

$$\text{integrand}(r) = 1 - \exp\!\left(-C_{kk}\,\frac{\varepsilon_{kk}}{T}\left[\left(\frac{\sigma_{kk}}{r}\right)^{\lambda^r} - \left(\frac{\sigma_{kk}}{r}\right)^{\lambda^a}\right]\right)$$

For cross pairs, the BH diameter is computed as the arithmetic mean of self diameters:

$$d_{kl} = \frac{d_{kk} + d_{ll}}{2}$$

---

### 3.5 Module 3b — First-order monomer perturbation $a_1$

**Source:** [1] Eqs. 17–20, 25–27; [4] Sec. II.

The first-order mean-attractive perturbation per segment for molecule $i$ is:

$$a_1 = \sum_k\sum_l x_{s,k}\,x_{s,l}\,a_{1,kl}$$

where $x_{s,k}$ are segment fractions (Module 5) and $a_{1,kl}$ is the pair-level first-order term:

$$a_{1,kl} = C_{kl}\left[x_0^{\lambda^a}\!\left(a_1^S(\lambda^a) + B(\lambda^a)\right) - x_0^{\lambda^r}\!\left(a_1^S(\lambda^r) + B(\lambda^r)\right)\right]$$

where $x_0 = \sigma_{kl}/d_{kl}$ ([1] Eq. 19).

#### Sutherland integral $a_1^S$

The Sutherland first-order term is ([1] Eq. 25):

$$a_1^S(\rho_s;\lambda) = -2\pi\rho_s\,\frac{\varepsilon_{kl}\,d_{kl}^3}{\lambda - 3}\cdot\frac{1 - \zeta_{\mathrm{eff}}/2}{(1 - \zeta_{\mathrm{eff}})^3}$$

with the effective packing fraction $\zeta_{\mathrm{eff}}$ replacing the bare $\zeta_x$ to account for the finite range of the reference Sutherland potential. It is parameterised as ([1] Eq. 26–27):

$$\zeta_{\mathrm{eff}}(\zeta_x;\lambda) = c_1\zeta_x + c_2\zeta_x^2 + c_3\zeta_x^3 + c_4\zeta_x^4$$

$$\begin{pmatrix}c_1\\c_2\\c_3\\c_4\end{pmatrix} = \mathbf{M}\begin{pmatrix}1\\1/\lambda\\1/\lambda^2\\1/\lambda^3\end{pmatrix}$$

with the $4\times4$ matrix $\mathbf{M}$ from Lafitte et al. [4] (Table I), stored as `_ZETA_EFF_MATRIX` in the code.

#### Residual term $B$

The $B_{kl}$ term accounts for the correction beyond the Sutherland approximation ([1] Eq. 20). In the proxy this term is set to **zero** (`_B_kl` always returns 0.0). This is equivalent to the Sutherland-only or mean-field approximation, consistent with truncating the perturbation expansion at first order.

#### Hard-sphere reference $a^{HS}$

The one-component Carnahan–Starling (CS) hard-sphere free energy per segment ([1] Eq. 12, reduced):

$$a^{HS} = \frac{4\eta - 3\eta^2}{(1-\eta)^2}$$

In the full multi-component BMCSL theory this requires all moment densities $\zeta_m$. In the proxy, the one-fluid approximation sets $\eta = \zeta_x = 0.40$ (reference liquid packing fraction).

#### Monomer free energy

Combining the above at the molecule level ([1] Eq. 9, truncated at first order):

$$F^{\mathrm{mono}} = \frac{A^{\mathrm{mono}}}{Nk_BT} \approx m_i\,a^{HS}(\eta) + \frac{m_i}{T}\,a_1$$

The second-order term $a_2$ (fluctuation term, involving $K^{HS}$) and the third-order term $a_3$ (skewness correction) are omitted. They are corrections that reduce in relative importance as $T$ increases above the critical point.

---

### 3.6 Module 3c — Chain free energy (Wertheim TPT1)

**Source:** [1] Eqs. 46–52.

Wertheim's first-order thermodynamic perturbation theory for chain connectivity gives ([1] Eq. 46):

$$F^{\mathrm{chain}} = \frac{A^{\mathrm{chain}}}{Nk_BT} = -(m_i - 1)\ln g^{\mathrm{Mie}}(\bar\sigma;\zeta_x)$$

where $\bar\sigma$ is the molecular-averaged segment diameter and $g^{\mathrm{Mie}}$ is the radial distribution function of the reference Mie fluid evaluated at contact ($r = \bar\sigma$). In the proxy, $g^{\mathrm{Mie}}$ is approximated by its **zeroth-order (hard-sphere) term** using the Boublík exponential form ([1] Eq. 48):

$$g^{HS}_d(\bar\sigma;\zeta_x) = \exp\!\left(k_0 + k_1\bar{x}_0 + k_2\bar{x}_0^2 + k_3\bar{x}_0^3\right)$$

where $\bar{x}_0 = \bar\sigma/\bar d$ and the coefficients $k_0$–$k_3$ are functions of $\zeta_x$ ([1] Eqs. 49–52):

$$k_0 = -\ln(1-z) + \frac{42z - 39z^2 + 9z^3 - 2z^4}{6(1-z)^3}$$

$$k_1 = \frac{z^4 + 6z^2 - 12z}{2(1-z)^3}, \quad k_2 = \frac{-3z^2}{8(1-z)^2}, \quad k_3 = \frac{-z^4 + 3z^2 + 3z}{6(1-z)^3}$$

The molecular-averaged diameters use the vdW one-fluid mixing rules ([1] Eqs. 40, 42):

$$\bar\sigma^3 = \sum_k\sum_l x_{s,k}\,x_{s,l}\,\sigma_{kl}^3, \qquad \bar d^3 = \sum_k\sum_l x_{s,k}\,x_{s,l}\,d_{kl}^3$$

For a single-segment molecule ($m_i \le 1$) the chain contribution is identically zero.

Note: the Boublík form is clamped to $\bar x_0 \in [1,\sqrt{2}]$. The exact contact case $\bar x_0 = 1$ recovers the standard Carnahan–Starling contact value; the maximum $\sqrt{2}$ corresponds to the geometric packing limit.

---

### 3.7 Module 3d — Association free energy (Wertheim TPT1)

**Source:** [1] Eqs. 36–38, 64–65.

#### Association strength $\Delta_{kl,ab}$

For each site-site pair $(a \in k, b \in l)$ the association strength is ([1] Eqs. 36–37):

$$\Delta_{kl,ab} = F_{kl,ab}\,K_{kl,ab}\,g^{HS}_d(\sigma_{kl};\eta)$$

where:

**Mayer-f function** ([1] Eq. 36):
$$F_{kl,ab} = \exp\!\left(\frac{\varepsilon^{\mathrm{assoc}}_{kl,ab}}{T}\right) - 1$$

$K_{kl,ab}$ is the bonding volume [m³]. The RDF $g^{HS}_d$ is evaluated at $x_0 = \sigma_{kl}/d_{kl}$ using the same Boublík form as the chain term — this ensures thermodynamic consistency. The older Carnahan–Starling contact expression $(1-\eta/2)/(1-\eta)^3$ is only valid for $x_0 = 1$ (exact contact); since the association sites sit at $r = \sigma_{kl} > d_{kl}$, we have $x_0 > 1$ and the Boublík form is necessary.

The total pair association strength, summed over all site multiplicities, is ([1] Eq. 38):

$$\Delta_{kl} = \sum_{a \in k}\sum_{b \in l} m_{k,a}\,m_{l,b}\,\Delta_{kl,ab}$$

#### Association free energy

The Wertheim TPT1 association Helmholtz contribution per molecule is ([1] Eq. 64):

$$F^{\mathrm{assoc}} = \frac{A^{\mathrm{assoc}}}{Nk_BT} = \sum_k\nu_k\sum_a n_{k,a}\left[\ln X_{k,a} - \frac{X_{k,a}}{2} + \frac{1}{2}\right]$$

where $X_{k,a}$ is the fraction of molecules unbonded at site $a$ on group $k$, obtained by solving the mass-action equations iteratively ([1] Eq. 65):

$$X_{k,a} = \frac{1}{1 + \rho_{\mathrm{mol}}\sum_l\sum_b \nu_{l,i}\,n_{l,b}\,X_{l,b}\,\Delta_{kl,ab}}$$

Iteration starts from $X_{k,a} = 1$ (fully unbonded) and converges to within $10^{-10}$ (up to 200 iterations). The molecular number density is estimated from the reference packing fraction: $\rho_{\mathrm{mol}} = \rho_s / m_i$ with $\rho_s = 6\eta/(\pi\bar d^3)$.

For a molecule with no association sites, $F^{\mathrm{assoc}} = 0$.

---

### 3.8 Module 4 — Pre-computed pair tables

**Function:** `build_pair_tables(group_names, groups, cross, T, eta)`

Before signatures can be computed, all unique unordered pairs $(k,l)$ (with $k \le l$) among the groups of interest are pre-computed and stored:

- `a1_table[(k,l)]`: first-order perturbation $a_{1,kl}$ [K·m³]
- `delta_table[(k,l)]`: pair-level total association strength $\Delta_{kl}$ [m³]
- `param_table[(k,l)]`: resolved $\{\varepsilon_{kl}, \sigma_{kl}, d_{kl}, \lambda^r_{kl}, \lambda^a_{kl}\}$

Both orderings $(k,l)$ and $(l,k)$ are stored for O(1) lookup. Self-diameters $d_{kk}$ are computed first by Gauss–Legendre; cross diameters $d_{kl}$ use the arithmetic mean of self diameters.

These tables are exported as `table_computed_a1_delta_florian.csv` and `table_cross_dispersion_florian.csv`.

---

### 3.9 Module 5 — Molecular signature

**Function:** `signature(vector, group_names, groups, a1_table, delta_table, param_table, cross, T, eta)`

A molecule is represented by a group-count vector $\mathbf{n}$ with entries $n_k$ = number of occurrences of group $k$.

#### Segment fractions ([1] Eqs. 7–8)

$$m_i = \sum_k n_k\,\nu_k\,S_k, \qquad x_{s,k} = \frac{n_k\,\nu_k\,S_k}{m_i}$$

$\nu_k$ is the number of identical Mie segments within group $k$, $S_k$ is the shape factor. $x_{s,k}$ are the SAFT segment fractions (sum to 1) and replace intramolecular group-pair counting.

#### The 6-component signature

| Component | Symbol | Formula | Units |
|---|---|---|---|
| Monomer free energy | $F^{\mathrm{mono}}$ | $m_i\,a^{HS}(\eta) + m_i\,a_1/T$ | dimensionless ($A/Nk_BT$) |
| Chain free energy | $F^{\mathrm{chain}}$ | $-(m_i-1)\ln g^{HS}(\bar\sigma;\eta)$ | dimensionless |
| Association free energy | $F^{\mathrm{assoc}}$ | Wertheim TPT1, iterative | dimensionless |
| Chain length | $m$ | $\sum_k n_k\nu_kS_k$ | dimensionless |
| Packing proxy | $\bar\sigma^3$ | $\sum_k\sum_l x_{s,k}x_{s,l}\sigma_{kl}^3$ | m³ |
| Shape average | $\bar S$ | $\sum_k x_{s,k}S_k$ | dimensionless |

---

### 3.10 Module 6 — Distance metrics and ranking

**Functions:** `euclidean_distance_thermo`, `distance_structure`, `euclidean_distance_vector`, `cosine_distance_vector`, `rank_candidates`

#### Thermodynamic distance $D_{\mathrm{thermo}}$

Weighted Euclidean distance in 3D free-energy space:

$$D_{\mathrm{thermo}} = \sqrt{w_{\mathrm{mono}}\,\Delta F_{\mathrm{mono}}^2 + w_{\mathrm{chain}}\,\Delta F_{\mathrm{chain}}^2 + w_{\mathrm{assoc}}\,\Delta F_{\mathrm{assoc}}^2}$$

All three components are dimensionless ($A/Nk_BT$) and on comparable scales. Default weights $w = 1$ (equal). Optional `auto_weights=True` applies inverse-variance weights computed across the candidate set, normalising each dimension by its spread.

#### Structural distance $D_{\mathrm{struct}}$

Log-ratio Euclidean distance in 3D structural space:

$$D_{\mathrm{struct}} = \sqrt{w_m\left(\ln\frac{m_c}{m_t}\right)^2 + w_\sigma\left(\ln\frac{\bar\sigma^3_c}{\bar\sigma^3_t}\right)^2 + w_S\left(\ln\frac{\bar S_c}{\bar S_t}\right)^2}$$

Log ratios are used because these quantities span orders of magnitude; the log ratio equals zero for identical molecules and is symmetric under exchange. The three components capture complementary aspects: chain length (backbone size), excluded volume (segment size), and shape topology.

#### Naive group-count distances (for comparison only)

$$D_{\mathrm{vec}} = \sqrt{\sum_k(n_{k,c} - n_{k,t})^2}$$

$$D_{\cos} = 1 - \frac{\mathbf{n}_c \cdot \mathbf{n}_t}{|\mathbf{n}_c||\mathbf{n}_t|}$$

These carry no thermodynamic information and are used exclusively as baseline references in `plot_distance_comparaison.py`.

#### Ranking

`rank_candidates` returns candidates sorted by ascending $D_{\mathrm{thermo}}$. For each candidate the JSON output records both $D_{\mathrm{thermo}}$ and $D_{\mathrm{struct}}$ separately; the full signature and raw group vector are also stored.

---

### 3.11 Global approximations and assumptions

| Assumption | Where applied | Impact |
|---|---|---|
| $T = 298.15$ K, $\eta = 0.40$ fixed | All free-energy evaluations | Signatures are valid only near ambient conditions; no temperature sensitivity |
| Perturbation truncated at $a_1$ (first order) | Monomer term | Ignores $a_2$ (fluctuation) and $a_3$; acceptable for $T \gg T_c/3$ |
| $B_{kl} = 0$ (Sutherland-only $a_1$) | $a_{1,kl}$ computation | Slight underestimation of $a_1$ for steep potentials; correction is small |
| $g^{\mathrm{Mie}} \approx g^{HS}$ (zeroth-order) | Chain term, association $I$ kernel | First-order correction to $g$ neglected; consistent with $a_1$-level perturbation |
| One-fluid approximation $\eta = \zeta_x$ | $a^{HS}$, $a_1^S$ | Multi-component BMCSL expressions simplify; accurate for homogeneous mixtures |
| Cross $d_{kl}$ = arithmetic mean of self $d_{kk}$ | Pair table construction | Exact only for equal-size segments; small error for similar groups |
| Association fallback CR-1 | Cross pairs lacking DB entry | May differ from fitted value; validated by `compare_combining_rules.py` |
| $\nu_k$ and $S_k$ absorbed into segment fraction $x_{s,k}$ | Chain, association | Uses $x_{s,k} = n_k\nu_kS_k/m_i$ directly as the vdW one-fluid weighting; exact for homonuclear chains |

---

## 4. Distance comparison study — `plot_distance_comparaison.py`

### 4.1 Metrics defined

The script loads a ranking JSON output from `saft_similarity_florian.py` and compares four distance measures across all ranked candidates:

| Metric | Symbol | Description |
|---|---|---|
| SAFT thermodynamic | $D_{\mathrm{SAFT}}$ | Euclidean distance in $(F^{\mathrm{mono}}, F^{\mathrm{chain}}, F^{\mathrm{assoc}})$ space |
| Structural | $D_{\mathrm{struct}}$ | Log-Euclidean distance in $(m, \bar\sigma^3, \bar S)$ space |
| Euclidean (naive) | $D_{\mathrm{vec}}$ | $\ell_2$ distance between raw group-count vectors |
| Cosine (naive) | $D_{\cos}$ | $1 - \cos(\mathbf{n}_c, \mathbf{n}_t)$ |

All four are normalised to $[0, 1]$ by dividing by their maximum before comparison (so that relative ordering can be assessed directly). The key question is: **how much information does the naive structural count contain about the SAFT-physics-based distance?**

Spearman rank correlation $\rho$ is computed for every pair of metrics, quantifying monotonic agreement.

### 4.2 Figures and results

Seven figures are produced:

#### `fig_euclidean_scatter.png` and `fig_cosine_scatter.png`

Two-panel scatter plots comparing $D_{\mathrm{struct}}$ (left) and $D_{\mathrm{SAFT}}$ (right) against the naive distance, normalised to $[0,1]$. The diagonal line marks perfect agreement. Spearman $\rho$ is annotated. The 5 molecules closest to the diagonal in each panel are highlighted by name.

**What this reveals:** Points below the diagonal are molecules that appear more similar according to SAFT physics than the naive metric suggests; points above are penalised more by the naive metric. Large scatter indicates the naive metric misranks molecules relative to their thermodynamic similarity.

#### `fig_euclidean_residuals.png` and `fig_cosine_residuals.png`

Two-panel residual histograms of $D_{\mathrm{struct}}^{\mathrm{norm}} - D_{\mathrm{naive}}^{\mathrm{norm}}$ (left) and $D_{\mathrm{SAFT}}^{\mathrm{norm}} - D_{\mathrm{naive}}^{\mathrm{norm}}$ (right). Mean absolute error (MAE), standard deviation $\sigma$, and median are reported.

**What this reveals:** A histogram centred near zero with low MAE indicates the naive metric is a good proxy; skewed or broad histograms indicate systematic disagreement.

#### `fig_euclidean_rank_rank.png` and `fig_cosine_rank_rank.png`

Rank-rank scatter plots comparing the ordering by $D_{\mathrm{struct}}$ (left) and $D_{\mathrm{SAFT}}$ (right) to the ordering by the naive metric. Perfect agreement would align all points on the diagonal. Spearman $\rho$ quantifies rank concordance.

**What this reveals:** Even if the absolute distances differ, the ranking question ("which molecule is most similar to the target?") may still agree. Deviations from the diagonal at the top of the x-axis (i.e., small naive distance but large SAFT rank) identify molecules that appear structurally close but are thermodynamically far — the most dangerous false positives for a naive screening.

#### `fig_euclidean_vs_cosine.png`

Four-panel figure comparing the two naive metrics directly:

- **(a)** Scatter $D_{\mathrm{vec}}^{\mathrm{norm}}$ vs $D_{\cos}^{\mathrm{norm}}$ (with Spearman $\rho$ and 5 largest-divergence molecules highlighted)
- **(b)** Histogram of the difference $D_{\mathrm{vec}}^{\mathrm{norm}} - D_{\cos}^{\mathrm{norm}}$
- **(c)** $D_{\mathrm{SAFT}}$ vs. the difference: shows whether the thermodynamic distance correlates with the divergence between naive metrics
- **(d)** $D_{\mathrm{struct}}$ vs. the difference: same for the structural distance

**What this reveals:** Euclidean distance penalises molecules that differ in absolute group count (e.g. a long chain vs a short one), while cosine distance is insensitive to overall size. Panels (c)–(d) reveal whether the SAFT physics aligns with the size-sensitive (Euclidean) or size-agnostic (cosine) view of similarity.

---

## 5. Combining rules validation — `compare_combining_rules.py`

The script computes, for every cross pair $(k,l)$ that has an explicit entry in the XML database, both the **database value** and the value predicted by the **combining rule**. The relative error is:

$$\text{rel. error} = \frac{v^{\mathrm{CR}} - v^{\mathrm{DB}}}{|v^{\mathrm{DB}}|} \times 100\%$$

### 5.1 Dispersive parameters

For each cross pair with explicit database entries, the four combining rules are tested:

| Parameter | Combining rule | Expected quality |
|---|---|---|
| $\sigma_{kl}$ [Å] | Arithmetic mean | High — diameters are near-additive |
| $\lambda^r_{kl}$ | $3 + \sqrt{(\lambda^r_{kk}-3)(\lambda^r_{ll}-3)}$ | Good for similar groups; deviates for very different exponents |
| $\lambda^a_{kl}$ | $3 + \sqrt{(\lambda^a_{kk}-3)(\lambda^a_{ll}-3)}$ | Usually excellent — most groups share $\lambda^a = 6$ |
| $\varepsilon_{kl}/k_B$ [K] | Modified Berthelot | Moderate — sensitive to polar or associating pairs |

Pairs where one parameter is zero in the database are excluded from that parameter's comparison (zero may indicate the parameter was not fitted independently).

### 5.2 Association parameters (CR-1)

For each cross pair where the database contains explicit site-site association parameters, the CR-1 rule is evaluated by finding the matching self-association interaction of each group (by canonical site types) and applying:

$$\varepsilon^{\mathrm{assoc,CR}}_{kl} = \sqrt{\varepsilon^{\mathrm{assoc}}_{kk}\,\varepsilon^{\mathrm{assoc}}_{ll}}, \qquad K^{\mathrm{CR}}_{kl} = \left[\frac{K_{kk}^{1/3} + K_{ll}^{1/3}}{2}\right]^3$$

### 5.3 Figures and results

Four figures are produced:

#### `combining_rules_dispersive_parity.png`

A 2×2 grid of parity plots (database vs. combining rule) for $\varepsilon_{kl}/k_B$, $\sigma_{kl}$, $\lambda^r_{kl}$, $\lambda^a_{kl}$. Points on the diagonal indicate the combining rule reproduces the fitted database value exactly. Pearson $r$ and MAE are annotated. Pairs with relative error > 10% are labelled.

**Reading the figure:** The closer all points are to the diagonal, the less error the combining rule introduces. Systematic displacement (all points above or below) indicates bias; scatter indicates noise.

#### `combining_rules_dispersive_errors.png`

Relative-error histograms for each dispersive parameter. Bars show the distribution of (CR − DB)/|DB| × 100%. The median error and sample size $n$ are annotated; outliers beyond ±500% are excluded from the histogram (counted separately).

**Reading the figure:** A narrow histogram centred at 0% means the combining rule is a reliable predictor. A histogram shifted to positive/negative values indicates systematic over/under-prediction.

#### `combining_rules_association_parity.png`

Parity plots for $\varepsilon^{\mathrm{assoc}}_{kl}/k_B$ [K] and $K_{kl}$ [m³]. Same layout as the dispersive parity figure.

**Reading the figure:** Association parameters are typically harder to predict by combining rules because they are more sensitive to electronic structure. Wider scatter here than for dispersive parameters is expected.

#### `combining_rules_association_errors.png`

Relative-error histograms for $\varepsilon^{\mathrm{assoc}}_{kl}$ and $K_{kl}$.

**Reading the figure:** If the bonding volume histogram is broad (high MAE), this indicates that the geometric mean of cube roots may not capture the specific steric accessibility of cross association between different groups. The $\varepsilon^{\mathrm{assoc}}$ histogram reflects whether the geometric mean of association energies is a reliable estimate for unlike-group hydrogen bonding.

---

## 6. Parameter tables (CSV outputs)

`saft_similarity_florian.py` exports four publication-ready CSV files:

| File | Content | Key columns |
|---|---|---|
| `table_self_parameters_florian.csv` | Per-group self-interaction parameters | `nu`, `S_k`, `eps_kk/kB [K]`, `sigma_kk [A]`, `lambda_r_kk`, `lambda_a_kk`, `assoc_sites` |
| `table_cross_dispersion_florian.csv` | All cross-pair dispersion parameters (DB + CR) | `eps_kl/kB [K]`, `sigma_kl [A]`, `d_kl [A]`, `lambda_r_kl`, `lambda_a_kl`, `C_kl`, `source_eps/sigma/lambdaR/lambdaA` |
| `table_association_florian.csv` | Site-site association parameters with computed $F$ and $\Delta$ | `eps_HB_kl_ab/kB [K]`, `K_kl_ab [A^3]`, `F_kl_ab`, `g_HS`, `Delta_kl_ab [m^3]`, `Delta_kl_total`, `source` |
| `table_computed_a1_delta_florian.csv` | Pre-computed $a_{1,kl}$ and $\Delta_{kl}$ at reference state | `a1_kl [K.m^3]`, `Delta_kl [m^3]`, `eps_kl/kB [K]`, `sigma_kl [A]`, `d_kl [A]`, `lambda_r/a_kl` |

The `source` column in dispersion tables uses `"self"` for self-pairs, `"database"` for explicit cross entries, and `"CR"` for combining-rule-derived values.

---

## 7. Running the scripts

```bash
# 1. Compute similarity and rank candidates against a target
python saft_similarity_florian.py
#    → generates ranking JSON and four CSV parameter tables

# 2. Compare SAFT distances against naive group-count metrics
python plot_distance_comparaison.py ranking_vs_NCCO.json
#    → generates 7 figures in ./figures/

# 3. Validate combining rules against explicit database values
python compare_combining_rules.py --xml CCS_Mie_Databank_221020.xml
#    → generates 4 combining-rule figures in ./figures/
```

---

## 8. Dependencies

- Python ≥ 3.10
- `numpy`
- `scipy` (`spearmanr`, `pearsonr`)
- `matplotlib`

---

## 9. References

1. V. Papaioannou, T. Lafitte, C. Avendaño, C. S. Adjiman, G. Jackson, E. A. Müller, A. Galindo, *J. Chem. Phys.* **140**, 054107 (2014). — SAFT-γ Mie group-contribution EoS; primary source for all equations used in `saft_similarity_florian.py`.

2. S. Dufal, T. Lafitte, A. J. Haslam, A. Galindo, G. N. I. Clark, C. Vega, G. Jackson, *J. Chem. Eng. Data* **59**, 3272–3288 (2014). — Cross-interaction parameterisation and group database used in `CCS_Mie_Databank_221020.xml`.

3. A. J. Haslam, A. Galindo, G. Jackson — SAFT-γ Mie group-contribution framework review; group combining rules and association site topology.

4. T. Lafitte, A. Apostolakou, C. Avendaño, A. Galindo, C. S. Adjiman, E. A. Müller, G. Jackson, *J. Chem. Phys.* **139**, 154504 (2013). — SAFT-VR Mie EoS; source for the effective-packing-fraction matrix $\mathbf{M}$ (`_ZETA_EFF_MATRIX`) and Barker–Henderson quadrature.

5. T. Boublík, *J. Chem. Phys.* **53**, 471 (1970). — Hard-sphere RDF exponential form used in chain term and association kernel ($g^{HS}_d$, [1] Eqs. 48–52).

6. J. A. Barker, D. Henderson, *J. Chem. Phys.* **47**, 4714 (1967). — Perturbation theory and effective hard-sphere diameter definition.