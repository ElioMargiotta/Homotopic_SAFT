# SAFT-γ Mie Group Similarity & Molecule Ranking

Composition-agnostic similarity metrics for molecules represented by **SAFT-γ Mie group-contribution** vectors.  
Two complementary approaches are implemented, both reading from the same XML parameter database.

---

## Table of contents

1. [Repository layout](#repository-layout)
2. [Groups of interest](#groups-of-interest)
3. [Approach 1 – Feature-based cosine similarity](#approach-1--feature-based-soft-cosine-similarity-compute_similaritypy)
4. [Approach 2 – SAFT-γ Mie physics-based ranking](#approach-2--saft-γ-mie-physics-based-molecule-ranking-saft_similaritypy)
   - [Theoretical background](#theoretical-background)
   - [Step-by-step workflow](#step-by-step)
   - [Summary of assumptions](#summary-of-assumptions-and-approximations)
5. [Visualisation](#visualisation--group-level-similarity-plot_group_similaritypy)
   - [Group-group distance metric](#group-group-distance-metric)
   - [Figure 1 – Heatmap](#figure-1--distance-heatmap)
   - [Figure 2 – MDS embedding](#figure-2--2-d-mds-embedding)
   - [Figure 3 – Similarity network](#figure-3--similarity-network)
   - [Figure 4 – Amine & hydroxyl bars](#figure-4--amine--hydroxyl-distance-ranking)
   - [Figure 5 – Radar chart](#figure-5--radar-chart-of-group-descriptors)
6. [Running the scripts](#running-the-scripts)
7. [Database format](#database-format)
8. [References](#references)

---

## Repository layout

```
├── database/
│   └── database.xml               # SAFT-γ Mie group parameters (self + cross)
├── scripts/
│   ├── compute_similarity.py      # Approach 1 – feature-based cosine similarity matrix
│   ├── saft_similarity.py         # Approach 2 – SAFT-γ Mie physics-based ranking
│   └── plot_group_similarity.py   # Visualisation – group-level similarity figures
├── figures/
│   ├── group_distance_heatmap.png
│   ├── group_mds_map.png
│   ├── group_similarity_network.png
│   ├── amine_hydroxyl_distances.png
│   └── group_radar_descriptors.png
├── similarity_matrix.npy          # 40×40 output of Approach 1
├── saft_pair_tables.json          # D_{kl} / Δ_{kl} pair tables (Approach 2)
├── ranking_vs_MEA.json            # Example ranking output (Approach 2)
└── README.md
```

---

## Groups of interest

Both scripts operate on the same 20 functional groups, drawn from the SAFT-γ Mie database:

| # | Group | Family | # | Group | Family |
|---|-------|--------|---|-------|--------|
| 1 | `NH2_2nd` | Amine | 11 | `NH` | Amine |
| 2 | `NH_2nd` | Amine | 12 | `N` | Amine |
| 3 | `N_2nd` | Amine | 13 | `OH` | Hydroxyl |
| 4 | `CH3` | Alkyl | 14 | `OH_Short` | Hydroxyl |
| 5 | `CH2` | Alkyl | 15 | `cCH2` | Cycloalkyl |
| 6 | `CH` | Alkyl | 16 | `cCH` | Cycloalkyl |
| 7 | `C` | Alkyl | 17 | `cNH` | Cyclo-N |
| 8 | `CH2OH` | Hydroxyl | 18 | `cN` | Cyclo-N |
| 9 | `CH2OH_Short` | Hydroxyl | 19 | `cCHNH` | Cyclo-N |
| 10 | `NH2` | Amine | 20 | `cCHN` | Cyclo-N |

The `_2nd` and `_Short` variants are re-parameterisations of the same chemical moiety for use in different molecular contexts (e.g. primary vs. secondary amines, or different chain-length environments for hydroxyl groups).

---

## Approach 1 – Feature-based soft-cosine similarity (`compute_similarity.py`)

### What it does

Builds a **20×20 similarity matrix** $\mathbf{S}$ that quantifies how similar each pair of SAFT-γ Mie groups is, based on standardised self- and cross-interaction feature vectors.

### Step-by-step

#### Step 1 — Parse the XML database

Extract per-group properties from `database.xml`:

- Number of segments $\nu_k$, shape factor $S_k$
- Self-dispersion: $\varepsilon_{kk}$, $\sigma_{kk}$, $\lambda^{\mathrm{rep}}_{kk}$, $\lambda^{\mathrm{att}}_{kk}$
- Association site counts: $n_H$, $n_e$, $n_a$
- Self-association totals: $\sum \varepsilon^{\mathrm{assoc}}_{kk}$, $\sum K_{kk}$
- Thermodynamic properties at $T_{\mathrm{ref}} = 298.15\;\mathrm{K}$: heat capacity $C_p^*$, enthalpy of formation $\hat{H}_f$, entropy of formation $\hat{S}_f$

Also parse all explicit cross-interaction entries $(\varepsilon_{kl},\;\lambda^{\mathrm{rep}}_{kl},\;\varepsilon^{\mathrm{assoc}}_{kl},\;K_{kl})$.

#### Step 2 — Build self-feature vector $\boldsymbol{\varphi}^{\mathrm{self}}_k$

For each group $k$, assemble an 11-dimensional vector:

```math
\boldsymbol{\varphi}^{\mathrm{self}}_k = \bigl[\,\nu_k,\; S_k,\; \ln\varepsilon_{kk},\; \ln\sigma_{kk},\; \lambda^{\mathrm{rep}}_{kk},\; \lambda^{\mathrm{att}}_{kk},\; n_H,\; n_e,\; n_a,\; \ln(1+\varepsilon^{\mathrm{assoc}}_{kk}),\; \ln(1+K_{kk})\,\bigr]
```

#### Step 3 — Build cross-feature vector $\boldsymbol{\varphi}^{\mathrm{cross}}_k$

For each group $k$ and every reference group $r$ in the list, extract a 4-dimensional cross fingerprint:

```math
\boldsymbol{f}_{k,r} = \bigl[\,\varepsilon_{kr},\; \lambda^{\mathrm{rep}}_{kr},\; \ln(1+\varepsilon^{\mathrm{assoc}}_{kr}),\; \ln(1+K_{kr})\,\bigr]
```

Concatenate over all 20 reference groups to get $\boldsymbol{\varphi}^{\mathrm{cross}}_k \in \mathbb{R}^{80}$.

#### Step 4 — Standardise and combine

Column-wise z-standardise both matrices, then form the combined feature:

```math
\boldsymbol{\phi}_k = \alpha\,\mathbf{z}^{\mathrm{self}}_k \;\|\; \beta\,\mathbf{z}^{\mathrm{cross}}_k
```

with default weights $\alpha = 0.7$, $\beta = 0.3$.

#### Step 5 — Compute similarity matrix

Normalise each $\boldsymbol{\phi}_k$ to unit length and compute:

```math
S_{ij} = \frac{\hat{\boldsymbol{\phi}}_i \cdot \hat{\boldsymbol{\phi}}_j + 1}{2} \;\in [0,\,1]
```

The diagonal is set to 1, and $\mathbf{S}$ is symmetrised.

#### Step 6 — Expand to 40×40

For two-component mixture use, the matrix is block-duplicated:

```math
\mathbf{S}_{40\times40} = \begin{pmatrix} \mathbf{S} & \mathbf{S} \\ \mathbf{S} & \mathbf{S} \end{pmatrix}
```

and saved as `similarity_matrix.npy`.

---

## Approach 2 – SAFT-γ Mie physics-based molecule ranking (`saft_similarity.py`)

### What it does

Given a **target molecule** (as a group-count vector) and a set of **candidate molecules**, ranks candidates by "nearest thermodynamic behaviour" using five physics-derived scalars — **dispersion strength** $\bar{D}$, **association strength** $\bar{A}$, **chain length** $m$, **packing proxy** $\bar{\sigma}^3$, and **shape average** $\bar{S}$ — without running a full equation of state.

The method strictly follows the SAFT-γ Mie group-contribution framework as described by Papaioannou et al. [1], Dufal et al. [2], and Haslam et al. [3].

### Theoretical background

The SAFT-γ Mie equation of state writes the residual Helmholtz energy as a sum of contributions:

```math
\frac{A^{\mathrm{res}}}{Nk_BT} = \frac{A^{\mathrm{mono}}}{Nk_BT} + \frac{A^{\mathrm{chain}}}{Nk_BT} + \frac{A^{\mathrm{assoc}}}{Nk_BT}
```

Rather than evaluating the full Helmholtz function, this script extracts **proxy quantities** that capture the dominant group-level contributions to each term:

| Helmholtz term | Physical origin | Proxy quantity | Symbol |
|----------------|-----------------|----------------|--------|
| $A^{\mathrm{mono}}$ (dispersion) | Van der Waals attraction between Mie segments | First-order perturbation integral $a_1$ | $D_{kl}$ |
| $A^{\mathrm{chain}}$ | Connectivity of fused segments | Total chain length | $m_i$ |
| $A^{\mathrm{assoc}}$ | Hydrogen bonding via Wertheim TPT1 | Site-summed association strength | $\Delta_{kl}$ |
| HS reference | Excluded volume of segments | Averaged cubic diameter | $\bar{\sigma}^3$ |
| Segment architecture | Fraction of each segment in the chain | Shape-weighted average | $\bar{S}$ |

The workflow below describes how each proxy is computed from the database parameters.

---

### Step-by-step

#### Step 1 — Parse the XML database (Module 1)

From `database.xml` extract:

- **Per-group self-interaction parameters**: $\varepsilon_{kk}$, $\sigma_{kk}$, $\lambda^r_{kk}$, $\lambda^a_{kk}$, number of segments $\nu_k$, shape factor $S_k$
- **Association site multiplicities**: $\{m_{k,a}\}$ — e.g. group `NH2` may have 2 `H`-type sites and 1 `e1`-type site
- **Self-association interactions**: site labels, $\varepsilon^{\mathrm{assoc}}_{kk,ab}$, $K_{kk,ab}$
- **Cross-interaction entries**: explicit unlike-pair dispersion ($\varepsilon_{kl}$, $\sigma_{kl}$, $\lambda^r_{kl}$, $\lambda^a_{kl}$) and association ($\varepsilon^{\mathrm{assoc}}_{kl,ab}$, $K_{kl,ab}$) with site labels

Cross entries are stored with the mirrored pair $(l, k)$ having **swapped site labels**, so that `site1` always refers to the first group in the key.

---

#### Step 2 — Combining rules for missing cross parameters (Module 2)

When the database does not provide an explicit cross entry for a parameter in pair $(k, l)$, **SAFT-γ Mie combining rules** are applied. These rules are split into two categories: **dispersion** (always available) and **association** (applied as a fallback when both groups carry compatible association sites).

##### 2a. Dispersion combining rules

###### Segment diameter — arithmetic mean ([1] Eq. 24)

```math
\sigma_{kl} = \frac{\sigma_{kk} + \sigma_{ll}}{2}
```

###### Mie exponents — nonlinear combining rule ([1] Eq. 23)

Applied independently to both the repulsive and attractive exponents:

```math
\lambda^r_{kl} = 3 + \sqrt{(\lambda^r_{kk} - 3)(\lambda^r_{ll} - 3)}
```

```math
\lambda^a_{kl} = 3 + \sqrt{(\lambda^a_{kk} - 3)(\lambda^a_{ll} - 3)}
```

This rule preserves the constraint $\lambda > 3$ and correctly reduces to the self-interaction value when $k = l$.

###### Dispersion energy — modified Berthelot rule with σ³ correction ([1] Eq. 25)

```math
\varepsilon_{kl} = \sqrt{\varepsilon_{kk}\,\varepsilon_{ll}} \;\cdot\; \frac{\sqrt{\sigma_{kk}^3\,\sigma_{ll}^3}}{\sigma_{kl}^3}
```

The $\sigma^3$ correction ensures thermodynamic consistency: it accounts for the different effective volumes of unlike-sized segments.

###### Priority

If the database provides a non-zero explicit value for any individual parameter ($\varepsilon_{kl}$, $\sigma_{kl}$, $\lambda^r_{kl}$, $\lambda^a_{kl}$), **the database value is used** and the combining rule is only applied to the remaining parameters.

##### 2b. Association combining rules (CR-1 fallback)

Many cross-interaction pairs do not have explicit association entries in the database. When **both groups carry association sites** but no cross entry exists, the code applies SAFT-γ Mie combining rules to estimate the cross-association parameters from the self-association data.

###### Association energy — geometric mean ([1])

```math
\varepsilon^{\mathrm{assoc}}_{kl,ab} = \sqrt{\varepsilon^{\mathrm{assoc}}_{kk,aa}\;\cdot\;\varepsilon^{\mathrm{assoc}}_{ll,bb}}
```

###### Bonding volume — cubic mean of cube roots ([1])

```math
K_{kl,ab} = \left(\frac{\sqrt[3]{K_{kk,aa}} + \sqrt[3]{K_{ll,bb}}}{2}\right)^{\!3}
```

This accounts for the **volumetric** nature of the bonding parameter (units: m³). It differs from a simple arithmetic mean and is the standard SAFT-γ Mie prescription.

###### Site-type canonical aliasing

Different groups may use different string labels for physically identical site types (e.g. `e1` on `CH2OH` and `e` on `CH2OH_Short` both represent the same oxygen lone-pair donor site). To ensure correct matching, all site names are mapped to a **canonical type** before comparison:

| Site name(s) | Canonical type | Physical meaning |
|-------------|----------------|------------------|
| `e`, `e1`, `e2` | `e` | Electron-donor (lone pair) |
| `H` | `H` | Proton donor |
| `a1`, `a2` | `a` | Acceptor |

The combining rule is applied to every pair of self-association interactions whose canonical site types match (in same or swapped order). The resulting synthetic interaction list uses the **actual site names** of each group so that the multiplicity lookup in the association calculation works correctly.

###### Important caveat

The CR-1 combining rules systematically underestimate cross-association for groups with genuinely different chemistries (e.g. OH ↔ NH₂: CR-1 gives ~60–100% below the database fitted value). However, they are **exact** for variant pairs of the same moiety (e.g. `CH2OH` ↔ `CH2OH_Short`), which is the primary use case — preventing zero-association artefacts between groups that clearly should interact.

---

#### Step 3 — Mie prefactor $C_{kl}$ (Module 3a)

The Mie potential between segments of groups $k$ and $l$ is:

```math
u^{\mathrm{Mie}}(r) = C_{kl}\,\varepsilon_{kl}\left[\left(\frac{\sigma_{kl}}{r}\right)^{\!\lambda^r_{kl}} - \left(\frac{\sigma_{kl}}{r}\right)^{\!\lambda^a_{kl}}\right]
```

where the normalisation constant ensures $u(\sigma) = -\varepsilon$ at the minimum ([1] Eq. 2):

```math
C_{kl} = \frac{\lambda^r_{kl}}{\lambda^r_{kl} - \lambda^a_{kl}} \left(\frac{\lambda^r_{kl}}{\lambda^a_{kl}}\right)^{\!\lambda^a_{kl} / (\lambda^r_{kl} - \lambda^a_{kl})}
```

---

#### Step 4 — Dispersion proxy $D_{kl}$: first-order perturbation (Module 3a)

The dominant contribution to intermolecular attraction in SAFT comes from the first-order monomer perturbation term $a_1$ ([1] Eqs. 19–20). For each group pair we compute a **proxy** proportional to $a_1$:

```math
D_{kl} = C_{kl}\,\varepsilon_{kl}\,\sigma_{kl}^3 \left[a_1^S(\eta;\,\lambda^a_{kl}) - a_1^S(\eta;\,\lambda^r_{kl})\right]
```

where $a_1^S(\eta;\lambda)$ is the Sutherland-$\lambda$ perturbation integral evaluated at a fixed reference packing fraction $\eta_{\mathrm{ref}}$.

##### Sutherland perturbation integral

In the full SAFT-γ Mie theory, the Sutherland integral $a_1^S$ is parameterised by a polynomial in $\eta$ with $\lambda$-dependent coefficients. For the **proxy** we use the leading-order mean-field / contact-value approximation:

```math
a_1^S(\eta;\lambda) \approx -\frac{1}{\lambda - 3}\;\frac{1 - \eta/2}{(1 - \eta)^3}
```

This captures the dominant dependence on both $\lambda$ (potential range) and $\eta$ (packing), and is exact in the van-der-Waals-1-fluid limit.

##### Reference packing fraction

```math
\eta_{\mathrm{ref}} = 0.40
```

This is a liquid-like reference state; the proxy does not depend on the actual density of the system. Any monotonic change of $\eta_{\mathrm{ref}}$ rescales all $D_{kl}$ values by the same factor and therefore does not affect the ranking.

---

#### Step 5 — Association proxy $\Delta_{kl}$: Wertheim TPT1 strength (Module 3b)

Hydrogen bonding is described via Wertheim's thermodynamic perturbation theory of first order (TPT1), as formulated in SAFT-γ Mie ([1] Eqs. 36–38).

##### Single site-pair interaction

For a specific pair of association sites $a$ on group $k$ and $b$ on group $l$:

```math
\Delta_{kl,ab}(T, \eta) = F_{kl,ab}\;\cdot\;K_{kl,ab}\;\cdot\;I_{kl}(\eta)
```

where:

**Mayer-f function** — the Boltzmann factor for the square-well association potential:

```math
F_{kl,ab} = \exp\!\left(\frac{\varepsilon^{\mathrm{assoc}}_{kl,ab}}{T}\right) - 1
```

**Bonding volume** — the geometric parameter $K_{kl,ab}$ (units: m³), taken from the database or estimated via combining rules (Step 2b).

**Association kernel** — in the full theory, $I_{kl}$ involves the Mie radial distribution function $g^{\mathrm{Mie}}(\sigma_{kl})$ evaluated at contact distance. For the proxy we use the leading hard-sphere (Carnahan–Starling) contact value:

```math
I_{kl} \approx g^{HS}(\sigma_{kl};\,\eta) = \frac{1 - \eta/2}{(1 - \eta)^3}
```

Higher-order $a_1$/$a_2$ corrections to $g^{\mathrm{Mie}}$ require the full Helmholtz machinery and are omitted.

##### Total pair association strength

Sum over all site-site interactions, weighted by site multiplicities ([1] Eq. 38):

```math
\Delta_{kl}(T, \eta) = \sum_{a \in k}\;\sum_{b \in l} m_{k,a}\;m_{l,b}\;\Delta_{kl,ab}
```

where $m_{k,a}$ is the multiplicity of site type $a$ on group $k$ (e.g., `NH2` has $m_{H}=2$, $m_{e1}=1$).

##### Source of association parameters

The association interaction list for a pair $(k, l)$ is resolved as follows:

1. **Self-pair** ($k = l$): use the group's own self-association list from the database.
2. **Cross-pair with explicit database entry**: use the database cross-association list.
3. **Cross-pair without database entry, but both groups have association sites**: apply the **CR-1 combining-rule fallback** (Step 2b) to estimate $\varepsilon^{\mathrm{assoc}}_{kl}$ and $K_{kl}$ from the self-association parameters.
4. **No association sites on either group**: $\Delta_{kl} = 0$.

##### Reference temperature

```math
T_{\mathrm{ref}} = 298.15\;\mathrm{K}
```

---

#### Step 6 — Build pair tables (Module 4)

Pre-compute $D_{kl}$ and $\Delta_{kl}(T_{\mathrm{ref}}, \eta_{\mathrm{ref}})$ for every unordered pair among the 20 groups of interest, yielding two symmetric $20 \times 20$ tables plus a parameter table:

| Table | Contents | Units |
|-------|----------|-------|
| $D_{kl}$ | Dispersion a₁-proxy | J·m³ (all values negative — net attractive) |
| $\Delta_{kl}$ | Association strength | m³ (zero for non-associating pairs) |
| param_table | Resolved $\varepsilon_{kl}$, $\sigma_{kl}$, $\lambda^r_{kl}$, $\lambda^a_{kl}$ | K, m, dimensionless |

These tables are exported as `saft_pair_tables.json`.

---

#### Step 7 — Segment fractions $x_{s,k}$ (Module 5)

SAFT-γ Mie treats molecules as heteronuclear chains of fused segments. The total chain length of molecule $i$ is ([1] Eqs. 7–8):

```math
m_i = \sum_k n_k \,\nu_k \,S_k
```

where $n_k$ is the multiplicity of group $k$ in the molecule, $\nu_k$ is the number of identical segments per group, and $S_k$ is the shape factor (the fraction of segment $k$ that contributes to the chain).

The **segment fraction** of group $k$ in molecule $i$ is:

```math
x_{s,k} = \frac{n_k\,\nu_k\,S_k}{m_i}
```

These sum to unity: $\sum_k x_{s,k} = 1$.

---

#### Step 8 — Molecule signature (Module 5)

Each molecule is characterised by a **5-scalar thermodynamic signature**:

##### 1. Dispersion  $\bar{D}$

The pair-averaged dispersion, using the same double-sum structure as the monomer Helmholtz contribution ([1] Eq. 19):

```math
\bar{D}_i = \sum_k \sum_l x_{s,k}\;x_{s,l}\;D_{kl}
```

**Physical meaning**: measures the overall cohesive (van der Waals) attraction of the molecule. Higher $|\bar{D}|$ → stronger dispersion → higher boiling point, lower vapour pressure.

##### 2. Association  $\bar{A}$

```math
\bar{A}_i = \sum_k \sum_l x_{s,k}\;x_{s,l}\;\Delta_{kl}
```

**Physical meaning**: measures the total hydrogen-bonding capacity of the molecule. Molecules with $\bar{A} \gg 0$ (amines, alcohols, water) behave very differently from non-associating molecules ($\bar{A} \approx 0$, alkanes) in terms of excess mixing properties, activity coefficients, and heat of absorption.

##### 3. Chain length  $m$

```math
m_i = \sum_k n_k\,\nu_k\,S_k
```

**Physical meaning**: effective number of spherical segments in the chain. Distinguishes molecules with identical group types but different multiplicities (e.g. ethane vs propane vs butane; cyclopentane vs cyclohexane).

##### 4. Packing proxy  $\bar{\sigma}^3$

```math
\bar{\sigma}^3_i = \sum_k \sum_l x_{s,k}\;x_{s,l}\;\sigma_{kl}^3
```

**Physical meaning**: proportional to the effective excluded volume of the molecule's segments. Captures **size differences** between molecules that share similar dispersion energies but differ in segment diameter. In the full SAFT-γ Mie theory, $\sigma_{kl}^3$ appears in the hard-sphere reference term and in the $\varepsilon$ combining rule.

##### 5. Shape average  $\bar{S}$

```math
\bar{S}_i = \sum_k x_{s,k}\;S_k
```

**Physical meaning**: the fraction of each segment that participates in the fused chain, averaged over the molecule. A single sum (not a double sum) is used because the shape factor is a **per-group** property, not a pair interaction. Groups with $S_k < 1$ contribute less to the chain connectivity than groups with $S_k \approx 1$.

##### Key properties

- Because the double sums use *segment fractions* (which normalise to 1), molecules composed of a single group type always yield $\bar{D} = D_{kk}$, $\bar{A} = \Delta_{kk}$, and $\bar{\sigma}^3 = \sigma_{kk}^3$ regardless of how many copies of that group are present.
- The chain length $m_i$, packing proxy $\bar{\sigma}^3_i$, and shape average $\bar{S}_i$ together distinguish molecules with identical group *types* but different *sizes* or *architectures*.

---

#### Step 9 — Log-Euclidean distance (Module 6)

Compare a candidate signature against the target using a **log-Euclidean metric** in 5-D signature space:

```math
d_D = \ln\frac{|\bar{D}_c|}{|\bar{D}_t|}, \qquad d_A = \ln\frac{\bar{A}_c + S_0}{\bar{A}_t + S_0}, \qquad d_m = \ln\frac{m_c}{m_t}
```

```math
d_{\sigma} = \ln\frac{\bar{\sigma}^3_c}{\bar{\sigma}^3_t}, \qquad d_S = \ln\frac{\bar{S}_c}{\bar{S}_t}
```

```math
\mathcal{D} = \sqrt{w_D\,d_D^2 \;+\; w_A\,d_A^2 \;+\; w_m\,d_m^2 \;+\; w_{\sigma}\,d_{\sigma}^2 \;+\; w_S\,d_S^2}
```

##### Why logarithmic?

The five signature components span many orders of magnitude ($D \sim 10^{-26}$, $A \sim 10^{-25}$, $m \sim 1$, $\sigma^3 \sim 10^{-29}$, $S \sim 0.5$). A direct Euclidean distance would be dominated by whichever component has the largest absolute value. The logarithm converts multiplicative ratios into additive differences, making the metric **scale-invariant**: doubling $\bar{D}$ contributes the same $|\ln 2|$ regardless of the absolute magnitude.

##### Association floor $S_0$

```math
S_0 = 5 \times 10^{-29}\;\mathrm{m^3}
```

This prevents $\ln(0)$ singularities when one or both molecules have zero association ($\bar{A} = 0$). Physically, $S_0$ represents a negligible background association strength much smaller than any real H-bonding interaction ($\Delta \sim 10^{-26}$ to $10^{-25}$), so it does not distort the ranking among associating molecules.

##### Weights

Default values:

| Weight | Symbol | Default | Role |
|--------|--------|---------|------|
| Dispersion | $w_D$ | 1.0 | Mie attraction strength |
| Association | $w_A$ | 3.0 | H-bonding — upweighted as it is the primary differentiator for amine/alkanolamine screening |
| Chain length | $w_m$ | 0.7 | Molecular size — downweighted to avoid over-penalising small size differences |
| Packing | $w_{\sigma}$ | 1.0 | Segment excluded volume |
| Shape | $w_S$ | 0.5 | Shape factor contribution — downweighted as it varies less across the candidate set |

These can be adjusted to emphasise specific physical aspects. An alternative **inverse-variance** weighting mode is also implemented, which automatically normalises each component by its variance across the candidate set so that all features contribute equally on average.

---

#### Step 10 — Rank candidates (Module 6)

Sort all candidates by ascending $\mathcal{D}$. The output includes each candidate's group-count vector, full 5-scalar signature, and distance. The ranking is exported as `ranking_vs_MEA.json`.

---

### Summary of assumptions and approximations

| # | Assumption | Justification |
|---|-----------|---------------|
| 1 | **No full Helmholtz evaluation** — only proxy quantities $D_{kl}$ and $\Delta_{kl}$ are computed | We need *relative* ordering, not absolute thermodynamic properties; the leading-order terms dominate the ranking |
| 2 | **Sutherland $a_1^S$ in mean-field limit** — polynomial parameterisation replaced by contact-value expression | The mean-field form captures the correct $\lambda$ and $\eta$ dependence; polynomial coefficients are not needed for a proxy |
| 3 | **Association kernel $I_{kl} \approx g^{HS}$** — Mie RDF reduced to Carnahan–Starling hard-sphere contact value | The HS contribution is the dominant term; $a_1$/$a_2$ corrections to the RDF require the full EOS and are state-dependent |
| 4 | **Fixed reference state** $(\eta_{\mathrm{ref}} = 0.40,\;T_{\mathrm{ref}} = 298.15\;\mathrm{K})$ | The proxy is evaluated at a single liquid-like state point; any monotonic rescaling by $\eta$ cancels in the log-ratio distance |
| 5 | **Segment fractions** $x_{s,k}$ used instead of intramolecular pair counts | Consistent with SAFT-γ Mie monomer contribution formalism ([1] Eqs. 7–8, 19) |
| 6 | **Chain length** $m_i$ included as a distance component | Necessary to distinguish molecules with identical group types but different multiplicities; $m$ enters SAFT via the chain term $A^{\mathrm{chain}}$ |
| 7 | **Packing proxy** $\bar{\sigma}^3_i$ included as a distance component | Captures segment-size differences; $\sigma^3$ appears in the HS reference and the $\varepsilon$ combining rule |
| 8 | **Shape average** $\bar{S}_i$ included as a distance component | The shape factor $S_k$ modulates how each group contributes to the chain; averaging over segment fractions gives a molecular-level architecture indicator |
| 9 | **Dispersion cross parameters**: database values have priority; combining rules used as fallback | Standard practice in SAFT-γ Mie; the nonlinear $\lambda$ and $\sigma^3$-corrected $\varepsilon$ rules are physically motivated |
| 10 | **Association cross parameters**: database values have priority; CR-1 combining rules (geometric mean $\varepsilon^{\mathrm{assoc}}$, cubic-mean-of-cube-roots $K$) used as fallback when both groups carry compatible sites | Prevents zero-association artefacts for variant pairs (e.g. `CH2OH` ↔ `CH2OH_Short`); systematically underestimates for genuinely different groups, which is conservative |
| 11 | **Site-type canonical aliasing** (`e1` ↔ `e`, `a1` ↔ `a`, etc.) | Required because different groups may label the same physical site type differently; without aliasing, compatible sites would not be matched by the combining rules |

---

## Visualisation – Group-level similarity (`plot_group_similarity.py`)

This script reuses the **same SAFT-γ Mie pair tables** computed by `saft_similarity.py` — specifically $D_{kl}$, $\Delta_{kl}$, and $\sigma_{kl}$ — to visualise how similar the 20 functional groups are to each other.

The visualisation operates at the **group level** (not the molecule level): it asks "how differently do two groups interact with each other compared to how they interact with themselves?" This is complementary to the molecule-level ranking in Approach 2.

All figures are saved to `figures/`.

### Group-group distance metric

The distance between two groups $k$ and $l$ measures how much their **cross interaction** deviates from the geometric mean of their self interactions. Three components, each a log-ratio:

```math
d_D = \ln \frac{|D_{kl}|}{\sqrt{|D_{kk}|\,|D_{ll}|}}
```

```math
d_{\Delta} = \ln \frac{\Delta_{kl} + S_0}{\sqrt{(\Delta_{kk} + S_0)(\Delta_{ll} + S_0)}}
```

```math
d_{\sigma} = \ln \frac{\sigma_{kl}^3}{\sqrt{\sigma_{kk}^3\,\sigma_{ll}^3}}
```

```math
d_{kl} = \sqrt{d_D^2 + d_{\Delta}^2 + d_{\sigma}^2}
```

**Physical interpretation**: each ratio measures *thermodynamic compatibility*. When the cross quantity matches the geometric mean of the self quantities ($d = 0$), the two groups interact with each other just as strongly as they interact with themselves — they are perfectly interchangeable in the SAFT-γ Mie sense. Positive/negative deviations indicate the cross interaction is stronger/weaker than what the geometric-mean reference would predict.

For self-pairs ($k = l$), all ratios are 1 and $d_{kk} = 0$ by construction.

The same floor $S_0 = 5 \times 10^{-29}\;\mathrm{m^3}$ is used for association.

---

### Figure 1 — Distance heatmap

![Group distance heatmap](figures/group_distance_heatmap.png)

**What it shows**: the full $20 \times 20$ symmetric matrix of pairwise group distances $d_{kl}$, with numerical annotations in each cell.

**How to read it**:
- **Yellow cells** (low values) indicate groups that are near-interchangeable — their cross interactions closely follow the geometric-mean rule. For example, alkyl groups (`CH3`, `CH2`, `CH`) tend to cluster in a yellow block because they are chemically similar.
- **Red cells** (high values) indicate groups whose cross interaction deviates strongly from the geometric mean — they are thermodynamically very different. For instance, a hydroxyl group (`OH`) versus a quaternary carbon (`C`) will appear red.
- **Diagonal** is always 0 (white/yellow).

Rows and columns are reordered by a **greedy nearest-neighbour chain** (starting from the first group, repeatedly jump to the nearest unvisited group) so that similar groups cluster together visually.

---

### Figure 2 — 2-D MDS embedding

![MDS map](figures/group_mds_map.png)

**What it shows**: the 20 groups projected into 2 dimensions so that their pairwise distances in the plot approximate the true $d_{kl}$ distances as closely as possible.

**Method**: classical (metric) Multi-Dimensional Scaling (Torgerson, 1952):

1. Square the distance matrix: $\mathbf{D}^{(2)}$
2. Double-centre: $\mathbf{B} = -\tfrac{1}{2}\,\mathbf{H}\,\mathbf{D}^{(2)}\,\mathbf{H}$, where $\mathbf{H} = \mathbf{I} - \tfrac{1}{N}\mathbf{11}^\top$
3. Eigendecompose $\mathbf{B}$ and take the two largest eigenvalues $\lambda_1, \lambda_2$ with corresponding eigenvectors $\mathbf{v}_1, \mathbf{v}_2$
4. Coordinates: $\mathbf{X} = [\mathbf{v}_1\sqrt{\lambda_1},\;\mathbf{v}_2\sqrt{\lambda_2}]$

**How to read it**:
- Groups that are **close together** in the plot have similar SAFT-γ Mie cross interactions — they can be substituted for each other with minimal thermodynamic impact.
- Groups that are **far apart** have very different interaction profiles.
- **Colour** indicates chemical family (Alkyl = blue, Cycloalkyl = green, Hydroxyl = red, Amine = purple, Cyclo-N = gold).
- Light grey edges connect the closest 30% of pairs, giving a visual sense of the neighbourhood structure.

**Typical patterns**: alkyl groups form a tight cluster; hydroxyl and amine groups separate from the alkyl cluster but may overlap with each other if their association strengths are similar; cyclo-N groups sit between the amine and cycloalkyl clusters.

---

### Figure 3 — Similarity network

![Similarity network](figures/group_similarity_network.png)

**What it shows**: a force-directed graph where each group is a node, and edges connect each group to its **4 nearest neighbours** (4-NN).

**Method**: Fruchterman–Reingold force-directed layout, initialised from the MDS coordinates. Repulsive forces push all nodes apart; attractive forces pull connected (nearest-neighbour) nodes together. The algorithm iterates 500 times to find an equilibrium layout.

**How to read it**:
- **Edge thickness** $\propto 1/d$ — thicker lines mean more similar groups.
- **Edge opacity** $\propto 1/d$ — faint lines mean distant groups.
- **Midpoint labels** show the numerical distance value on each edge.
- Groups with many thick, opaque connections to their neighbours are well-embedded in a cluster of similar groups. Groups with only thin, faint connections are "outliers" in the SAFT-γ Mie parameter space.

**What it reveals**: the network highlights the local neighbourhood structure more clearly than the MDS map. For example, you can directly see that `NH2` connects most strongly to `NH2_2nd` (its variant) and to `OH` (similar association), while `C` connects only weakly to its neighbours because its small dispersion energy sets it apart from the other alkyl groups.

---

### Figure 4 — Amine & hydroxyl distance ranking

![Amine/hydroxyl distances](figures/amine_hydroxyl_distances.png)

**What it shows**: a horizontal bar chart of **all pairwise distances** among 14 nitrogen- and oxygen-containing groups, sorted from most similar (top) to most different (bottom).

**How to read it**:
- **Short bars** (top) = very similar group pairs. These are candidates for substitution in molecular design — replacing one group with the other will minimally perturb the molecule's SAFT-γ Mie thermodynamic behaviour.
- **Long bars** (bottom) = very different pairs. Substituting one for the other will significantly change dispersion, association, or both.
- **Bar colour** indicates pair type:
  - **Purple** — Amine ↔ Amine
  - **Red** — Hydroxyl ↔ Hydroxyl
  - **Gold** — Amine ↔ Hydroxyl (cross-family)

**What it answers directly**: "How far is `NH2` from `NH`?", "Is `NH2` more similar to `OH` or to `N`?", "Which hydroxyl variant is closest to which amine?"

---

### Figure 5 — Radar chart of group descriptors

![Radar chart](figures/group_radar_descriptors.png)

**What it shows**: a spider (radar) plot of 6 **self-pair** SAFT-γ Mie descriptors for a representative subset of groups, all min-max normalised to $[0, 1]$.

| Axis | Quantity | Source |
|------|----------|--------|
| $\varepsilon_{kk}$ | Dispersion well depth | Database self-interaction |
| $\sigma_{kk}$ | Segment diameter | Database self-interaction |
| $\lambda^r_{kk}$ | Repulsive exponent | Database self-interaction |
| $\lambda^a_{kk}$ | Attractive exponent | Database self-interaction |
| $|D_{kk}|$ | Dispersion proxy (a₁) | Computed (Step 4) |
| $\Delta_{kk}$ | Self-association strength | Computed (Step 5) |

**How to read it**:
- Each group is a coloured polygon. The polygon's shape reveals the group's SAFT "fingerprint".
- Groups with similar polygon shapes are similar in their fundamental SAFT parameters.
- The $\Delta_{kk}$ axis is the most discriminating: associating groups (`NH2`, `OH`) extend far on this axis, while alkyl groups (`CH3`, `CH2`) collapse to zero.
- The $\varepsilon_{kk}$ axis separates strongly interacting groups (e.g. `OH` with high well depth) from weakly interacting ones (e.g. `CH3`).
- Differences in $\lambda^r_{kk}$ indicate how "hard" or "soft" the repulsive wall is — groups with high $\lambda^r$ have a steeper, more hard-sphere-like repulsion.

**Limitation**: this plot only shows **self-pair** descriptors. The cross-interaction information (which is what determines the group-group distance in Figures 1–4) is not visible here. The radar chart is best used as a companion to the other plots, to understand *why* two groups are similar or different.

---

## Running the scripts

```bash
# Approach 1: build the 40×40 similarity matrix
python scripts/compute_similarity.py

# Approach 2: rank compounds against MEA (default target)
python scripts/saft_similarity.py

# Visualisation: generate all 5 group-similarity figures
python scripts/plot_group_similarity.py
```

### Outputs

| File | Description |
|------|-------------|
| `similarity_matrix.npy` | 40×40 group similarity matrix (Approach 1) |
| `similarity_matrix_plot.png` | Heatmap visualisation of $\mathbf{S}$ |
| `saft_pair_tables.json` | $D_{kl}$ and $\Delta_{kl}$ pair tables (Approach 2) |
| `ranking_vs_MEA.json` | Full ranking with 5-scalar signatures, weights, and distances |
| `figures/group_distance_heatmap.png` | Group-group distance heatmap |
| `figures/group_mds_map.png` | 2-D MDS embedding of groups |
| `figures/group_similarity_network.png` | 4-NN similarity network graph |
| `figures/amine_hydroxyl_distances.png` | Amine/hydroxyl pairwise distance bar chart |
| `figures/group_radar_descriptors.png` | Radar chart of raw SAFT descriptors |

---

## Database format

The XML database (`database/database.xml`) follows the SAFT-γ Mie group-contribution schema:

- **`<compounds>`** — molecule definitions as lists of group multiplicities
- **`<groups>`** — per-group self-interaction parameters:
  - `<numberOfSegments>` ($\nu_k$), `<shapeFactor>` ($S_k$)
  - `<selfInteraction><dispersion>`: $\varepsilon_{kk}$, $\sigma_{kk}$, $\lambda^r_{kk}$, $\lambda^a_{kk}$
  - `<selfInteraction><association>`: site-pair interactions with $\varepsilon^{\mathrm{assoc}}_{kk,ab}$, $K_{kk,ab}$
  - `<associationSites>`: site multiplicities $m_{k,a}$
- **`<crossInteractions>`** — explicit unlike-pair parameters (dispersion and association)

---

## References

1. V. Papaioannou, T. Lafitte, C. Avendaño, C. S. Adjiman, G. Jackson, E. A. Müller, A. Galindo, *J. Chem. Phys.* **140**, 054107 (2014).
2. S. Dufal, T. Lafitte, A. J. Haslam, A. Galindo, G. N. I. Clark, C. Vega, G. Jackson, *J. Chem. Eng. Data* **59**, 3272–3288 (2014).
3. A. J. Haslam, A. Galindo, G. Jackson, "SAFT-γ Mie group-contribution framework" — comprehensive review of the group-contribution methodology.

---

## Dependencies

- Python ≥ 3.10
- `numpy`
- `matplotlib` (Approach 1 plotting + group-similarity visualisation)
