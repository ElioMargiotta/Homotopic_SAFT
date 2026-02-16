# SAFT-γ Mie Group Similarity & Molecule Ranking

Composition-agnostic similarity metrics for molecules represented by **SAFT-γ Mie group-contribution** vectors.  
Two complementary approaches are implemented, both reading from the same XML parameter database.

---

## Repository layout

```
├── database/
│   └── database.xml          # SAFT-γ Mie group parameters (self + cross)
├── scripts/
│   ├── compute_similarity.py  # Approach 1 – feature-based cosine similarity matrix
│   └── saft_similarity.py     # Approach 2 – physics-derived J/S ranking
├── similarity_matrix.npy      # 40×40 output of Approach 1
├── saft_pair_tables.json      # J_disp / S_HB pair tables (Approach 2)
├── ranking_vs_MEA.json        # Example ranking output (Approach 2)
└── README.md
```

---

## Groups of interest

Both scripts operate on the same 20 functional groups:

| # | Group | # | Group |
|---|-------|---|-------|
| 1 | `NH2_2nd` | 11 | `NH` |
| 2 | `NH_2nd` | 12 | `N` |
| 3 | `N_2nd` | 13 | `OH` |
| 4 | `CH3` | 14 | `OH_Short` |
| 5 | `CH2` | 15 | `cCH2` |
| 6 | `CH` | 16 | `cCH` |
| 7 | `C` | 17 | `cNH` |
| 8 | `CH2OH` | 18 | `cN` |
| 9 | `CH2OH_Short` | 19 | `cCHNH` |
| 10 | `NH2` | 20 | `cCHN` |

---

## Approach 1 – Feature-based soft-cosine similarity (`compute_similarity.py`)

### What it does

Builds a **20×20 similarity matrix** $\mathbf{S}$ that quantifies how similar each pair of SAFT-γ Mie groups is, based on standardised self- and cross-interaction feature vectors.

### Step-by-step

#### Step 1 — Parse the XML database

Extract per-group properties from `database.xml`:

- Number of segments $m_k$, shape factor $S_k$
- Self-dispersion: $\varepsilon_{kk}$, $\sigma_{kk}$, $\lambda^{\mathrm{rep}}_{kk}$, $\lambda^{\mathrm{att}}_{kk}$
- Association site counts: $n_H$, $n_e$, $n_a$
- Self-association totals: $\sum \varepsilon^{\mathrm{assoc}}_{kk}$, $\sum K_{kk}$
- Thermodynamic properties at $T_{\mathrm{ref}} = 298.15\;\mathrm{K}$: heat capacity $C_p^*$, enthalpy of formation $\hat{H}_f$, entropy of formation $\hat{S}_f$

Also parse all explicit cross-interaction entries $(\varepsilon_{kl},\;\lambda^{\mathrm{rep}}_{kl},\;\varepsilon^{\mathrm{assoc}}_{kl},\;K_{kl})$.

#### Step 2 — Build self-feature vector $\boldsymbol{\varphi}^{\mathrm{self}}_k$

For each group $k$, assemble an 11-dimensional vector:

```math
\boldsymbol{\varphi}^{\mathrm{self}}_k = \bigl[\,m_k,\; S_k,\; \ln\varepsilon_{kk},\; \ln\sigma_{kk},\; \lambda^{\mathrm{rep}}_{kk},\; \lambda^{\mathrm{att}}_{kk},\; n_H,\; n_e,\; n_a,\; \ln(1+\varepsilon^{\mathrm{assoc}}_{kk}),\; \ln(1+K_{kk})\,\bigr]
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

## Approach 2 – Physics-based molecule ranking (`saft_similarity.py`)

### What it does

Given a **target molecule** (as a group-count vector) and a set of **candidate molecules**, ranks candidates by "nearest thermodynamic behaviour" using two physics-derived scalars—**dispersion strength** $\bar{J}$ and **association strength** $\bar{S}$—without running a full equation of state.

### Step-by-step

#### Step 1 — Parse the XML database

Same XML, but this time we also retain:

- Per-group **association site multiplicities** $m_{k,a}$ (e.g. site `H` with multiplicity 2)
- Per-interaction **site labels** (`site1`, `site2`) so that cross-association terms can be correctly weighted

#### Step 2 — Combining rules for missing cross parameters

When the database does not provide an explicit cross entry for a pair $(k, l)$:

| Parameter | Combining rule |
|-----------|---------------|
| $\sigma_{kl}$ | $\displaystyle\sigma_{kl} = \frac{\sigma_{kk}+\sigma_{ll}}{2}$ |
| $\lambda^{\mathrm{rep}}_{kl}$ | $\displaystyle\lambda^{\mathrm{rep}}_{kl} = \frac{\lambda^{\mathrm{rep}}_{kk}+\lambda^{\mathrm{rep}}_{ll}}{2}$ |
| $\varepsilon_{kl}$ | $\displaystyle\varepsilon_{kl} = \sqrt{\varepsilon_{kk}\,\varepsilon_{ll}}$ |
| $\lambda^{\mathrm{att}}_{kl}$ | $6.0$ (default) |

#### Step 3 — Mie prefactor $C_{kl}$

```math
C_{kl} = \frac{\lambda^{\mathrm{rep}}_{kl}}{\lambda^{\mathrm{rep}}_{kl} - \lambda^{\mathrm{att}}_{kl}} \left(\frac{\lambda^{\mathrm{rep}}_{kl}}{\lambda^{\mathrm{att}}_{kl}}\right)^{\!\lambda^{\mathrm{att}}_{kl}/(\lambda^{\mathrm{rep}}_{kl}-\lambda^{\mathrm{att}}_{kl})}
```

#### Step 4 — Dispersion scalar $J^{\mathrm{disp}}_{kl}$

Integrated Mie well depth:

```math
J^{\mathrm{disp}}_{kl} = 4\pi\,C_{kl}\,\varepsilon_{kl}\,\sigma_{kl}^{\,3}\left(\frac{1}{\lambda^{\mathrm{att}}_{kl}-3} - \frac{1}{\lambda^{\mathrm{rep}}_{kl}-3}\right)
```

Valid for $\lambda^{\mathrm{rep}}_{kl} > 3$ and $\lambda^{\mathrm{att}}_{kl} > 3$.

#### Step 5 — Association scalar $S^{\mathrm{HB}}_{kl}$

For each site-pair interaction $(a, b)$ between groups $k$ and $l$:

```math
\Delta_{kl}^{ab}(T) = K_{kl}^{ab}\left(\exp\!\left(\frac{\varepsilon^{\mathrm{assoc},\,ab}_{kl}}{T}\right) - 1\right)
```

Sum over all site pairs, weighted by site multiplicities:

```math
S^{\mathrm{HB}}_{kl}(T) = \sum_{a \in k}\sum_{b \in l} m_{k,a}\;m_{l,b}\;\Delta_{kl}^{ab}(T)
```

If no association interaction exists between $k$ and $l$, then $S^{\mathrm{HB}}_{kl} = 0$.

#### Step 6 — Build pair tables

Pre-compute $J^{\mathrm{disp}}_{kl}$ and $S^{\mathrm{HB}}_{kl}(T_{\mathrm{ref}})$ for every unordered pair among the 20 groups of interest, yielding two symmetric $20 \times 20$ tables.

#### Step 7 — Molecule signature $\{\bar{J},\;\bar{S}\}$

A molecule is described by a group-count vector $\mathbf{n} = (n_1, \dots, n_G)$.  
Total groups: $N = \sum_k n_k$, total unordered pairs: $N_{\mathrm{pairs}} = N(N-1)/2$.

Unordered pair counts:

```math
N_{kl} = \begin{cases} n_k\,(n_k - 1)/2 & k = l \\ n_k\,n_l & k \neq l \end{cases}
```

Pair-averaged signatures:

```math
\bar{J} = \frac{1}{N_{\mathrm{pairs}}}\sum_{k \le l} N_{kl}\,J^{\mathrm{disp}}_{kl}, \qquad \bar{S} = \frac{1}{N_{\mathrm{pairs}}}\sum_{k \le l} N_{kl}\,S^{\mathrm{HB}}_{kl}
```

#### Step 8 — Log-Euclidean distance

Compare a candidate signature $(\bar{J}_c,\,\bar{S}_c)$ against the target $(\bar{J}_t,\,\bar{S}_t)$:

```math
D = \sqrt{w_J\!\left(\ln\frac{\bar{J}_c}{\bar{J}_t}\right)^{\!2} + w_S\!\left(\ln\frac{\bar{S}_c}{\bar{S}_t}\right)^{\!2}}
```

Default weights: $w_J = w_S = 1$.

**Edge cases:**
- If $\bar{S}_t = 0$ or $\bar{S}_c = 0$, a floor $S_0 = 10^{-27}$ is added before taking the logarithm.
- If $N < 2$ (molecule has fewer than 2 groups), the signature is $(0, 0)$.

#### Step 9 — Rank candidates

Sort all candidates by ascending $D$. The output includes each candidate's group-count vector, signature, and distance.

---

## Running the scripts

```bash
# Approach 1: build the 40×40 similarity matrix
python scripts/compute_similarity.py

# Approach 2: rank compounds against MEA (default target)
python scripts/saft_similarity.py
```

### Outputs

| File | Description |
|------|-------------|
| `similarity_matrix.npy` | 40×40 group similarity matrix (Approach 1) |
| `similarity_matrix_plot.png` | Heatmap visualisation of $\mathbf{S}$ |
| `saft_pair_tables.json` | $J^{\mathrm{disp}}_{kl}$ and $S^{\mathrm{HB}}_{kl}$ tables (Approach 2) |
| `ranking_vs_MEA.json` | Full ranking with signatures and distances |

---

## Database format

The XML database (`database/database.xml`) follows the SAFT-γ Mie group-contribution schema:

- **`<compounds>`** — molecule definitions as lists of group multiplicities
- **`<groups>`** — per-group self-interaction parameters (dispersion, association, thermodynamic)
- **`<crossInteractions>`** — explicit unlike-pair parameters (dispersion ε, λ, association ε/K)

---

## Dependencies

- Python ≥ 3.8
- `numpy`
- `matplotlib` (only for Approach 1 plotting)