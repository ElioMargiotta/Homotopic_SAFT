# Association in SAFT-γ Mie: Theory, Implementation, and Current Limitations

## 1. What is association?

In SAFT, **association** refers to strong, short-range, directional interactions between specific sites on molecules — primarily hydrogen bonds. Unlike dispersion (which acts between all segments), association only occurs between **complementary site pairs**: an electron donor (e-site) and a proton donor (H-site).

Examples:

| Molecule | Groups | Association sites |
|----------|--------|-------------------|
| MEA (monoethanolamine) | NH₂ + CH₂ + CH₂OH | NH₂: e1 + H (donor+acceptor), CH₂OH: e1 + H |
| DEA (diethanolamine) | NH + 2×CH₂ + 2×CH₂OH | NH: e1 + H, CH₂OH: e1 + H |
| MDEA (methyldiethanolamine) | N + 3×CH₂ + 2×CH₂OH | N: e1 only (acceptor, no H), CH₂OH: e1 + H |
| Hexane | 2×CH₃ + 4×CH₂ | None → F_assoc = 0 |

The key distinction: a **primary amine** (NH₂) has both e and H sites and can **self-associate** (one molecule's H bonds to another's e). A **tertiary amine** (N) has only an e-site — it cannot self-associate, but it *can* accept H-bonds from other groups like OH or NH₂. This asymmetry is central to the problems discussed in §3.

---

## 2. How is it calculated?

### 2.1 The Wertheim TPT1 free energy (Papaioannou Eq. 64)

The association contribution to the Helmholtz free energy is:

```
A^assoc / NkBT = Σ_k  ν_{k,i}  Σ_a  n_{k,a} [ ln X_{k,a} − X_{k,a}/2 + 1/2 ]
```

where:
- `k` runs over **groups** in the molecule (NH₂, CH₂, CH₂OH, ...)
- `a` runs over **site types** on group k (e1, H, ...)
- `ν_{k,i}` = number of times group k appears in molecule i
- `n_{k,a}` = number of sites of type a on one copy of group k
- `X_{k,a}` = fraction of molecules **not bonded** at site a on group k (0 < X ≤ 1)

**Physical meaning**: when X ≈ 1 (no bonding), each term → ln(1) − 1/2 + 1/2 = 0. When X → 0 (fully bonded), ln(X) → −∞, making F_assoc strongly negative. Association always **lowers** the free energy.

### 2.2 The mass-action equation (Eq. 65)

X_{k,a} is **not** a free parameter — it is determined self-consistently from:

```
X_{k,a} = 1 / ( 1 + ρ_s  Σ_l  Σ_b  ν_{l,i} · n_{l,b} · X_{l,b} · Δ_{kl,ab} )
```

where:
- `ρ_s` = segment number density at the reference state [m⁻³]
- The sum runs over **all** (group, site) pairs in the molecule
- `Δ_{kl,ab}` = association strength between site a on group k and site b on group l

This is a set of **coupled nonlinear equations** (one per site). We solve by **successive substitution**: start with X = 1 for all sites, iterate until convergence (typically 5–15 iterations).

### 2.3 The association strength Δ_{kl,ab} (Eqs. 66–68)

```
Δ_{kl,ab} = F_{kl,ab} · K_{kl,ab} · g^HS_d(σ_{kl}; η)
```

Three factors:

| Factor | Formula | Physical meaning |
|--------|---------|------------------|
| F (Mayer-f) | exp(ε^HB_{kl,ab} / kBT) − 1 | Boltzmann weight of the H-bond energy |
| K (bonding volume) | geometric parameter [m³] | Solid-angle integral over bonding geometry |
| g^HS (RDF) | Boublík exponential form at x₀ = σ/d | Probability of finding a partner at bonding distance |

The RDF uses the Boublík exponential form (Eq. 48):

```
g^HS_d(σ) = exp( k₀ + k₁·x₀ + k₂·x₀² + k₃·x₀³ )
```

with x₀ = σ_{kl}/d_{kl} and coefficients k₀...k₃ from Eqs. 49–52. This is the same RDF used in the chain term, ensuring consistency across all Helmholtz contributions.

### 2.4 Source of ε^HB and K parameters

The association parameters (ε^HB, K) for each site pair come from three sources, tried in order:

```
┌─────────────────────────────────────────────────────────┐
│  1. Self-pair (k == l)?                                 │
│     → Use group's own self_assoc list from database     │
│                                                         │
│  2. Cross-pair with explicit database entry?            │
│     → Use fitted cross-association parameters           │
│     (try both key orderings: (k,l) and (l,k))          │
│                                                         │
│  3. Cross-pair, no database entry?                      │
│     → CR-1 combining-rule fallback:                     │
│        ε^HB_{kl} = √( ε^HB_{kk} · ε^HB_{ll} )        │
│        K_{kl}    = [ (∛K_{kk} + ∛K_{ll}) / 2 ]³       │
│     BUT: requires BOTH groups to have self_assoc ≠ []   │
│                                                         │
│  4. No association sites on either group?               │
│     → Δ_{kl,ab} = 0                                    │
└─────────────────────────────────────────────────────────┘
```

### 2.5 The CR-1 combining-rule fallback (detailed)

When no explicit cross entry exists in the database, the code attempts to **construct** cross-association parameters from each group's self-association data:

```python
def _cr1_association_fallback(k, l, groups):
    self_k = groups[k]["self_assoc"]
    self_l = groups[l]["self_assoc"]
    if not self_k or not self_l:       #  ← THE CRITICAL CHECK
        return []                      #  ← returns EMPTY → Δ = 0

    # For each self-interaction on l, check if k has a matching
    # canonical site-type pair:
    #   e.g. (e, H) on l matches (e, H) on k → combine
    for il in self_l:
        canon_l = (canonical(il.site1), canonical(il.site2))
        if canon_l found in self_k:
            ε_cross = √(ε_kk · ε_ll)
            K_cross = ((∛K_kk + ∛K_ll) / 2)³
            → add to result
    return result
```

The combining rules work by:
1. Listing all self-association interactions on group k: e.g. NH₂ has (e1, H) with ε = 1553 K, K = 1.28e-28 m³
2. Listing all self-association interactions on group l: e.g. CH₂OH has (e1, H) with ε = 2097 K, K = 6.49e-29 m³
3. For each pair with **matching canonical site types** (e↔e, H↔H), combining:
   - ε_cross = √(1553 × 2097) = 1805 K
   - K_cross = ((∛1.28e-28 + ∛6.49e-29)/2)³

---

## 3. Current problems

### 3.1 The solvation gap: N groups with no self-association

**The core issue**: CR-1 combining rules **require both groups to have self-association** (`self_assoc ≠ []`). A tertiary amine (N) has an electron-donor site (e1) but **no proton-donor** (H), so it cannot self-associate. Its `self_assoc` list is empty.

This means:

```
N + CH₂OH:   cross.get() = None
              _cr1_fallback → self_assoc(N) = [] → return []
              → Δ(N, CH₂OH) = 0          ← WRONG!
```

In reality, N's lone pair **does** accept H-bonds from CH₂OH's H-site. This "solvation" interaction (where one partner is a non-self-associating acceptor) is well-known in the SAFT literature and is typically handled by **fitted cross-parameters** in the database.

**Evidence from the Δ_kl table:**

| Pair | Δ [m³] | Source |
|------|--------|--------|
| N_2nd \| CH₂OH | 0.0 | CR-1 fails (N has no self_assoc) |
| N_2nd \| CH₂OH_Short | 1.65e-25 | **Explicit database entry** |
| N_2nd \| OH_Short | 0.0 | CR-1 fails |
| N \| CH₂OH | 6.86e-26 | **Explicit database entry** |
| N \| OH_Short | 2.08e-25 | **Explicit database entry** |
| N \| N | 0.0 | Correct — N cannot self-associate |

The pattern is clear: wherever the database has **explicit cross entries** for N with an H-donor group, we get non-zero Δ. Wherever we rely on CR-1, we get zero.

### 3.2 Impact on F_assoc

For a molecule like **MDEA** (N + 3×CH₂ + 2×CH₂OH):
- The N⋯HO-CH₂OH cross-association is **missed** if no explicit entry exists
- The mass-action solver sees fewer bonding interactions → X_{k,a} stays closer to 1 → |F_assoc| is **underestimated**
- MDEA appears "less associating" than it really is

This distorts the ranking: MDEA might rank closer to non-associating alkanes than to MEA/DEA, when physically it should be intermediate.

### 3.3 Affected group pairs

Any pair where **one group has e-sites but no self-association** is affected. In our database:

| Non-self-associating group | Sites | Has self_assoc? |
|---------------------------|-------|-----------------|
| N (tertiary amine) | e1 only | No |
| N_2nd (tertiary amine) | e1 only | No |

These groups can accept H-bonds from any H-donor (NH₂, NH, CH₂OH, CHOH, OH_Short) but the CR-1 fallback gives Δ = 0 for all such pairs unless explicit cross data exists.

### 3.4 Groups that ARE handled correctly by CR-1

| Self-associating group | Sites | self_assoc | CR-1 works? |
|----------------------|-------|------------|-------------|
| NH₂ (primary amine) | e1, H | (e1, H) → ε, K | ✓ |
| NH (secondary amine) | e1, H | (e1, H) → ε, K | ✓ |
| CH₂OH | e1, H | (e1, H) → ε, K | ✓ |
| CH₂OH_Short | e1, H | (e1, H) → ε, K | ✓ |
| CHOH | e1, H | (e1, H) → ε, K | ✓ |
| OH_Short | e1, H | (e1, H) → ε, K | ✓ |

Any pair of two self-associating groups will get a reasonable CR-1 estimate. The problem is **exclusively** with non-self-associating acceptors (N, N_2nd).

---

## 4. Possible solutions

### Option A: Do nothing (accept the limitation)

- The database has explicit cross entries for the most important N-containing pairs
- CR-1 is already an approximation; missing some pairs may be acceptable
- **Risk**: molecules with N + CH₂OH but no explicit cross entry get F_assoc = 0 for that interaction

### Option B: Solvation-aware fallback

When group k has e-sites but `self_assoc = []` (non-self-associating acceptor), and group l has self-association with an H-site:

```
ε^HB_{kl,eH} ≈ ε^HB_{ll,eH} / 2     (half the donor's self-association energy)
K_{kl}       ≈ K_{ll}                  (same bonding geometry)
```

This is the "solvation scheme" used in CPA and some SAFT variants. It captures the physical reality that N's lone pair can accept H-bonds, but with a weaker energy than a full e↔H self-association.

**Pros**: physically motivated, simple to implement, handles all N-containing pairs
**Cons**: the factor of 1/2 is empirical; real systems may deviate significantly

### Option C: Use delta_table for F_assoc

Since `build_pair_tables` already computes the group-pair Δ_{kl} correctly (using explicit DB entries), we could reformulate F_assoc to use the pre-computed delta_table rather than re-resolving site-level interactions.

**Pros**: guaranteed consistency with the Δ_{kl} values shown in the JSON output
**Cons**: loses site-level resolution (can't distinguish which specific sites bond); the mass-action equations require site-level Δ_{kl,ab}, not group-summed Δ_{kl}

### Option D: Distribute delta_table Δ over site pairs

Hybrid approach: when `_get_site_site_delta` returns 0 for a pair where `delta_table[(k,l)] > 0`, distribute the known group-level Δ evenly across compatible site pairs.

**Pros**: uses the best available data; preserves site-level mass-action structure
**Cons**: arbitrary distribution when multiple site pairs exist; adds complexity

---

## 5. Current implementation status

| Component | Status | Notes |
|-----------|--------|-------|
| Mayer-f function F_{kl,ab} | ✓ Implemented | Eq. 66 |
| Bonding volume K_{kl,ab} | ✓ From database | |
| RDF g^HS (Boublík) | ✓ Consistent with chain term | Eq. 48, x₀ = σ/d |
| Mass-action solver (Eq. 65) | ✓ Successive substitution | Converges in ~5–15 iterations |
| F_assoc evaluation (Eq. 64) | ✓ Implemented | Dimensionless A^assoc/NkBT |
| Cross-key ordering bug | ✓ Fixed | Now tries (k,l) and (l,k) |
| Self-associating cross pairs (CR-1) | ✓ Works | NH₂↔CH₂OH, NH↔OH_Short, etc. |
| **Non-self-associating acceptors** | ✗ **Gap** | **N, N_2nd with H-donor groups** |
| Solvation fallback | ✗ Not implemented | Option B above |