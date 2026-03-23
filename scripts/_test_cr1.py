"""
Compare CR-1 combining-rule association estimates vs fitted database values.

For every cross pair that HAS an explicit entry in the XML, compute:
  1. Delta_db   = association using the fitted database parameters
  2. Delta_cr1  = association using only self-pair params + CR-1 combining rules

This tells us how accurate the CR-1 fallback is relative to the real fits.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from saft_similarity import (
    load_database, build_pair_tables, delta_pair, get_pair_params,
    _cr1_association_fallback, delta_site_pair,
    T_REF, ETA_REF,
)

xml = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "database", "database.xml"))
groups, cross = load_database(xml)

# All groups that have association sites
assoc_groups = [g for g, d in groups.items() if d["sites"]]
print(f"Groups with association sites: {assoc_groups}\n")

# For every cross pair that EXISTS in the database and involves two
# associating groups, compare database Delta vs CR-1 Delta.
print(f"{'Pair':<35s}  {'Delta_db':>14s}  {'Delta_cr1':>14s}  {'Ratio':>8s}  {'Diff%':>8s}")
print("-" * 90)

pairs_tested = 0
for (g1, g2), ci in cross.items():
    # Only process each unordered pair once (g1 < g2 alphabetically)
    if g1 >= g2:
        continue
    # Both must have association sites
    if not groups.get(g1, {}).get("sites") or not groups.get(g2, {}).get("sites"):
        continue
    # Must have association data in the cross entry
    if not ci["association"]:
        continue

    # 1. Delta from DATABASE (real fitted cross params)
    eps, sig, lr, la = get_pair_params(g1, g2, groups, cross)
    delta_db = delta_pair(g1, g2, groups, cross, sig, T_REF, ETA_REF)

    # 2. Delta from CR-1 only (pretend no cross entry exists)
    #    Use the same sigma (from combining rules)
    from saft_similarity import combining_sigma
    sig_cr1 = combining_sigma(groups[g1]["sigma"], groups[g2]["sigma"])
    cr1_list = _cr1_association_fallback(g1, g2, groups)
    sites_k = groups[g1]["sites"]
    sites_l = groups[g2]["sites"]
    delta_cr1 = 0.0
    for inter in cr1_list:
        s1 = inter["site1"]
        s2 = inter["site2"]
        ea = inter["epsilonAssoc"]
        bv = inter["bondingVolume"]
        m1 = sites_k.get(s1, 0.0)
        m2 = sites_l.get(s2, 0.0)
        delta_cr1 += m1 * m2 * delta_site_pair(ea, bv, sig_cr1, T_REF, ETA_REF)

    if delta_db == 0.0 and delta_cr1 == 0.0:
        continue

    pairs_tested += 1
    ratio = delta_cr1 / delta_db if delta_db != 0.0 else float("inf")
    diff_pct = (delta_cr1 - delta_db) / delta_db * 100 if delta_db != 0 else float("inf")

    flag = ""
    if abs(diff_pct) > 50:
        flag = " <<<< LARGE"
    elif abs(diff_pct) > 20:
        flag = " << notable"

    print(f"  {g1:>14s} <-> {g2:<14s}  {delta_db:14.6e}  {delta_cr1:14.6e}  "
          f"{ratio:8.4f}  {diff_pct:+7.1f}%{flag}")

print(f"\nTotal cross pairs compared: {pairs_tested}")

# --- Also show the key fallback pairs (no database entry) ---
print("\n" + "=" * 70)
print("Pairs with NO database entry (CR-1 fallback only):")
print("=" * 70)
no_entry_pairs = [
    ("CH2OH", "CH2OH_Short"),
    ("OH", "OH_Short"),
    ("NH2", "NH2_2nd"),
    ("NH2", "NH_2nd"),
    ("NH2_2nd", "NH"),
]
for g1, g2 in no_entry_pairs:
    if g1 not in groups or g2 not in groups:
        continue
    has_cross = (g1, g2) in cross or (g2, g1) in cross
    eps, sig, lr, la = get_pair_params(g1, g2, groups, cross)
    d = delta_pair(g1, g2, groups, cross, sig, T_REF, ETA_REF)
    d_self1 = delta_pair(g1, g1, groups, cross, groups[g1]["sigma"], T_REF, ETA_REF)
    d_self2 = delta_pair(g2, g2, groups, cross, groups[g2]["sigma"], T_REF, ETA_REF)
    import math
    geo = math.sqrt(d_self1 * d_self2) if d_self1 > 0 and d_self2 > 0 else 0
    print(f"  {g1:>14s} <-> {g2:<14s}  cross_in_db={has_cross!s:<6s}"
          f"  Delta_cr1={d:14.6e}  Delta_kk={d_self1:.6e}  Delta_ll={d_self2:.6e}"
          f"  geo_mean={geo:.6e}")
