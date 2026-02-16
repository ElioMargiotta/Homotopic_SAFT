import xml.etree.ElementTree as ET
import numpy as np
import math

R_GAS = 8.314462618
T_REF = 298.15  # K

# ====== 1. Load database ======

def _parse_float(text, default=0.0):
    if text is None:
        return default
    try:
        return float(text)
    except Exception:
        return default

def load_gsaft_database(xml_path):
    """
    Parse database.xml into:
    - groups: dict[name] -> dict of self / thermo / association data
    - cross: dict[(g1, g2)] -> dict of cross dispersion / association data
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # ---- groups ----
    groups = {}
    groups_el = root.find("groups")
    for g in groups_el.findall("group"):
        name = g.attrib["name"]
        data = {}

        # Heat capacity at T_REF: Cp(T) = a + bT + cT^2 + dT^3
        hc = g.find("heatCapacity")
        a = _parse_float(hc.findtext("a"), 0.0)
        b = _parse_float(hc.findtext("b"), 0.0)
        c = _parse_float(hc.findtext("c"), 0.0)
        d = _parse_float(hc.findtext("d"), 0.0)
        Cp_star = a + b*T_REF + c*T_REF**2 + d*T_REF**3

        Hf = _parse_float(g.findtext("enthalpyOfFormation"), 0.0)
        Sf = _parse_float(g.findtext("entropyOfFormation"), 0.0)

        data["Cp_star"] = Cp_star
        data["Hf"] = Hf
        data["Sf"] = Sf
        data["Hhat"] = Hf / (R_GAS*T_REF) if Hf is not None else 0.0
        data["Shat"] = Sf / R_GAS if Sf is not None else 0.0

        data["m"] = _parse_float(g.findtext("numberOfSegments"), 0.0)
        data["shapeFactor"] = _parse_float(g.findtext("shapeFactor"), 0.0)

        # self-interaction: dispersion + association
        self_inter = g.find("selfInteraction")
        eps = sigma = lrep = latt = 0.0
        self_assoc = []
        if self_inter is not None:
            disp = self_inter.find("dispersion")
            if disp is not None:
                eps = _parse_float(disp.findtext("epsilon"), 0.0)
                sigma = _parse_float(disp.findtext("sigma"), 0.0)
                lrep = _parse_float(disp.findtext("lambdaRepulsive"), 0.0)
                latt = _parse_float(disp.findtext("lambdaAttractive"), 0.0)
            assoc_block = self_inter.find("association")
            if assoc_block is not None:
                for inter in assoc_block.findall("interaction"):
                    ea = _parse_float(inter.findtext("epsilonAssoc"), 0.0)
                    bv = _parse_float(inter.findtext("bondingVolume"), 0.0)
                    self_assoc.append((ea, bv))

        data["eps"] = eps
        data["sigma"] = sigma
        data["lambdaRepulsive"] = lrep
        data["lambdaAttractive"] = latt

        # association sites: H / e* / a*
        nH = ne = na = 0.0
        assoc_sites = g.find("associationSites")
        if assoc_sites is not None:
            for site in assoc_sites.findall("siteMultiplicity"):
                sname = site.attrib.get("name", "")
                mult = _parse_float(site.text, 0.0)
                if sname.startswith("H"):
                    nH += mult
                if sname.startswith("e"):
                    ne += mult
                if sname.startswith("a"):
                    na += mult
        data["nH"] = nH
        data["ne"] = ne
        data["na"] = na

        # self association totals (sum over all interactions)
        e_assoc_tot = sum(e for e, _ in self_assoc)
        v_assoc_tot = sum(v for _, v in self_assoc)
        data["e_assoc_tot"] = e_assoc_tot
        data["v_assoc_tot"] = v_assoc_tot

        # logs (only for self epsilon, sigma, association)
        data["log_eps"] = math.log(eps) if eps > 0 else 0.0
        data["log_sigma"] = math.log(sigma) if sigma > 0 else 0.0
        data["log_e_assoc"] = math.log(1 + e_assoc_tot) if e_assoc_tot > 0 else 0.0
        data["log_v_assoc"] = math.log(1 + v_assoc_tot) if v_assoc_tot > 0 else 0.0

        groups[name] = data

    # ---- cross-interactions ----
    cross = {}
    cross_el = root.find("crossInteractions")
    for ci in cross_el.findall("crossInteraction"):
        g1 = ci.attrib["group1"]
        g2 = ci.attrib["group2"]

        disp = ci.find("dispersion")
        eps = _parse_float(disp.findtext("epsilon"), 0.0) if disp is not None else 0.0
        lrep = _parse_float(disp.findtext("lambdaRepulsive"), 0.0) if disp is not None else 0.0
        latt = _parse_float(disp.findtext("lambdaAttractive"), 0.0) if disp is not None else 0.0

        assoc_block = ci.find("association")
        e_assoc_sum = 0.0
        v_assoc_sum = 0.0
        if assoc_block is not None:
            for inter in assoc_block.findall("interaction"):
                ea = _parse_float(inter.findtext("epsilonAssoc"), 0.0)
                bv = _parse_float(inter.findtext("bondingVolume"), 0.0)
                e_assoc_sum += ea
                v_assoc_sum += bv

        info = {
            "eps": eps,
            "lambdaRepulsive": lrep,
            "lambdaAttractive": latt,
            "e_assoc_sum": e_assoc_sum,
            "v_assoc_sum": v_assoc_sum,
        }
        cross[(g1, g2)] = info
        # symmetric
        if (g2, g1) not in cross:
            cross[(g2, g1)] = info

    return groups, cross

# ====== 2. Feature vectors φ_self and φ_cross ======

def phi_self(name, groups):
    """
    Intrinsic group features at T_REF.
    """
    d = groups[name]
    return np.array([
        d["m"],
        d["shapeFactor"],
        d["log_eps"],
        d["log_sigma"],
        d["lambdaRepulsive"],
        d["lambdaAttractive"],
        d["nH"],
        d["ne"],
        d["na"],
        d["log_e_assoc"],
        d["log_v_assoc"],
    ], dtype=float)

def cross_features(g, r, cross):
    """
    4-dimensional fingerprint of how group g interacts with reference r:
    [epsilon_disp, lambdaRepulsive, log(1+sum epsilonAssoc), log(1+sum bondingVolume)].
    """
    info = cross.get((g, r))
    if info is None:
        eps = 0.0
        lrep = 0.0
        e_assoc = 0.0
        v_assoc = 0.0
    else:
        eps = info["eps"]
        lrep = info["lambdaRepulsive"]
        e_assoc = info["e_assoc_sum"]
        v_assoc = info["v_assoc_sum"]
    logE = math.log(1 + e_assoc) if e_assoc > 0 else 0.0
    logV = math.log(1 + v_assoc) if v_assoc > 0 else 0.0
    return np.array([eps, lrep, logE, logV], dtype=float)

def phi_cross(name, ref_groups, cross):
    """
    Concatenate cross-interaction fingerprints vs each reference group r in ref_groups.
    """
    return np.concatenate([cross_features(name, r, cross) for r in ref_groups])

# ====== 3. Build S for a chosen set of groups ======

def _standardize(mat):
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    std_safe = np.where(std > 0, std, 1.0)
    return (mat - mean) / std_safe, mean, std_safe

def build_similarity_matrix_from_db(
    xml_path,
    group_names,
    ref_groups,
    alpha=0.7,
    beta=0.3,
):
    """
    Build a similarity matrix S for group_names using database features.
    S[i,j] represents the intrinsic similarity between groups i and j.
    This matrix is then used in soft cosine distance calculations.

    Returns:
        S (numpy.ndarray), ordered_group_names (list[str])
    """
    groups, cross = load_gsaft_database(xml_path)

    # feature matrices
    phi_self_mat = np.vstack([phi_self(g, groups) for g in group_names])
    phi_cross_mat = np.vstack([phi_cross(g, ref_groups, cross) for g in group_names])

    z_self, _, _ = _standardize(phi_self_mat)
    z_cross, _, _ = _standardize(phi_cross_mat)

    phi_comb = np.hstack([alpha * z_self, beta * z_cross])

    # Build similarity matrix using dot product of normalized features
    # This gives similarity in [0, 1] range
    norms = np.linalg.norm(phi_comb, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    phi_norm = phi_comb / norms
    
    # Compute pairwise similarities: S[i,j] = phi_norm[i] · phi_norm[j]
    # Rescale from [-1, 1] to [0, 1]
    S = (phi_norm @ phi_norm.T + 1.0) / 2.0
    
    # Ensure diagonal is exactly 1.0 and symmetry
    np.fill_diagonal(S, 1.0)
    S = (S + S.T) / 2.0  # Ensure perfect symmetry

    return S, group_names

# ====== 4. Example: your 20 groups ======

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

REF_GROUPS = GROUPS_OF_INTEREST

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    xml_path = "database/database.xml"  # or full path
    S, order = build_similarity_matrix_from_db(
        xml_path,
        GROUPS_OF_INTEREST,
        REF_GROUPS,
        alpha=0.7,
        beta=0.3,
    )

    # quick sanity checks
    idx = {name: i for i, name in enumerate(order)}
    def sim(g1, g2):
        return float(S[idx[g1], idx[g2]])

    print("Similarity CH2OH vs CH2OH_Short:", sim("CH2OH", "CH2OH_Short"))
    print("Similarity OH vs OH_Short:",     sim("OH", "OH_Short"))
    print("Similarity NH2 vs NH2_2nd:",     sim("NH2", "NH2_2nd"))
    print("Similarity CH2 vs cCH2:",        sim("CH2", "cCH2"))
    print("Similarity CH vs cCH:",          sim("CH", "cCH"))
    
    # Plot similarity matrix
    plt.figure(figsize=(12, 10))
    im = plt.imshow(S, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, label='Similarity')
    plt.title('Group Similarity Matrix S', fontsize=14, fontweight='bold')
    plt.xlabel('Group Index', fontsize=12)
    plt.ylabel('Group Index', fontsize=12)
    
    # Add tick labels with group names
    plt.xticks(range(len(order)), order, rotation=90, fontsize=8)
    plt.yticks(range(len(order)), order, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('similarity_matrix_plot.png', dpi=150)
    print(f"\nSimilarity matrix plot saved to similarity_matrix_plot.png")
    plt.show()
    
    # Save S matrix for use in chem_soft_similarity.py
    # Need to expand 20x20 to 40x40 (duplicate for two components)
    S_40x40 = np.zeros((40, 40))
    S_40x40[:20, :20] = S
    S_40x40[20:40, 20:40] = S
    S_40x40[:20, 20:40] = S
    S_40x40[20:40, :20] = S
    
    np.save('similarity_matrix.npy', S_40x40)
    print(f"Saved 40x40 similarity matrix to similarity_matrix.npy")
