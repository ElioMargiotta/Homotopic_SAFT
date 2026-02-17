#!/usr/bin/env python3
"""
Simple app to compute SAFT-γ Mie distance between two molecules.

This script defines two test vectors and computes their distance.
"""

from compute_distance import distance

def main():
    print("SAFT-γ Mie Molecule Distance Calculator")
    print("=" * 40)

    # Define test vectors (MEA vs DMEA)
    base_vector = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # MEA: CH2 + NH2
    target_vector = [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  # DMEA: 2×CH3 + NH

    print(f"Base vector (MEA): {base_vector}")
    print(f"Target vector (DMEA): {target_vector}")

    # Compute distance
    try:
        dist = distance(base_vector, target_vector)
        print(f"\nDistance: {dist:.6f}")
    except Exception as e:
        print(f"Error computing distance: {e}")

if __name__ == "__main__":
    main()