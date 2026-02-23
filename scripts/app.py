#!/usr/bin/env python3
"""
Simple app to compute SAFT-γ Mie distance between two molecules.

This script defines two test vectors and computes their distance.
"""

from compute_distance import compute_all_distances, load_saft_tables

def main():
    print("SAFT-γ Mie Molecule Distance Calculator")
    print("=" * 40)

    # Define test vectors (MEA vs DEA)
    base_vector = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # NCCO:
    target_vector = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0] # CC(N)CO

    print(f"Base vector (NCCO): {base_vector}")
    print(f"Target vector (CC(N)CO): {target_vector}")

    # Load SAFT tables
    try:
        tables = load_saft_tables()
        print("SAFT tables loaded successfully.")
    except Exception as e:
        print(f"Error loading SAFT tables: {e}")
        return

    # Compute distance
    try:
        result = compute_all_distances(base_vector, target_vector, tables)
        print(f"\nThermodynamic distance: {result['d_thermo']:.6f}")
        print(f"Structural distance: {result['d_struct']:.6f}")
        print(f"Euclidean distance: {result['d_euclidean']:.6f}")
        print(f"Cosine distance: {result['d_cosine']:.6f}")
    except Exception as e:
        print(f"Error computing distance: {e}")

if __name__ == "__main__":
    main()