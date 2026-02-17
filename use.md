# Quick Start Guide

## Setup

1. **Create virtual environment:**
   ```bash
   python -m venv .venv
   ```

2. **Activate virtual environment:**
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`

3. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

4. **Generate pair tables (run once):**
   ```bash
   python scripts/saft_similarity.py
   ```
   This creates `saft_pair_tables.json` with pre-computed SAFT-γ Mie pair data.

5. **Compute distance between molecules:**
   ```bash
   python scripts/app.py
   ```
   This runs a simple test comparing MEA vs DMEA molecules.

## Vector Format

Molecule vectors are 20-element lists representing group counts in this order:
- [NH2_2nd, NH_2nd, N_2nd, CH3, CH2, CH, C, CH2OH, CH2OH_Short, NH2, NH, N, OH, OH_Short, cCH2, cCH, cNH, cN, cCHNH, cCHN]

Example vectors:
- MEA (monoethanolamine): `[1,0,0,0,1,0,0,0,1,0,0,0,0,0]`
- DMEA (diethanolamine): `[1,0,0,0,2,0,0,0,1,0,0,0,0,0]`

## API Usage

```python
from scripts.compute_distance import distance

# Compute distance between two vectors
base = [1,0,0,0,1,0,0,0,1,0,0,0,0,0]  # MEA
target = [1,0,0,0,2,0,0,0,1,0,0,0,0,0]  # DMEA
dist = distance(base, target)
print(f"Distance: {dist:.6f}")
```