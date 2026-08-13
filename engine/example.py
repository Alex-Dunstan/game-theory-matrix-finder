"""
example.py — quick demo of the game theory matrix library.

Shows how to:
  1. Generate all 2×2 matrices for a given range
  2. Inspect Nash equilibria on individual matrices
  3. Use the n×m parameterisation for non-square grids
  4. Save results to CSV

For large-scale dataset generation (multiple ranges, streaming to disk),
use generate_datasets.py instead.
"""

from src.matrix_permutations import (
    generate_dual_matrices,
    generate_2x2_dual_matrices,   # backwards-compat alias
    find_nash_equilibria,
    save_dual_matrices_to_csv,
    print_dual_matrices,
)

# ── 1. Generate all 2×2 matrices for range 0–5 ──────────────────────────────
print("Generating all 2×2 matrices (range 0–5)…")
matrices = generate_dual_matrices(rows=2, cols=2, min_val=0, max_val=5)
print(f"  Total matrices: {len(matrices):,}")   # 1,679,616

# ── 2. Inspect the first 3 matrices ─────────────────────────────────────────
print("\nAnalysing first 3 matrices:")
for i, matrix in enumerate(matrices[:3]):
    print(f"\nMatrix {i + 1}:")
    for row in matrix:
        print([f"({pair[0]},{pair[1]})" for pair in row])
    eq = find_nash_equilibria(matrix)
    if eq:
        print(f"  Nash equilibria at: {eq}")
    else:
        print("  No pure-strategy Nash equilibrium")

# ── 3. Classic game: Prisoner's Dilemma ─────────────────────────────────────
import numpy as np
pd_matrix = np.array([
    [[3, 3], [0, 5]],   # Row 0: Cooperate vs Cooperate, Cooperate vs Defect
    [[5, 0], [1, 1]],   # Row 1: Defect vs Cooperate,    Defect vs Defect
], dtype=np.int32)

eq = find_nash_equilibria(pd_matrix)
print(f"\nPrisoner's Dilemma NE: {eq}")   # → [(1, 1)]

# ── 4. n×m example: 2×3 matrix with range 0–3 ───────────────────────────────
print("\nGenerating all 2×3 matrices (range 0–3)…")
mats_2x3 = generate_dual_matrices(rows=2, cols=3, min_val=0, max_val=3)
print(f"  Total: {len(mats_2x3):,}")   # 16^6 = 16,777,216

# ── 5. Save a small subset to CSV ───────────────────────────────────────────
sample = matrices[:1000]
save_dual_matrices_to_csv(sample, "sample_matrices.csv")
print("\nSaved 1,000 sample matrices to sample_matrices.csv")
print("\nDone. Run  python app.py  to open the web viewer.")
