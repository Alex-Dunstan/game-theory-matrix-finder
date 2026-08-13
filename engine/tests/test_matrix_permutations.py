"""
Tests for matrix_permutations.py

These cover:
  - generate_dual_matrices: count, shape, value ranges for several sizes
  - check_nash_equilibrium / find_nash_equilibria: known results for
    classic game theory matrices (Prisoner's Dilemma, Coordination, etc.)
  - Backwards-compatibility alias generate_2x2_dual_matrices
"""

import numpy as np
import pytest

from src.matrix_permutations import (
    generate_dual_matrices,
    generate_2x2_dual_matrices,
    check_nash_equilibrium,
    find_nash_equilibria,
    print_dual_matrices,
)


# ---------------------------------------------------------------------------
# generate_dual_matrices
# ---------------------------------------------------------------------------

class TestGenerateDualMatrices:

    def test_2x2_range_0_1_count(self):
        # values = {0, 1}  →  pairs per cell = 2×2 = 4  →  4^4 = 256
        matrices = generate_dual_matrices(2, 2, 0, 1)
        assert len(matrices) == 256

    def test_2x2_range_0_5_count(self):
        # values = {0..5}  →  pairs per cell = 36  →  36^4 = 1,679,616
        matrices = generate_dual_matrices(2, 2, 0, 5)
        assert len(matrices) == 1_679_616

    def test_2x2_shape(self):
        matrices = generate_dual_matrices(2, 2, 0, 1)
        for m in matrices:
            assert m.shape == (2, 2, 2)

    def test_2x3_count_and_shape(self):
        # 2×3 with range 0-1: 4^6 = 4,096 matrices; shape (2, 3, 2)
        matrices = generate_dual_matrices(2, 3, 0, 1)
        assert len(matrices) == 4 ** 6
        for m in matrices:
            assert m.shape == (2, 3, 2)

    def test_values_within_range(self):
        matrices = generate_dual_matrices(2, 2, 0, 3)
        for m in matrices:
            assert np.all(m >= 0)
            assert np.all(m <= 3)

    def test_backwards_compat_alias(self):
        a = generate_2x2_dual_matrices(0, 2)
        b = generate_dual_matrices(2, 2, 0, 2)
        assert len(a) == len(b)
        assert a[0].shape == b[0].shape


# ---------------------------------------------------------------------------
# Nash equilibrium detection — hand-crafted known matrices
# ---------------------------------------------------------------------------

def make_matrix(tl1, tl2, tr1, tr2, bl1, bl2, br1, br2) -> np.ndarray:
    """Helper: build a 2×2 dual matrix from 8 scalars (row-major, p1 then p2)."""
    return np.array([[[tl1, tl2], [tr1, tr2]],
                     [[bl1, bl2], [br1, br2]]], dtype=np.int32)


class TestNashEquilibrium:

    def test_prisoners_dilemma(self):
        """
        Classic Prisoner's Dilemma (Defect dominates for both players).
        Payoffs (row=Cooperate/Defect, col=Cooperate/Defect):
            (3,3)  (0,5)
            (5,0)  (1,1)   ← sole NE at (1,1)
        """
        m = make_matrix(3, 3,  0, 5,
                        5, 0,  1, 1)
        eq = find_nash_equilibria(m)
        assert eq == [(1, 1)], f"Expected [(1,1)] but got {eq}"

    def test_coordination_game(self):
        """
        Pure coordination: two NE on the diagonal.
            (2,2)  (0,0)
            (0,0)  (2,2)
        """
        m = make_matrix(2, 2,  0, 0,
                        0, 0,  2, 2)
        eq = find_nash_equilibria(m)
        assert set(eq) == {(0, 0), (1, 1)}

    def test_dominant_strategy_top_left(self):
        """
        Both players strictly prefer row=0 / col=0 regardless of the other.
            (5,5)  (5,2)
            (2,5)  (2,2)
        """
        m = make_matrix(5, 5,  5, 2,
                        2, 5,  2, 2)
        eq = find_nash_equilibria(m)
        assert (0, 0) in eq

    def test_no_pure_nash(self):
        """
        Matching pennies has no pure-strategy NE.
            (1,-1)  (-1,1)
            (-1,1)  (1,-1)
        """
        m = make_matrix( 1, -1,  -1,  1,
                        -1,  1,   1, -1)
        eq = find_nash_equilibria(m)
        assert eq == [], f"Expected no NE but got {eq}"

    def test_four_nash_equilibria(self):
        """
        When all cells are identical payoffs, every cell is a NE.
            (3,3)  (3,3)
            (3,3)  (3,3)
        """
        m = make_matrix(3, 3,  3, 3,
                        3, 3,  3, 3)
        eq = find_nash_equilibria(m)
        assert len(eq) == 4

    def test_single_cell_check(self):
        """check_nash_equilibrium on individual cells."""
        m = make_matrix(3, 3,  0, 5,
                        5, 0,  1, 1)
        assert check_nash_equilibrium(m, (1, 1)) is True
        assert check_nash_equilibrium(m, (0, 0)) is False


# ---------------------------------------------------------------------------
# Display utility (smoke test — just confirm it doesn't raise)
# ---------------------------------------------------------------------------

def test_print_dual_matrices_runs(capsys):
    matrices = generate_dual_matrices(2, 2, 0, 1)
    print_dual_matrices(matrices[:2])   # print just two to keep output short
    captured = capsys.readouterr()
    assert "Matrix 1:" in captured.out
    assert "Matrix 2:" in captured.out
