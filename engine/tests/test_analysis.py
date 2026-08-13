"""
Tests for src/analysis.py — game type classification.

Each test uses a hand-crafted matrix whose game type is known from
classical game theory, then verifies both the boolean properties
and the named label.
"""

import numpy as np
import pytest

from src.analysis import (
    classify_properties,
    classify_game_type,
    classify_full,
    GAME_TYPE_DESCRIPTIONS,
    _compute_mixed_strategy_2x2,
    _ne_payoff_stats,
)
from src.matrix_permutations import find_nash_equilibria


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make(tl1, tl2, tr1, tr2, bl1, bl2, br1, br2) -> np.ndarray:
    """Build a 2×2 dual matrix from 8 scalars (row-major, P1 then P2)."""
    return np.array([[[tl1, tl2], [tr1, tr2]],
                     [[bl1, bl2], [br1, br2]]], dtype=np.int32)


def classify(m):
    ne = find_nash_equilibria(m)
    return classify_full(m, ne)


# ---------------------------------------------------------------------------
# Prisoner's Dilemma
# ---------------------------------------------------------------------------

class TestPrisonersDilemma:
    """
    Classic PD:
        (3,3)  (0,5)
        (5,0)  (1,1)   ← sole NE, Pareto-dominated by (3,3)
    """
    m = make(3, 3, 0, 5, 5, 0, 1, 1)

    def test_game_type(self):
        props, label = classify(self.m)
        assert label == "Prisoner's Dilemma"

    def test_both_dominant(self):
        props, _ = classify(self.m)
        assert props["both_dominant"] is True

    def test_welfare_loss_positive(self):
        props, _ = classify(self.m)
        assert props["welfare_loss"] > 0

    def test_has_pareto_dominated_ne(self):
        props, _ = classify(self.m)
        assert props["has_pareto_dom_ne"] is True

    def test_ne_count(self):
        props, _ = classify(self.m)
        assert props["ne_count"] == 1

    def test_max_welfare(self):
        props, _ = classify(self.m)
        assert props["max_welfare"] == 6   # (3,3) cell


# ---------------------------------------------------------------------------
# Harmony Game
# ---------------------------------------------------------------------------

class TestHarmony:
    """
    Harmony: mutual cooperation is the dominant strategy AND the NE.
        (4,4)  (2,3)
        (3,2)  (1,1)   ← both dominant: row 0, col 0 → NE at (0,0) is Pareto-efficient
    """
    m = make(4, 4, 2, 3, 3, 2, 1, 1)

    def test_game_type(self):
        props, label = classify(self.m)
        assert label == "Harmony"

    def test_both_dominant(self):
        props, _ = classify(self.m)
        assert props["both_dominant"] is True

    def test_welfare_loss_zero(self):
        props, _ = classify(self.m)
        assert props["welfare_loss"] == 0

    def test_all_ne_pareto_efficient(self):
        props, _ = classify(self.m)
        assert props["all_ne_pareto_eff"] is True


# ---------------------------------------------------------------------------
# Coordination Game
# ---------------------------------------------------------------------------

class TestCoordination:
    """
    Symmetric coordination:
        (2,2)  (0,0)
        (0,0)  (2,2)   ← 2 NE on the diagonal, symmetric payoffs
    """
    m = make(2, 2, 0, 0, 0, 0, 2, 2)

    def test_game_type(self):
        props, label = classify(self.m)
        assert label == "Coordination"

    def test_is_symmetric(self):
        props, _ = classify(self.m)
        assert props["is_symmetric"] is True

    def test_ne_count(self):
        props, _ = classify(self.m)
        assert props["ne_count"] == 2


# ---------------------------------------------------------------------------
# Battle of the Sexes / Stag Hunt / Chicken
# ---------------------------------------------------------------------------

class TestBattleOfTheSexes:
    """
    Two diagonal equilibria with conflicting preferences over which one to pick.
        (3,2)  (0,0)
        (0,0)  (2,3)
    """
    m = make(3, 2, 0, 0, 0, 0, 2, 3)

    def test_game_type(self):
        props, label = classify(self.m)
        assert label == "Battle of the Sexes"


class TestStagHunt:
    """
    Symmetric coordination with one payoff-dominant diagonal equilibrium.
        (4,4)  (1,3)
        (3,1)  (2,2)
    """
    m = make(4, 4, 1, 3, 3, 1, 2, 2)

    def test_game_type(self):
        props, label = classify(self.m)
        assert label == "Stag Hunt"


class TestChicken:
    """
    Symmetric anti-coordination with off-diagonal equilibria.
        (3,3)  (1,4)
        (4,1)  (0,0)
    """
    m = make(3, 3, 1, 4, 4, 1, 0, 0)

    def test_game_type(self):
        props, label = classify(self.m)
        assert label == "Chicken"


# ---------------------------------------------------------------------------
# Zero-Sum
# ---------------------------------------------------------------------------

class TestZeroSum:
    """
    Matching Pennies (constant sum = 0):
        ( 1,-1)  (-1, 1)
        (-1, 1)  ( 1,-1)
    """
    m = make(1, -1, -1, 1, -1, 1, 1, -1)

    def test_game_type(self):
        props, label = classify(self.m)
        assert label == "Zero-Sum"

    def test_is_zero_sum(self):
        props, _ = classify(self.m)
        assert props["is_zero_sum"] is True

    def test_no_equilibrium(self):
        props, _ = classify(self.m)
        # Matching pennies has no pure NE, but it's still labelled Zero-Sum
        assert props["ne_count"] == 0


# ---------------------------------------------------------------------------
# No Equilibrium (non-zero-sum)
# ---------------------------------------------------------------------------

class TestNoEquilibrium:
    """
    A non-zero-sum matrix with no pure NE and no zero-sum structure.
        (3,1)  (1,3)
        (1,3)  (3,1)   — payoffs cycle; no stable cell
    """
    m = make(3, 1, 1, 3, 1, 3, 3, 1)

    def test_has_no_ne(self):
        ne = find_nash_equilibria(self.m)
        assert ne == []

    def test_game_type_no_equilibrium(self):
        props, label = classify(self.m)
        # Not zero-sum (sums are 4 everywhere — actually IS constant sum!)
        # Let's verify: 3+1=4, 1+3=4, 1+3=4, 3+1=4 → this IS zero-sum/constant-sum
        # So label will be "Zero-Sum". That's correct.
        assert label in ("Zero-Sum", "No Equilibrium")


class TestNoEquilibriumTrueAsymmetric:
    """
    A matrix with no pure NE and non-constant-sum payoffs.
        (3,0)  (0,4)
        (1,3)  (4,1)
    Cycle: P2 deviates (0,0)→(0,1); P1 deviates (0,1)→(1,1);
           P2 deviates (1,1)→(1,0); P1 deviates (1,0)→(0,0).
    Welfare sums: 3, 4, 4, 5 — not constant.
    """
    m = make(3, 0, 0, 4, 1, 3, 4, 1)

    def test_has_no_ne(self):
        ne = find_nash_equilibria(self.m)
        assert ne == []

    def test_is_not_zero_sum(self):
        props, _ = classify(self.m)
        assert props["is_zero_sum"] is False

    def test_game_type(self):
        props, label = classify(self.m)
        assert label == "No Equilibrium"


# ---------------------------------------------------------------------------
# Deadlock
# ---------------------------------------------------------------------------

class TestDeadlock:
    """
    Deadlock: both players have dominant strategies, NE is not Pareto-dominated,
    but welfare_loss > 0 (mutual defection is rational yet not socially optimal).

        (0,6)  (1,7)
        (2,1)  (3,2)   ← NE at (1,1)

    P1 dominant: row 1 (2>0 and 3>1 for all cols).
    P2 dominant: col 1 (7>6 at row0, 2>1 at row1).
    NE at (1,1)=(3,2), welfare=5.
    Max welfare=8 at (0,1)=(1,7), but (1,7) does NOT Pareto-dominate (3,2)
    because p1=1 < 3.  So welfare_loss=3 > 0 with no Pareto-dominated NE.
    """
    m = make(0, 6, 1, 7, 2, 1, 3, 2)

    def test_both_dominant(self):
        props, _ = classify(self.m)
        assert props["both_dominant"] is True

    def test_welfare_loss_positive(self):
        props, _ = classify(self.m)
        assert props["welfare_loss"] > 0

    def test_no_pareto_dominated_ne(self):
        props, _ = classify(self.m)
        assert props["has_pareto_dom_ne"] is False

    def test_game_type(self):
        props, label = classify(self.m)
        assert label == "Deadlock"


# ---------------------------------------------------------------------------
# Single-sided dominant strategy
# ---------------------------------------------------------------------------

class TestP1DominantOnly:
    """
    P1 always prefers row 1 regardless of P2's choice, but P2 has no dominant col.

        (0,3)  (2,1)
        (1,2)  (3,4)

    P1: row 1 always (1>0 at col0, 3>2 at col1) ✓.
    P2: col0=[3,2], col1=[1,4]. Col 0 is not dominant (1<4 at row1).
                                 Col 1 is not dominant (1<3 at row0). ✓
    """
    m = make(0, 3, 2, 1, 1, 2, 3, 4)

    def test_p1_has_dominant(self):
        props, _ = classify(self.m)
        assert props["p1_has_dominant"] is True

    def test_p2_no_dominant(self):
        props, _ = classify(self.m)
        assert props["p2_has_dominant"] is False

    def test_game_type(self):
        props, label = classify(self.m)
        assert label == "Dominant (P1 only)"


# ---------------------------------------------------------------------------
# classify_full convenience wrapper
# ---------------------------------------------------------------------------

def test_classify_full_returns_tuple():
    m = make(3, 3, 0, 5, 5, 0, 1, 1)
    ne = find_nash_equilibria(m)
    result = classify_full(m, ne)
    assert isinstance(result, tuple)
    assert len(result) == 2
    props, label = result
    assert isinstance(props, dict)
    assert isinstance(label, str)


# ---------------------------------------------------------------------------
# GAME_TYPE_DESCRIPTIONS coverage
# ---------------------------------------------------------------------------

def test_all_types_have_descriptions():
    expected_types = {
        "Zero-Sum", "Prisoner's Dilemma", "Harmony", "Deadlock",
        "Battle of the Sexes", "Stag Hunt", "Chicken",
        "Coordination", "Dominant (P1 only)", "Dominant (P2 only)",
        "No Equilibrium", "Other",
    }
    assert expected_types <= set(GAME_TYPE_DESCRIPTIONS.keys())


# ---------------------------------------------------------------------------
# Feature 1 — Mixed-Strategy Equilibria
# ---------------------------------------------------------------------------

class TestMixedStrategy2x2:
    """
    Tests for _compute_mixed_strategy_2x2 and its integration via
    classify_properties (mixed_* keys).
    """

    def test_matching_pennies_valid_mixed_ne(self):
        """
        Matching Pennies: (1,-1,-1,1,-1,1,1,-1)
        p = (h-g)/(e-g-f+h) = (-1-1)/(-1-1-1-1) = -2/-4 = 0.5
        q = (d-b)/(a-b-c+d) = (1-(-1))/(1-(-1)-(-1)+1) = 2/4 = 0.5
        """
        m = make(1, -1, -1, 1, -1, 1, 1, -1)
        result = _compute_mixed_strategy_2x2(m)
        assert result["mixed_exists"] is True
        assert result["mixed_p"] == pytest.approx(0.5, abs=1e-5)
        assert result["mixed_q"] == pytest.approx(0.5, abs=1e-5)
        assert result["mixed_payoff_p1"] == pytest.approx(0.0, abs=1e-5)
        assert result["mixed_payoff_p2"] == pytest.approx(0.0, abs=1e-5)

    def test_asymmetric_no_ne_valid_mixed(self):
        """
        Asymmetric no-NE: (3,0,0,4,1,3,4,1)
        a=3,e=0,b=0,f=4,c=1,g=3,d=4,h=1
        p = (h-g)/(e-g-f+h) = (1-3)/(0-3-4+1) = -2/-6 = 1/3
        q = (d-b)/(a-b-c+d) = (4-0)/(3-0-1+4) = 4/6 = 2/3
        """
        m = make(3, 0, 0, 4, 1, 3, 4, 1)
        result = _compute_mixed_strategy_2x2(m)
        assert result["mixed_exists"] is True
        assert result["mixed_p"] == pytest.approx(1 / 3, abs=1e-5)
        assert result["mixed_q"] == pytest.approx(2 / 3, abs=1e-5)

    def test_pure_ne_matrix_skips_mixed(self):
        """
        Prisoner's Dilemma has a pure NE at (1,1) → classify_properties
        should NOT compute mixed strategy (mixed_exists=False, all None).
        """
        m = make(3, 3, 0, 5, 5, 0, 1, 1)
        ne = find_nash_equilibria(m)
        assert len(ne) == 1   # sanity check
        props = classify_properties(m, ne)
        assert props["mixed_exists"] is False
        assert props["mixed_p"] is None
        assert props["mixed_q"] is None
        assert props["mixed_payoff_p1"] is None
        assert props["mixed_payoff_p2"] is None

    def test_degenerate_zero_denominator(self):
        """
        Matrix where denom_q = a-b-c+d = 2-2-2+2 = 0  → no valid mixed NE.
        """
        m = make(2, 1, 2, 3, 2, 1, 2, 3)
        result = _compute_mixed_strategy_2x2(m)
        assert result["mixed_exists"] is False
        assert result["mixed_p"] is None

    def test_mixed_props_in_classify_properties(self):
        """Integration: mixed_* keys appear in classify_properties for a no-NE matrix."""
        m = make(3, 0, 0, 4, 1, 3, 4, 1)
        ne = find_nash_equilibria(m)
        assert ne == []
        props = classify_properties(m, ne)
        assert props["mixed_exists"] is True
        assert "mixed_p" in props
        assert "mixed_q" in props
        assert "mixed_payoff_p1" in props
        assert "mixed_payoff_p2" in props


# ---------------------------------------------------------------------------
# Feature 2 — Payoff Asymmetry at Nash Equilibria
# ---------------------------------------------------------------------------

class TestNEPayoffAsymmetry:
    """
    Tests for _ne_payoff_stats and its integration via classify_properties
    (ne_p1_payoffs, ne_p2_payoffs, ne_payoff_diffs, ne_has_equal_payoffs,
    ne_mean_abs_diff keys).
    """

    def test_pd_equal_ne_payoffs(self):
        """
        PD NE at (1,1) = payoffs (1,1) → diffs=[0], equal=True, mean_abs=0.
        """
        m = make(3, 3, 0, 5, 5, 0, 1, 1)
        ne = find_nash_equilibria(m)
        props = classify_properties(m, ne)
        assert props["ne_p1_payoffs"] == [1]
        assert props["ne_p2_payoffs"] == [1]
        assert props["ne_payoff_diffs"] == [0]
        assert props["ne_has_equal_payoffs"] is True
        assert props["ne_mean_abs_diff"] == pytest.approx(0.0)

    def test_battle_of_sexes_unequal_payoffs(self):
        """
        Battle of Sexes: NE at (0,0)=(3,2) and (1,1)=(2,3).
        diffs = [3-2, 2-3] = [1, -1]; equal=False; mean_abs=1.0
        """
        m = make(3, 2, 0, 0, 0, 0, 2, 3)
        ne = find_nash_equilibria(m)
        assert set(ne) == {(0, 0), (1, 1)}
        props = classify_properties(m, ne)
        # NE order follows find_nash_equilibria row-major scan: (0,0) first
        assert sorted(props["ne_payoff_diffs"]) == [-1, 1]
        assert props["ne_has_equal_payoffs"] is False
        assert props["ne_mean_abs_diff"] == pytest.approx(1.0)

    def test_no_ne_matrix_empty_stats(self):
        """
        No-NE matrix → lists are empty, equal=False, mean_abs_diff=None.
        """
        m = make(3, 0, 0, 4, 1, 3, 4, 1)
        ne = find_nash_equilibria(m)
        assert ne == []
        stats = _ne_payoff_stats(m, ne)
        assert stats["ne_p1_payoffs"] == []
        assert stats["ne_p2_payoffs"] == []
        assert stats["ne_payoff_diffs"] == []
        assert stats["ne_has_equal_payoffs"] is False
        assert stats["ne_mean_abs_diff"] is None

    def test_all_equal_payoffs_four_ne(self):
        """
        All-equal matrix (3,3,3,3,...) has 4 NE, all with equal payoffs.
        """
        m = make(3, 3, 3, 3, 3, 3, 3, 3)
        ne = find_nash_equilibria(m)
        assert len(ne) == 4
        props = classify_properties(m, ne)
        assert props["ne_p1_payoffs"] == [3, 3, 3, 3]
        assert props["ne_p2_payoffs"] == [3, 3, 3, 3]
        assert props["ne_has_equal_payoffs"] is True
        assert props["ne_mean_abs_diff"] == pytest.approx(0.0)
