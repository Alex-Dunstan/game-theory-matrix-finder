import numpy as np
import pytest

from src.analysis import (
    GAME_TYPE_DESCRIPTIONS,
    _compute_mixed_strategy_2x2,
    classify_full,
    classify_properties,
)
from src.matrix_permutations import find_nash_equilibria


def make(tl1, tl2, tr1, tr2, bl1, bl2, br1, br2) -> np.ndarray:
    return np.array(
        [[[tl1, tl2], [tr1, tr2]], [[bl1, bl2], [br1, br2]]],
        dtype=np.int32,
    )


def classify(matrix: np.ndarray):
    ne = find_nash_equilibria(matrix)
    return classify_full(matrix, ne)


def test_prisoners_dilemma_label():
    matrix = make(3, 3, 0, 5, 5, 0, 1, 1)
    _, label = classify(matrix)
    assert label == "Prisoner's Dilemma"


def test_coordination_label():
    matrix = make(2, 2, 0, 0, 0, 0, 2, 2)
    _, label = classify(matrix)
    assert label == "Coordination"


def test_zero_sum_label():
    matrix = make(1, -1, -1, 1, -1, 1, 1, -1)
    _, label = classify(matrix)
    assert label == "Zero-Sum"


def test_no_equilibrium_label():
    matrix = make(3, 0, 0, 4, 1, 3, 4, 1)
    _, label = classify(matrix)
    assert label == "No Equilibrium"


def test_battle_of_the_sexes_label():
    matrix = make(3, 2, 0, 0, 0, 0, 2, 3)
    _, label = classify(matrix)
    assert label == "Battle of the Sexes"


def test_stag_hunt_label():
    matrix = make(4, 4, 1, 3, 3, 1, 2, 2)
    _, label = classify(matrix)
    assert label == "Stag Hunt"


def test_chicken_label():
    matrix = make(3, 3, 1, 4, 4, 1, 0, 0)
    _, label = classify(matrix)
    assert label == "Chicken"


def test_descriptions_cover_public_labels():
    expected = {
        "Zero-Sum",
        "Prisoner's Dilemma",
        "Harmony",
        "Deadlock",
        "Battle of the Sexes",
        "Stag Hunt",
        "Chicken",
        "Coordination",
        "Dominant (P1 only)",
        "Dominant (P2 only)",
        "No Equilibrium",
        "Other",
    }
    assert expected <= set(GAME_TYPE_DESCRIPTIONS)


def test_matching_pennies_mixed_equilibrium():
    matrix = make(1, -1, -1, 1, -1, 1, 1, -1)
    mixed = _compute_mixed_strategy_2x2(matrix)
    assert mixed["mixed_exists"] is True
    assert mixed["mixed_p"] == pytest.approx(0.5)
    assert mixed["mixed_q"] == pytest.approx(0.5)


def test_mixed_keys_present_for_no_pure_equilibrium():
    matrix = make(3, 0, 0, 4, 1, 3, 4, 1)
    ne = find_nash_equilibria(matrix)
    props = classify_properties(matrix, ne)
    assert ne == []
    assert props["mixed_exists"] is True
    assert props["mixed_p"] == pytest.approx(1 / 3, abs=1e-5)
    assert props["mixed_q"] == pytest.approx(2 / 3, abs=1e-5)
