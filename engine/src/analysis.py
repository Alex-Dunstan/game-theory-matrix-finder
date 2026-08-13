"""
analysis.py — Game type classification for 2-player payoff matrices.

Two-layer approach:
  Layer 1 — classify_properties(matrix) → dict of boolean/numeric flags
  Layer 2 — classify_game_type(props, ne_positions) → human-readable string label

Both layers work on a single matrix (numpy array, shape (R, C, 2)).
For bulk classification of large datasets, use classify_batch() which processes
a numpy batch (N, R, C, 2) in vectorised operations where possible.

Named game types (in priority order):
  "Zero-Sum"             — payoffs sum to constant in every cell
  "Prisoner's Dilemma"   — both dominant strategies, NE is Pareto-dominated
  "Harmony"              — both dominant strategies, NE is Pareto-efficient
  "Deadlock"             — both dominant strategies, inefficient but not PD
  "Battle of the Sexes"  — 2 diagonal NE, players prefer different ones
  "Stag Hunt"            — symmetric coordination with a payoff-dominant diagonal
  "Chicken"              — symmetric anti-coordination with off-diagonal NE
  "Coordination"         — ≥2 NE with coordination structure not covered above
  "Dominant (P1 only)"   — only Player 1 has dominant strategy
  "Dominant (P2 only)"   — only Player 2 has dominant strategy
  "No Equilibrium"       — no pure-strategy NE
  "Other"                — everything else

Usage
-----
    from src.analysis import classify_properties, classify_game_type
    from src.matrix_permutations import find_nash_equilibria
    import numpy as np

    m = np.array([[[3,3],[0,5]],[[5,0],[1,1]]], dtype=np.int32)
    ne = find_nash_equilibria(m)
    props = classify_properties(m, ne)
    label = classify_game_type(props, ne)   # → "Prisoner's Dilemma"
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Props = Dict[str, Any]


# ---------------------------------------------------------------------------
# Layer 1: Boolean + numeric properties for a single matrix
# ---------------------------------------------------------------------------

def _has_dominant_strategy(matrix: np.ndarray, player: int) -> bool:
    """
    Return True if `player` (0=row, 1=col) has a weakly dominant strategy.

    A strategy s* weakly dominates all others if, for every possible opponent
    strategy, s* gives at least as good a payoff as any other choice.

    For the row player:  there exists row r* such that
        matrix[r*, c, 0] >= matrix[r, c, 0]  for ALL r, c

    For the column player:  there exists col c* such that
        matrix[r, c*, 1] >= matrix[r, c, 1]  for ALL r, c
    """
    R, C, _ = matrix.shape
    payoffs = matrix[:, :, player]           # (R, C)

    if player == 0:   # row player — look for a dominant row
        for r_star in range(R):
            if np.all(payoffs[r_star, :] >= payoffs):
                return True
    else:             # col player — look for a dominant column
        for c_star in range(C):
            if np.all(payoffs[:, c_star][:, None] >= payoffs):
                return True
    return False


def _pareto_dominated(cell_r: int, cell_c: int, matrix: np.ndarray) -> bool:
    """
    Return True if cell (cell_r, cell_c) is Pareto-dominated by any other cell.

    A cell X is Pareto-dominated by cell Y if:
        Y gives at least as much to both players, and strictly more to at least one.
    """
    p1_here = int(matrix[cell_r, cell_c, 0])
    p2_here = int(matrix[cell_r, cell_c, 1])
    R, C, _ = matrix.shape

    for r in range(R):
        for c in range(C):
            if r == cell_r and c == cell_c:
                continue
            p1_there = int(matrix[r, c, 0])
            p2_there = int(matrix[r, c, 1])
            if p1_there >= p1_here and p2_there >= p2_here:
                if p1_there > p1_here or p2_there > p2_here:
                    return True
    return False


def _compute_mixed_strategy_2x2(matrix: np.ndarray) -> Dict[str, Any]:
    """
    Compute the mixed-strategy Nash equilibrium for a 2×2 payoff matrix.

    Layout:
        Row 0: [(a, e), (b, f)]   → matrix[0,0]=(a,e), matrix[0,1]=(b,f)
        Row 1: [(c, g), (d, h)]   → matrix[1,0]=(c,g), matrix[1,1]=(d,h)

    P1 mixes with probability p (plays Row 0), making P2 indifferent:
        p·e + (1−p)·g = p·f + (1−p)·h  →  p = (h−g) / (e−g−f+h)

    P2 mixes with probability q (plays Col 0), making P1 indifferent:
        q·a + (1−q)·b = q·c + (1−q)·d  →  q = (d−b) / (a−b−c+d)

    Returns a dict with keys:
        mixed_exists     bool
        mixed_p          float | None   P1 plays Row 0 with this probability
        mixed_q          float | None   P2 plays Col 0 with this probability
        mixed_payoff_p1  float | None   P1 expected payoff at mixed NE
        mixed_payoff_p2  float | None   P2 expected payoff at mixed NE
    """
    _null = {"mixed_exists": False, "mixed_p": None, "mixed_q": None,
             "mixed_payoff_p1": None, "mixed_payoff_p2": None}

    if matrix.shape != (2, 2, 2):
        return _null

    a, e = float(matrix[0, 0, 0]), float(matrix[0, 0, 1])
    b, f = float(matrix[0, 1, 0]), float(matrix[0, 1, 1])
    c, g = float(matrix[1, 0, 0]), float(matrix[1, 0, 1])
    d, h = float(matrix[1, 1, 0]), float(matrix[1, 1, 1])

    denom_p = e - g - f + h   # denominator for p
    denom_q = a - b - c + d   # denominator for q

    if denom_p == 0.0 or denom_q == 0.0:
        return _null

    p = (h - g) / denom_p
    q = (d - b) / denom_q

    eps = 1e-9
    if not (-eps <= p <= 1.0 + eps and -eps <= q <= 1.0 + eps):
        return _null

    # Clamp to [0, 1] to absorb floating-point edge cases
    p = max(0.0, min(1.0, p))
    q = max(0.0, min(1.0, q))

    ep1 = round(q * a + (1.0 - q) * b, 6)
    ep2 = round(p * e + (1.0 - p) * g, 6)

    return {
        "mixed_exists":    True,
        "mixed_p":         round(p, 6),
        "mixed_q":         round(q, 6),
        "mixed_payoff_p1": ep1,
        "mixed_payoff_p2": ep2,
    }


def _ne_payoff_stats(
    matrix: np.ndarray,
    ne_positions: List[Tuple[int, int]],
) -> Dict[str, Any]:
    """
    Compute per-NE payoff statistics for a single matrix.

    Returns a dict with keys:
        ne_p1_payoffs        list[int]   — P1 payoff at each NE
        ne_p2_payoffs        list[int]   — P2 payoff at each NE
        ne_payoff_diffs      list[int]   — (P1−P2) at each NE
        ne_has_equal_payoffs bool        — True if any NE has P1 == P2
        ne_mean_abs_diff     float|None  — mean |P1−P2| across all NE; None if no NE
    """
    if not ne_positions:
        return {
            "ne_p1_payoffs":        [],
            "ne_p2_payoffs":        [],
            "ne_payoff_diffs":      [],
            "ne_has_equal_payoffs": False,
            "ne_mean_abs_diff":     None,
        }

    p1_payoffs = [int(matrix[r, c, 0]) for r, c in ne_positions]
    p2_payoffs = [int(matrix[r, c, 1]) for r, c in ne_positions]
    diffs      = [p1 - p2 for p1, p2 in zip(p1_payoffs, p2_payoffs)]

    return {
        "ne_p1_payoffs":        p1_payoffs,
        "ne_p2_payoffs":        p2_payoffs,
        "ne_payoff_diffs":      diffs,
        "ne_has_equal_payoffs": any(d == 0 for d in diffs),
        "ne_mean_abs_diff":     sum(abs(d) for d in diffs) / len(diffs),
    }


def classify_properties(
    matrix: np.ndarray,
    ne_positions: List[Tuple[int, int]],
) -> Props:
    """
    Compute a dictionary of structural properties for a single payoff matrix.

    Parameters
    ----------
    matrix       : np.ndarray, shape (R, C, 2)
    ne_positions : list of (row, col) Nash equilibria (from find_nash_equilibria)

    Returns
    -------
    props : dict with the following keys:

      p1_has_dominant     bool       — P1 has a weakly dominant strategy
      p2_has_dominant     bool       — P2 has a weakly dominant strategy
      both_dominant       bool       — both players have dominant strategies
      is_zero_sum         bool       — all cells: p1+p2 == constant
      is_symmetric        bool       — matrix[r,c,0] == matrix[c,r,1] (requires R==C)
      ne_count            int        — number of pure-strategy NE
      has_pareto_dom_ne   bool       — any NE is Pareto-dominated by another cell
      all_ne_pareto_eff   bool       — no NE is Pareto-dominated
      max_welfare         int        — max(p1+p2) over all cells
      ne_welfare          list       — p1+p2 at each NE position
      welfare_loss        int        — max_welfare − max(ne_welfare); 0 means NE is optimal

      mixed_exists        bool       — True if a valid mixed-strategy NE exists
                                       (only computed for 2×2 matrices with 0 pure NE)
      mixed_p             float|None — P1 plays Row 0 with this probability
      mixed_q             float|None — P2 plays Col 0 with this probability
      mixed_payoff_p1     float|None — P1 expected payoff at mixed NE
      mixed_payoff_p2     float|None — P2 expected payoff at mixed NE

      ne_p1_payoffs        list[int]  — P1 payoff at each NE (empty if no pure NE)
      ne_p2_payoffs        list[int]  — P2 payoff at each NE
      ne_payoff_diffs      list[int]  — (P1−P2) at each NE
      ne_has_equal_payoffs bool       — True if any NE has P1 == P2
      ne_mean_abs_diff     float|None — mean |P1−P2| across all NE; None if no pure NE
    """
    R, C, _ = matrix.shape
    p1 = matrix[:, :, 0]
    p2 = matrix[:, :, 1]
    welfare = p1 + p2                        # (R, C)

    # Dominant strategies
    p1_dom = _has_dominant_strategy(matrix, 0)
    p2_dom = _has_dominant_strategy(matrix, 1)

    # Zero-sum: all payoff sums identical
    sums = (p1 + p2).ravel()
    is_zero_sum = bool(np.all(sums == sums[0]))

    # Symmetry: matrix[r,c,0] == matrix[c,r,1] (only meaningful for square)
    if R == C:
        is_sym = all(
            int(matrix[r, c, 0]) == int(matrix[c, r, 1])
            for r in range(R) for c in range(C)
        )
    else:
        is_sym = False

    # NE welfare
    ne_w = [int(welfare[r, c]) for r, c in ne_positions]
    max_w = int(welfare.max())
    max_ne_w = max(ne_w) if ne_w else 0
    w_loss = max_w - max_ne_w

    # Pareto efficiency of each NE
    ne_pareto_dom = [_pareto_dominated(r, c, matrix) for r, c in ne_positions]
    has_pd_ne = any(ne_pareto_dom)
    all_pe_ne = not has_pd_ne

    # Mixed-strategy equilibrium — 2×2 matrices with 0 pure NE only
    if R == 2 and C == 2 and len(ne_positions) == 0:
        mixed = _compute_mixed_strategy_2x2(matrix)
    else:
        mixed = {"mixed_exists": False, "mixed_p": None, "mixed_q": None,
                 "mixed_payoff_p1": None, "mixed_payoff_p2": None}

    # Payoff asymmetry at Nash equilibria
    asym = _ne_payoff_stats(matrix, ne_positions)

    return {
        "p1_has_dominant":     p1_dom,
        "p2_has_dominant":     p2_dom,
        "both_dominant":       p1_dom and p2_dom,
        "is_zero_sum":         is_zero_sum,
        "is_symmetric":        is_sym,
        "ne_count":            len(ne_positions),
        "has_pareto_dom_ne":   has_pd_ne,
        "all_ne_pareto_eff":   all_pe_ne,
        "max_welfare":         max_w,
        "ne_welfare":          ne_w,
        "welfare_loss":        w_loss,
        # Mixed strategy
        "mixed_exists":        mixed["mixed_exists"],
        "mixed_p":             mixed["mixed_p"],
        "mixed_q":             mixed["mixed_q"],
        "mixed_payoff_p1":     mixed["mixed_payoff_p1"],
        "mixed_payoff_p2":     mixed["mixed_payoff_p2"],
        # Payoff asymmetry
        "ne_p1_payoffs":       asym["ne_p1_payoffs"],
        "ne_p2_payoffs":       asym["ne_p2_payoffs"],
        "ne_payoff_diffs":     asym["ne_payoff_diffs"],
        "ne_has_equal_payoffs": asym["ne_has_equal_payoffs"],
        "ne_mean_abs_diff":    asym["ne_mean_abs_diff"],
    }


# ---------------------------------------------------------------------------
# Layer 2: Named game type (derived from properties)
# ---------------------------------------------------------------------------

# Plain-English descriptions keyed by type label
GAME_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "Zero-Sum": (
        "A zero-sum game: one player's gain is exactly the other's loss. "
        "The total welfare is constant across all outcomes. "
        "Classic examples: chess, poker, matching pennies."
    ),
    "Prisoner's Dilemma": (
        "A social dilemma: both players have a dominant strategy (defect), "
        "but the Nash equilibrium leaves both worse off than if they had "
        "cooperated. Rational individual behaviour produces a collectively "
        "suboptimal result. Welfare loss > 0."
    ),
    "Harmony": (
        "A harmony game: both players have dominant strategies AND the "
        "Nash equilibrium is Pareto-efficient. Rational self-interest "
        "happens to align with the socially optimal outcome — no dilemma."
    ),
    "Deadlock": (
        "Both players have dominant strategies leading to an equilibrium, "
        "but unlike the Prisoner's Dilemma the cooperative outcome is not "
        "better for both. Mutual defection is both rational and efficient."
    ),
    "Battle of the Sexes": (
        "An asymmetric coordination game with two diagonal equilibria. "
        "Both players want to coordinate, but each prefers a different "
        "equilibrium, so the main challenge is choosing which outcome to meet at."
    ),
    "Stag Hunt": (
        "A symmetric coordination game with one high-reward cooperative "
        "equilibrium and one safer fallback equilibrium. Trust matters because "
        "failing to coordinate can be costly."
    ),
    "Chicken": (
        "A symmetric anti-coordination game with off-diagonal equilibria. "
        "Each player wants the other side to yield, creating brinkmanship "
        "instead of stable mutual cooperation."
    ),
    "Coordination": (
        "A coordination-style game: multiple Nash equilibria exist and the "
        "main strategic problem is choosing which stable outcome to coordinate on."
    ),
    "Dominant (P1 only)": (
        "Only Player 1 has a dominant strategy. Player 2's best response "
        "depends on what Player 1 does, but P1 always plays the same way."
    ),
    "Dominant (P2 only)": (
        "Only Player 2 has a dominant strategy. Player 1's best response "
        "depends on what Player 2 does, but P2 always plays the same way."
    ),
    "No Equilibrium": (
        "No pure-strategy Nash equilibrium exists. Neither player has a "
        "stable resting point; best responses cycle. A mixed-strategy "
        "equilibrium always exists (Nash's theorem) and is computed for "
        "2×2 matrices — see the mixed-strategy properties below."
    ),
    "Other": (
        "A game that doesn't fit neatly into the classic taxonomy. "
        "Neither player has a dominant strategy and there is at least one "
        "pure-strategy Nash equilibrium."
    ),
}


def _is_diagonal_pair(ne_positions: List[Tuple[int, int]]) -> bool:
    return set(ne_positions) == {(0, 0), (1, 1)}


def _is_off_diagonal_pair(ne_positions: List[Tuple[int, int]]) -> bool:
    return set(ne_positions) == {(0, 1), (1, 0)}


def classify_game_type(
    props: Props,
    ne_positions: List[Tuple[int, int]],
    matrix: np.ndarray | None = None,
) -> str:
    """
    Assign a named game type label based on the properties dict.

    Checked in strict priority order — the first matching rule wins.
    """
    if props["is_zero_sum"]:
        return "Zero-Sum"

    if props["both_dominant"]:
        # True PD: the NE is Pareto-suboptimal AND achieves less than max welfare.
        # (welfare_loss > 0 rules out the edge case where one NE Pareto-dominates
        # another NE but the dominant NE still achieves max social welfare.)
        if props["has_pareto_dom_ne"] and props["welfare_loss"] > 0:
            return "Prisoner's Dilemma"
        elif props["welfare_loss"] == 0:
            return "Harmony"
        else:
            return "Deadlock"

    if matrix is not None and len(ne_positions) == 2 and _is_diagonal_pair(ne_positions):
        tl = matrix[0, 0]
        br = matrix[1, 1]

        p1_prefers_tl = int(tl[0]) > int(br[0])
        p1_prefers_br = int(br[0]) > int(tl[0])
        p2_prefers_tl = int(tl[1]) > int(br[1])
        p2_prefers_br = int(br[1]) > int(tl[1])

        if (p1_prefers_tl and p2_prefers_br) or (p1_prefers_br and p2_prefers_tl):
            return "Battle of the Sexes"

        if props["is_symmetric"]:
            tl_welfare = int(tl[0] + tl[1])
            br_welfare = int(br[0] + br[1])
            if tl_welfare != br_welfare:
                return "Stag Hunt"

        return "Coordination"

    if matrix is not None and len(ne_positions) == 2 and _is_off_diagonal_pair(ne_positions):
        if props["is_symmetric"]:
            return "Chicken"

    if len(ne_positions) >= 2 and props["is_symmetric"]:
        return "Coordination"

    if props["p1_has_dominant"] and not props["p2_has_dominant"]:
        return "Dominant (P1 only)"

    if props["p2_has_dominant"] and not props["p1_has_dominant"]:
        return "Dominant (P2 only)"

    if props["ne_count"] == 0:
        return "No Equilibrium"

    return "Other"


def classify_full(
    matrix: np.ndarray,
    ne_positions: List[Tuple[int, int]],
) -> Tuple[Props, str]:
    """
    Convenience wrapper: compute both layers and return (props, game_type).
    """
    props = classify_properties(matrix, ne_positions)
    label = classify_game_type(props, ne_positions, matrix)
    return props, label


# ---------------------------------------------------------------------------
# Bulk classification (used by enrich_datasets.py)
# ---------------------------------------------------------------------------

def classify_rows_batch(
    matrices_flat: np.ndarray,
    ne_positions_list: List[List[Tuple[int, int]]],
    rows: int,
    cols: int,
) -> List[Dict[str, Any]]:
    """
    Classify a list of matrices given their pre-computed NE positions.

    Parameters
    ----------
    matrices_flat     : (N, rows*cols*2) array of int — payoff values flat
    ne_positions_list : list of length N, each element a list of (r,c) NE positions
    rows, cols        : matrix dimensions

    Returns
    -------
    List of dicts, one per matrix, with keys:
      p1_has_dominant, p2_has_dominant, both_dominant, is_zero_sum,
      is_symmetric, has_pareto_dom_ne, all_ne_pareto_eff,
      max_welfare, ne_welfare, welfare_loss,
      mixed_exists, mixed_p, mixed_q, mixed_payoff_p1, mixed_payoff_p2,
      ne_p1_payoffs, ne_p2_payoffs, ne_payoff_diffs,
      ne_has_equal_payoffs, ne_mean_abs_diff,
      game_type
    """
    results = []
    for i, flat in enumerate(matrices_flat):
        matrix = flat.reshape(rows, cols, 2)
        ne = ne_positions_list[i]
        props, label = classify_full(matrix, ne)
        results.append({**props, "game_type": label})
    return results
