export const PRESETS = {
  "Prisoner's Dilemma": [3, 3, 0, 5, 5, 0, 1, 1],
  "Stag Hunt": [4, 4, 1, 3, 3, 1, 2, 2],
  "Chicken": [3, 3, 1, 4, 4, 1, 0, 0],
  "Coordination Game": [2, 2, 0, 0, 0, 0, 2, 2],
  "Battle of the Sexes": [3, 2, 0, 0, 0, 0, 2, 3],
  "Matching Pennies": [1, -1, -1, 1, -1, 1, 1, -1],
  "All Equal (4 NE)": [3, 3, 3, 3, 3, 3, 3, 3],
};

export const GAME_TYPE_COLORS = {
  "Prisoner's Dilemma": "#E8610A",
  Harmony: "#4CAF82",
  Deadlock: "#C0392B",
  "Battle of the Sexes": "#D97706",
  "Stag Hunt": "#1D8A6B",
  Chicken: "#C84B31",
  Coordination: "#3A9BD5",
  "Zero-Sum": "#9B59B6",
  "Dominant (P1 only)": "#F39C12",
  "Dominant (P2 only)": "#E67E22",
  "No Equilibrium": "#7A7570",
  Other: "#4A4540",
};

export const GAME_TYPE_DESCRIPTIONS = {
  "Zero-Sum":
    "A zero-sum game: one player's gain is exactly the other's loss. The total welfare is constant across all outcomes. Classic examples: chess, poker, matching pennies.",
  "Prisoner's Dilemma":
    "A social dilemma: both players have a dominant strategy, but the Nash equilibrium leaves both worse off than if they had cooperated. Rational individual behaviour produces a collectively suboptimal result.",
  Harmony:
    "A harmony game: both players have dominant strategies and the Nash equilibrium is Pareto-efficient. Rational self-interest aligns with the socially optimal outcome.",
  Deadlock:
    "Both players have dominant strategies leading to an equilibrium, but unlike the Prisoner's Dilemma the cooperative outcome is not better for both. Mutual defection is both rational and efficient.",
  "Battle of the Sexes":
    "An asymmetric coordination game with two diagonal equilibria. Both players want to coordinate, but each prefers a different equilibrium.",
  "Stag Hunt":
    "A symmetric coordination game with one high-reward cooperative equilibrium and one safer fallback equilibrium. Trust matters because failing to coordinate can be costly.",
  Chicken:
    "A symmetric anti-coordination game with off-diagonal equilibria. Each player wants the other side to yield, creating brinkmanship instead of stable mutual cooperation.",
  Coordination:
    "A coordination-style game: multiple Nash equilibria exist and the main strategic problem is choosing which stable outcome to coordinate on.",
  "Dominant (P1 only)":
    "Only Player 1 has a dominant strategy. Player 2's best response depends on what Player 1 does, but Player 1 always plays the same way.",
  "Dominant (P2 only)":
    "Only Player 2 has a dominant strategy. Player 1's best response depends on what Player 2 does, but Player 2 always plays the same way.",
  "No Equilibrium":
    "No pure-strategy Nash equilibrium exists. Best responses cycle, so the stable object is a mixed-strategy equilibrium.",
  Other:
    "A game that does not fit neatly into the classic taxonomy. Neither player has a dominant strategy and there is at least one pure-strategy Nash equilibrium.",
};

export function buildMatrix(payoffs) {
  return [
    [[payoffs[0], payoffs[1]], [payoffs[2], payoffs[3]]],
    [[payoffs[4], payoffs[5]], [payoffs[6], payoffs[7]]],
  ];
}

export function findNashEquilibria(matrix) {
  const equilibria = [];
  for (let row = 0; row < matrix.length; row += 1) {
    for (let col = 0; col < matrix[row].length; col += 1) {
      const p1 = matrix[row][col][0];
      const p2 = matrix[row][col][1];

      let rowBest = true;
      for (let otherRow = 0; otherRow < matrix.length; otherRow += 1) {
        if (matrix[otherRow][col][0] > p1) {
          rowBest = false;
          break;
        }
      }

      let colBest = true;
      for (let otherCol = 0; otherCol < matrix[row].length; otherCol += 1) {
        if (matrix[row][otherCol][1] > p2) {
          colBest = false;
          break;
        }
      }

      if (rowBest && colBest) {
        equilibria.push([row, col]);
      }
    }
  }
  return equilibria;
}

function hasDominantStrategy(matrix, player) {
  const rows = matrix.length;
  const cols = matrix[0].length;

  if (player === 0) {
    for (let candidateRow = 0; candidateRow < rows; candidateRow += 1) {
      let dominant = true;
      for (let row = 0; row < rows; row += 1) {
        for (let col = 0; col < cols; col += 1) {
          if (matrix[candidateRow][col][0] < matrix[row][col][0]) {
            dominant = false;
            break;
          }
        }
        if (!dominant) {
          break;
        }
      }
      if (dominant) {
        return true;
      }
    }
    return false;
  }

  for (let candidateCol = 0; candidateCol < cols; candidateCol += 1) {
    let dominant = true;
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        if (matrix[row][candidateCol][1] < matrix[row][col][1]) {
          dominant = false;
          break;
        }
      }
      if (!dominant) {
        break;
      }
    }
    if (dominant) {
      return true;
    }
  }

  return false;
}

function paretoDominated(targetRow, targetCol, matrix) {
  const hereP1 = matrix[targetRow][targetCol][0];
  const hereP2 = matrix[targetRow][targetCol][1];

  for (let row = 0; row < matrix.length; row += 1) {
    for (let col = 0; col < matrix[row].length; col += 1) {
      if (row === targetRow && col === targetCol) {
        continue;
      }
      const thereP1 = matrix[row][col][0];
      const thereP2 = matrix[row][col][1];
      if (thereP1 >= hereP1 && thereP2 >= hereP2 && (thereP1 > hereP1 || thereP2 > hereP2)) {
        return true;
      }
    }
  }

  return false;
}

export function computeMixedStrategy2x2(matrix) {
  if (matrix.length !== 2 || matrix[0].length !== 2) {
    return {
      mixed_exists: false,
      mixed_p: null,
      mixed_q: null,
      mixed_payoff_p1: null,
      mixed_payoff_p2: null,
    };
  }

  const a = Number(matrix[0][0][0]);
  const e = Number(matrix[0][0][1]);
  const b = Number(matrix[0][1][0]);
  const f = Number(matrix[0][1][1]);
  const c = Number(matrix[1][0][0]);
  const g = Number(matrix[1][0][1]);
  const d = Number(matrix[1][1][0]);
  const h = Number(matrix[1][1][1]);

  const denomP = e - g - f + h;
  const denomQ = a - b - c + d;

  if (denomP === 0 || denomQ === 0) {
    return {
      mixed_exists: false,
      mixed_p: null,
      mixed_q: null,
      mixed_payoff_p1: null,
      mixed_payoff_p2: null,
    };
  }

  const eps = 1e-9;
  let p = (h - g) / denomP;
  let q = (d - b) / denomQ;

  if (!(p >= -eps && p <= 1 + eps && q >= -eps && q <= 1 + eps)) {
    return {
      mixed_exists: false,
      mixed_p: null,
      mixed_q: null,
      mixed_payoff_p1: null,
      mixed_payoff_p2: null,
    };
  }

  p = Math.max(0, Math.min(1, p));
  q = Math.max(0, Math.min(1, q));

  return {
    mixed_exists: true,
    mixed_p: round(p, 6),
    mixed_q: round(q, 6),
    mixed_payoff_p1: round(q * a + (1 - q) * b, 6),
    mixed_payoff_p2: round(p * e + (1 - p) * g, 6),
  };
}

function nePayoffStats(matrix, nePositions) {
  if (nePositions.length === 0) {
    return {
      ne_p1_payoffs: [],
      ne_p2_payoffs: [],
      ne_payoff_diffs: [],
      ne_has_equal_payoffs: false,
      ne_mean_abs_diff: null,
    };
  }

  const p1Payoffs = nePositions.map(([row, col]) => matrix[row][col][0]);
  const p2Payoffs = nePositions.map(([row, col]) => matrix[row][col][1]);
  const diffs = p1Payoffs.map((value, index) => value - p2Payoffs[index]);
  const absMean = diffs.reduce((sum, diff) => sum + Math.abs(diff), 0) / diffs.length;

  return {
    ne_p1_payoffs: p1Payoffs,
    ne_p2_payoffs: p2Payoffs,
    ne_payoff_diffs: diffs,
    ne_has_equal_payoffs: diffs.some((diff) => diff === 0),
    ne_mean_abs_diff: absMean,
  };
}

export function classifyProperties(matrix, nePositions) {
  const p1Dominant = hasDominantStrategy(matrix, 0);
  const p2Dominant = hasDominantStrategy(matrix, 1);

  const sums = [];
  let maxWelfare = -Infinity;
  for (let row = 0; row < matrix.length; row += 1) {
    for (let col = 0; col < matrix[row].length; col += 1) {
      const welfare = matrix[row][col][0] + matrix[row][col][1];
      sums.push(welfare);
      if (welfare > maxWelfare) {
        maxWelfare = welfare;
      }
    }
  }
  const isZeroSum = sums.every((value) => value === sums[0]);

  let isSymmetric = matrix.length === matrix[0].length;
  if (isSymmetric) {
    for (let row = 0; row < matrix.length; row += 1) {
      for (let col = 0; col < matrix[row].length; col += 1) {
        if (matrix[row][col][0] !== matrix[col][row][1]) {
          isSymmetric = false;
          break;
        }
      }
      if (!isSymmetric) {
        break;
      }
    }
  }

  const neWelfare = nePositions.map(([row, col]) => matrix[row][col][0] + matrix[row][col][1]);
  const maxNeWelfare = neWelfare.length > 0 ? Math.max(...neWelfare) : 0;
  const welfareLoss = maxWelfare - maxNeWelfare;

  const paretoFlags = nePositions.map(([row, col]) => paretoDominated(row, col, matrix));
  const hasParetoDominatedNe = paretoFlags.some(Boolean);

  const mixed = nePositions.length === 0 ? computeMixedStrategy2x2(matrix) : {
    mixed_exists: false,
    mixed_p: null,
    mixed_q: null,
    mixed_payoff_p1: null,
    mixed_payoff_p2: null,
  };

  const asym = nePayoffStats(matrix, nePositions);

  return {
    p1_has_dominant: p1Dominant,
    p2_has_dominant: p2Dominant,
    both_dominant: p1Dominant && p2Dominant,
    is_zero_sum: isZeroSum,
    is_symmetric: isSymmetric,
    ne_count: nePositions.length,
    has_pareto_dom_ne: hasParetoDominatedNe,
    all_ne_pareto_eff: !hasParetoDominatedNe,
    max_welfare: maxWelfare,
    ne_welfare: neWelfare,
    welfare_loss: welfareLoss,
    mixed_exists: mixed.mixed_exists,
    mixed_p: mixed.mixed_p,
    mixed_q: mixed.mixed_q,
    mixed_payoff_p1: mixed.mixed_payoff_p1,
    mixed_payoff_p2: mixed.mixed_payoff_p2,
    ne_p1_payoffs: asym.ne_p1_payoffs,
    ne_p2_payoffs: asym.ne_p2_payoffs,
    ne_payoff_diffs: asym.ne_payoff_diffs,
    ne_has_equal_payoffs: asym.ne_has_equal_payoffs,
    ne_mean_abs_diff: asym.ne_mean_abs_diff,
  };
}

function isDiagonalPair(nePositions) {
  return nePositions.length === 2
    && nePositions.some(([row, col]) => row === 0 && col === 0)
    && nePositions.some(([row, col]) => row === 1 && col === 1);
}

function isOffDiagonalPair(nePositions) {
  return nePositions.length === 2
    && nePositions.some(([row, col]) => row === 0 && col === 1)
    && nePositions.some(([row, col]) => row === 1 && col === 0);
}

export function classifyGameType(props, nePositions, matrix) {
  if (props.is_zero_sum) {
    return "Zero-Sum";
  }

  if (props.both_dominant) {
    if (props.has_pareto_dom_ne && props.welfare_loss > 0) {
      return "Prisoner's Dilemma";
    }
    if (props.welfare_loss === 0) {
      return "Harmony";
    }
    return "Deadlock";
  }

  if (matrix && isDiagonalPair(nePositions)) {
    const tl = matrix[0][0];
    const br = matrix[1][1];

    const p1PrefersTl = tl[0] > br[0];
    const p1PrefersBr = br[0] > tl[0];
    const p2PrefersTl = tl[1] > br[1];
    const p2PrefersBr = br[1] > tl[1];

    if ((p1PrefersTl && p2PrefersBr) || (p1PrefersBr && p2PrefersTl)) {
      return "Battle of the Sexes";
    }

    if (props.is_symmetric) {
      const tlWelfare = tl[0] + tl[1];
      const brWelfare = br[0] + br[1];
      if (tlWelfare !== brWelfare) {
        return "Stag Hunt";
      }
    }

    return "Coordination";
  }

  if (matrix && isOffDiagonalPair(nePositions) && props.is_symmetric) {
    return "Chicken";
  }

  if (nePositions.length >= 2 && props.is_symmetric) {
    return "Coordination";
  }

  if (props.p1_has_dominant && !props.p2_has_dominant) {
    return "Dominant (P1 only)";
  }

  if (props.p2_has_dominant && !props.p1_has_dominant) {
    return "Dominant (P2 only)";
  }

  if (props.ne_count === 0) {
    return "No Equilibrium";
  }

  return "Other";
}

export function classifyFull(matrix) {
  const ne = findNashEquilibria(matrix);
  const props = classifyProperties(matrix, ne);
  const label = classifyGameType(props, ne, matrix);
  return { ne, props, label };
}

function round(value, digits) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}
