import {
  PRESETS,
  GAME_TYPE_COLORS,
  GAME_TYPE_DESCRIPTIONS,
  buildMatrix,
  classifyFull,
} from "./classifier.mjs";

const INPUT_IDS = [
  "r0c0_p1",
  "r0c0_p2",
  "r0c1_p1",
  "r0c1_p2",
  "r1c0_p1",
  "r1c0_p2",
  "r1c1_p1",
  "r1c1_p2",
];

const DEFAULT_PRESET = "Prisoner's Dilemma";

function init() {
  const presetSelect = document.querySelector("#preset");
  const analyzeButton = document.querySelector("#analyze");
  const resetButton = document.querySelector("#reset");

  for (const name of Object.keys(PRESETS)) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    presetSelect.append(option);
  }

  presetSelect.value = DEFAULT_PRESET;
  applyPreset(DEFAULT_PRESET);

  presetSelect.addEventListener("change", () => {
    applyPreset(presetSelect.value);
  });

  analyzeButton.addEventListener("click", render);
  resetButton.addEventListener("click", () => {
    presetSelect.value = DEFAULT_PRESET;
    applyPreset(DEFAULT_PRESET);
  });

  for (const inputId of INPUT_IDS) {
    document.querySelector(`#${inputId}`).addEventListener("input", render);
  }
}

function applyPreset(name) {
  const values = PRESETS[name];
  if (!values) {
    return;
  }

  INPUT_IDS.forEach((inputId, index) => {
    document.querySelector(`#${inputId}`).value = String(values[index]);
  });

  render();
}

function currentPayoffs() {
  return INPUT_IDS.map((inputId) => {
    const raw = document.querySelector(`#${inputId}`).value;
    const parsed = Number.parseInt(raw, 10);
    return Number.isNaN(parsed) ? 0 : parsed;
  });
}

function render() {
  const payoffs = currentPayoffs();
  const matrix = buildMatrix(payoffs);
  const result = classifyFull(matrix);

  renderMatrix(matrix, result.ne);
  renderSummary(matrix, result);
  renderClassification(result);
  renderProperties(result.props, result.ne);
}

function renderMatrix(matrix, equilibria) {
  const neKeys = new Set(equilibria.map(([row, col]) => `${row}-${col}`));
  const tbody = document.querySelector("#matrix-body");
  tbody.innerHTML = "";

  for (let row = 0; row < 2; row += 1) {
    const tr = document.createElement("tr");
    for (let col = 0; col < 2; col += 1) {
      const td = document.createElement("td");
      const [p1, p2] = matrix[row][col];
      const isNe = neKeys.has(`${row}-${col}`);
      td.className = isNe ? "matrix-cell is-ne" : "matrix-cell";
      td.innerHTML = `
        <div class="cell-coord">Row ${row} / Col ${col}</div>
        <div class="cell-payoff">(${p1}, ${p2})</div>
        <div class="cell-tag">${isNe ? "NASH EQUILIBRIUM" : " "}</div>
      `;
      tr.append(td);
    }
    tbody.append(tr);
  }
}

function renderSummary(matrix, result) {
  const summary = document.querySelector("#summary");
  const positions = result.ne.map(([row, col]) => `(${row}, ${col})`).join(", ");

  if (result.ne.length === 0) {
    if (result.props.mixed_exists) {
      summary.innerHTML = `
        <div class="summary-line">Pure NE: none</div>
        <div class="summary-line">Mixed NE: P1 Row 0 = <strong>${formatNumber(result.props.mixed_p)}</strong>, P2 Col 0 = <strong>${formatNumber(result.props.mixed_q)}</strong></div>
        <div class="summary-line">Expected payoffs: <strong>(${formatNumber(result.props.mixed_payoff_p1)}, ${formatNumber(result.props.mixed_payoff_p2)})</strong></div>
      `;
      return;
    }

    summary.innerHTML = `
      <div class="summary-line">Pure NE: none</div>
      <div class="summary-line">Mixed NE: degenerate</div>
    `;
    return;
  }

  if (result.ne.length === 1) {
    const [row, col] = result.ne[0];
    summary.innerHTML = `
      <div class="summary-line">Pure NE: <strong>1</strong></div>
      <div class="summary-line">Position: <strong>(${row}, ${col})</strong></div>
      <div class="summary-line">Payoffs: <strong>(${matrix[row][col][0]}, ${matrix[row][col][1]})</strong></div>
    `;
    return;
  }

  summary.innerHTML = `
    <div class="summary-line">Pure NE: <strong>${result.ne.length}</strong></div>
    <div class="summary-line">Positions: <strong>${positions}</strong></div>
    <div class="summary-line">Best NE welfare: <strong>${Math.max(...result.props.ne_welfare)}</strong></div>
  `;
}

function renderClassification(result) {
  const label = result.label;
  const badge = document.querySelector("#game-type-badge");
  const desc = document.querySelector("#game-type-description");
  const color = GAME_TYPE_COLORS[label] || "#7A7570";

  badge.textContent = label;
  badge.style.color = color;
  badge.style.borderColor = color;
  desc.textContent = GAME_TYPE_DESCRIPTIONS[label] || "";
}

function renderProperties(props, equilibria) {
  const rows = [
    propertyRow("P1 dominant strategy", boolBadge(props.p1_has_dominant), "weakly best in every column"),
    propertyRow("P2 dominant strategy", boolBadge(props.p2_has_dominant), "weakly best in every row"),
    propertyRow("Both dominant", boolBadge(props.both_dominant), ""),
    propertyRow("Zero-sum", boolBadge(props.is_zero_sum), "payoff sum constant across cells"),
    propertyRow("Symmetric", boolBadge(props.is_symmetric), "payoff swap across diagonal"),
    propertyRow("Pure NE count", String(props.ne_count), equilibria.length ? formatPositions(equilibria) : "none"),
    propertyRow("Any NE Pareto-dominated", boolBadge(props.has_pareto_dom_ne), ""),
    propertyRow("All NE Pareto-efficient", boolBadge(props.all_ne_pareto_eff), ""),
    propertyRow("Max social welfare", String(props.max_welfare), "best p1 + p2"),
    propertyRow("NE welfare", props.ne_welfare.length ? props.ne_welfare.join(", ") : "-", ""),
    propertyRow("Welfare loss", String(props.welfare_loss), "max welfare minus best NE welfare"),
  ];

  if (props.ne_count > 0) {
    rows.push(
      propertyRow("NE payoffs P1", props.ne_p1_payoffs.join(", "), "per equilibrium"),
      propertyRow("NE payoffs P2", props.ne_p2_payoffs.join(", "), "per equilibrium"),
      propertyRow("Payoff diff (P1-P2)", props.ne_payoff_diffs.join(", "), "per equilibrium"),
      propertyRow("Any NE equal payoffs", boolBadge(props.ne_has_equal_payoffs), ""),
      propertyRow("Mean abs diff at NE", props.ne_mean_abs_diff === null ? "-" : formatNumber(props.ne_mean_abs_diff), "")
    );
  }

  if (props.ne_count === 0) {
    if (props.mixed_exists) {
      rows.push(
        propertyRow("Mixed P1 plays Row 0", `p = ${formatNumber(props.mixed_p)}`, ""),
        propertyRow("Mixed P2 plays Col 0", `q = ${formatNumber(props.mixed_q)}`, ""),
        propertyRow(
          "Mixed expected payoffs",
          `(${formatNumber(props.mixed_payoff_p1)}, ${formatNumber(props.mixed_payoff_p2)})`,
          ""
        )
      );
    } else {
      rows.push(propertyRow("Mixed strategy", "degenerate", "denominator zero"));
    }
  }

  document.querySelector("#properties-body").innerHTML = rows.join("");
}

function propertyRow(label, value, note) {
  return `
    <tr>
      <td>${label}</td>
      <td>${value}</td>
      <td>${note}</td>
    </tr>
  `;
}

function boolBadge(value) {
  return `<span class="${value ? "bool-yes" : "bool-no"}">${value ? "YES" : "NO"}</span>`;
}

function formatPositions(positions) {
  return positions.map(([row, col]) => `(${row}, ${col})`).join(", ");
}

function formatNumber(value) {
  if (Number.isInteger(value)) {
    return String(value);
  }
  return Number(value).toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

init();
