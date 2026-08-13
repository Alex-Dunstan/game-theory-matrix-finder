import assert from "node:assert/strict";

import {
  buildMatrix,
  classifyFull,
  classifyProperties,
  computeMixedStrategy2x2,
  findNashEquilibria,
  GAME_TYPE_DESCRIPTIONS,
} from "./classifier.mjs";

function classify(payoffs) {
  return classifyFull(buildMatrix(payoffs));
}

function approx(actual, expected, tolerance = 1e-5) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
}

assert.equal(classify([3, 3, 0, 5, 5, 0, 1, 1]).label, "Prisoner's Dilemma");
assert.equal(classify([2, 2, 0, 0, 0, 0, 2, 2]).label, "Coordination");
assert.equal(classify([1, -1, -1, 1, -1, 1, 1, -1]).label, "Zero-Sum");
assert.equal(classify([3, 0, 0, 4, 1, 3, 4, 1]).label, "No Equilibrium");
assert.equal(classify([3, 2, 0, 0, 0, 0, 2, 3]).label, "Battle of the Sexes");
assert.equal(classify([4, 4, 1, 3, 3, 1, 2, 2]).label, "Stag Hunt");
assert.equal(classify([3, 3, 1, 4, 4, 1, 0, 0]).label, "Chicken");

for (const label of [
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
]) {
  assert.ok(label in GAME_TYPE_DESCRIPTIONS, `missing description for ${label}`);
}

const matchingPennies = buildMatrix([1, -1, -1, 1, -1, 1, 1, -1]);
const mixed = computeMixedStrategy2x2(matchingPennies);
assert.equal(mixed.mixed_exists, true);
approx(mixed.mixed_p, 0.5);
approx(mixed.mixed_q, 0.5);

const noPureMatrix = buildMatrix([3, 0, 0, 4, 1, 3, 4, 1]);
const noPureNe = findNashEquilibria(noPureMatrix);
assert.deepEqual(noPureNe, []);
const props = classifyProperties(noPureMatrix, noPureNe);
assert.equal(props.mixed_exists, true);
approx(props.mixed_p, 1 / 3);
approx(props.mixed_q, 2 / 3);

console.log("static classifier tests passed");
