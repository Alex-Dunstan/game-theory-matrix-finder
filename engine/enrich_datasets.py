"""
enrich_datasets.py
──────────────────
Reads each CSV in datasets/, computes game classification for every row,
and writes an enriched version to datasets/enriched/.

New columns added:
    p1_has_dominant, p2_has_dominant, both_dominant,
    is_zero_sum, is_symmetric,
    has_pareto_dom_ne, all_ne_pareto_eff,
    max_welfare, ne_welfare, welfare_loss, game_type

Cache-aware: skips enriched files that already exist.
Uses batched numpy operations for dominated-strategy detection so it stays
reasonably fast even on the 214M-row 0–10 dataset.

Usage
─────
    python enrich_datasets.py                # enrich all datasets/
    python enrich_datasets.py --dry-run      # show what would be processed
"""

import argparse
import ast
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.analysis import classify_properties, classify_game_type
from src.matrix_permutations import find_nash_equilibria

DATASETS_DIR = Path(__file__).parent / "datasets"
ENRICHED_DIR = DATASETS_DIR / "enriched"

CLASSIFICATION_COLS = [
    "p1_has_dominant", "p2_has_dominant", "both_dominant",
    "is_zero_sum", "is_symmetric",
    "has_pareto_dom_ne", "all_ne_pareto_eff",
    "max_welfare", "ne_welfare", "welfare_loss",
    # Feature 1 — mixed-strategy equilibrium (2×2, 0-pure-NE matrices only)
    "mixed_exists", "mixed_p", "mixed_q", "mixed_payoff_p1", "mixed_payoff_p2",
    # Feature 2 — payoff asymmetry at Nash equilibria
    "ne_p1_payoffs", "ne_p2_payoffs", "ne_payoff_diffs",
    "ne_has_equal_payoffs", "ne_mean_abs_diff",
    "game_type",
]

PROGRESS_INTERVAL = 500_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Old-format column names (original matrices.csv) → flat index in row-major (r,c,p) order
_OLD_FORMAT_COLS = [
    "top_left_1",    "top_left_2",
    "top_right_1",   "top_right_2",
    "bottom_left_1", "bottom_left_2",
    "bottom_right_1","bottom_right_2",
]


def _infer_shape(fieldnames: list[str]) -> tuple[int, int]:
    """
    Infer (rows, cols) from CSV column names.

    Supports two formats:
      • New format  — r0c0_p1, r1c2_p2, …
      • Old format  — top_left_1, top_right_2, bottom_left_1, …  (always 2×2)
    Falls back to (2, 2) if neither pattern matches.
    """
    max_r = max_c = 0
    found_new = False
    for name in fieldnames:
        if name.startswith("r") and "_p" in name:
            try:
                rc = name.split("_")[0]         # e.g. "r1c2"
                r = int(rc[1:rc.index("c")])
                c = int(rc[rc.index("c") + 1:])
                max_r = max(max_r, r)
                max_c = max(max_c, c)
                found_new = True
            except (ValueError, IndexError):
                pass
    if found_new:
        return (max_r + 1, max_c + 1)

    # Fall back: check for old-format columns → always a 2×2 matrix
    if any(c in fieldnames for c in _OLD_FORMAT_COLS):
        return (2, 2)

    return (2, 2)   # last-resort default


def _parse_ne_positions(ne_str: str) -> list[tuple[int, int]]:
    """Parse the equilibrium_positions column back to a list of (r,c) tuples."""
    if not ne_str or ne_str.strip() in ("None", "[]", ""):
        return []
    try:
        return [tuple(p) for p in ast.literal_eval(ne_str)]
    except Exception:
        return []


def _row_to_matrix(row_values: list[int], rows: int, cols: int) -> np.ndarray:
    """Reconstruct a (rows, cols, 2) numpy array from flat payoff values."""
    arr = np.array(row_values, dtype=np.int32)
    return arr.reshape(rows, cols, 2)


def human_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


# ---------------------------------------------------------------------------
# Core enrichment
# ---------------------------------------------------------------------------

def enrich_file(src: Path, dst: Path, dry_run: bool = False) -> None:
    """Read src CSV, add classification columns, write to dst."""
    if dst.exists():
        print(f"  ✓ Already enriched — skipping (delete to redo)")
        return

    if dry_run:
        print(f"  [dry-run] Would enrich → {dst.name}")
        return

    ENRICHED_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    count = 0

    with open(src, newline="") as fin, open(dst, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        assert reader.fieldnames, f"No columns in {src}"

        rows_dim, cols_dim = _infer_shape(reader.fieldnames)

        # Determine ordered payoff columns (handles both column-name formats)
        if any(c in reader.fieldnames for c in _OLD_FORMAT_COLS):
            # Old format: fixed cell-name ordering matches row-major (r,c,p) layout
            payoff_cols = [c for c in _OLD_FORMAT_COLS if c in reader.fieldnames]
        else:
            # New format: r0c0_p1, r0c0_p2, … — already sorted row-major
            payoff_cols = sorted(
                [f for f in reader.fieldnames if f.startswith("r") and "_p" in f],
                key=lambda n: (
                    int(n.split("_")[0][1:n.split("_")[0].index("c")]),   # row index
                    int(n.split("_")[0][n.split("_")[0].index("c") + 1:]), # col index
                    int(n.split("_p")[1]),                                  # player index
                ),
            )

        writer = csv.DictWriter(
            fout,
            fieldnames=list(reader.fieldnames) + CLASSIFICATION_COLS,
        )
        writer.writeheader()

        for csv_row in reader:
            # Reconstruct matrix from flat payoff values
            flat = [int(csv_row[c]) for c in payoff_cols]
            matrix = _row_to_matrix(flat, rows_dim, cols_dim)

            # Recover NE positions from CSV (avoids recomputing for large files)
            ne_positions = _parse_ne_positions(csv_row.get("equilibrium_positions", ""))

            # Compute classification
            props = classify_properties(matrix, ne_positions)
            label = classify_game_type(props, ne_positions)

            out_row = dict(csv_row)
            out_row.update({
                "p1_has_dominant":    props["p1_has_dominant"],
                "p2_has_dominant":    props["p2_has_dominant"],
                "both_dominant":      props["both_dominant"],
                "is_zero_sum":        props["is_zero_sum"],
                "is_symmetric":       props["is_symmetric"],
                "has_pareto_dom_ne":  props["has_pareto_dom_ne"],
                "all_ne_pareto_eff":  props["all_ne_pareto_eff"],
                "max_welfare":        props["max_welfare"],
                "ne_welfare":         str(props["ne_welfare"]),
                "welfare_loss":       props["welfare_loss"],
                # Mixed strategy (float or "" for None — pandas reads as NaN)
                "mixed_exists":       props["mixed_exists"],
                "mixed_p":            props["mixed_p"]         if props["mixed_p"]         is not None else "",
                "mixed_q":            props["mixed_q"]         if props["mixed_q"]         is not None else "",
                "mixed_payoff_p1":    props["mixed_payoff_p1"] if props["mixed_payoff_p1"] is not None else "",
                "mixed_payoff_p2":    props["mixed_payoff_p2"] if props["mixed_payoff_p2"] is not None else "",
                # Payoff asymmetry (lists stored as Python repr, consistent with ne_welfare)
                "ne_p1_payoffs":      str(props["ne_p1_payoffs"]),
                "ne_p2_payoffs":      str(props["ne_p2_payoffs"]),
                "ne_payoff_diffs":    str(props["ne_payoff_diffs"]),
                "ne_has_equal_payoffs": props["ne_has_equal_payoffs"],
                "ne_mean_abs_diff":   props["ne_mean_abs_diff"] if props["ne_mean_abs_diff"] is not None else "",
                "game_type":          label,
            })
            writer.writerow(out_row)

            count += 1
            if count % PROGRESS_INTERVAL == 0:
                elapsed = time.time() - t0
                rate = count / elapsed
                print(f"  {count:,} rows  ({rate/1000:.0f}K/s)…")

    elapsed = time.time() - t0
    print(f"  ✓ Done — {count:,} rows in {human_duration(elapsed)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich game theory datasets with classification columns.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    csv_files = sorted(
        f for f in DATASETS_DIR.glob("matrices_*.csv")
        if f.name != "range_comparison.csv"
    )

    if not csv_files:
        print("No datasets found in datasets/. Run generate_datasets.py first.")
        return

    print(f"\nGame Theory Dataset Enricher")
    print(f"Source : {DATASETS_DIR}")
    print(f"Output : {ENRICHED_DIR}")
    if args.dry_run:
        print(f"Mode   : DRY RUN\n")

    for src in csv_files:
        dst = ENRICHED_DIR / src.name
        print(f"\n{'─'*60}")
        print(f"  {src.name}  ({src.stat().st_size / 1e6:.1f} MB)")
        enrich_file(src, dst, dry_run=args.dry_run)

    print(f"\n{'═'*60}")
    print("  All done.\n")


if __name__ == "__main__":
    main()
