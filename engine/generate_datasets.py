"""
generate_datasets.py
────────────────────
Exhaustively generates payoff-matrix datasets for multiple integer ranges
and saves each as a CSV in the datasets/ folder.

Features
────────
  • Skips a range if the output CSV already exists (cache hit).
  • Streams rows directly to disk — RAM usage stays constant no matter how
    large the dataset gets (the 0-10 run with ~214M matrices will just take a
    while — grab a coffee).
  • Prints live progress every 1 million rows.
  • After all ranges are done, writes a summary comparison CSV.

Usage
─────
  python generate_datasets.py                 # uses the defaults below
  python generate_datasets.py --dry-run       # show what would be generated

Configuration
─────────────
  Edit RANGES and MATRIX_SIZE below, then re-run.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Make sure src/ is importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

from src.matrix_permutations import (
    generate_dual_matrices_iter,
    save_dual_matrices_iter_to_csv,
    save_batched_to_csv,
    _build_headers,
)

BATCH_SIZE = 50_000   # matrices per numpy batch for vectorised NE detection

# ---------------------------------------------------------------------------
# ✏️  Configure your runs here
# ---------------------------------------------------------------------------

RANGES: list[tuple[int, int]] = [
    (0, 2),   #         6,561 matrices  — instant
    (0, 3),   #        65,536 matrices  — instant
    (0, 5),   #     1,679,616 matrices  — a few seconds   (likely cached already)
    (0, 10),  #   214,358,881 matrices  — will run for a while; progress printed
]

MATRIX_SIZE: tuple[int, int] = (2, 2)   # (rows, cols) — change for n×m

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATASETS_DIR = Path(__file__).parent / "datasets"


def dataset_path(min_val: int, max_val: int, rows: int, cols: int) -> Path:
    return DATASETS_DIR / f"matrices_{rows}x{cols}_{min_val}_to_{max_val}.csv"


def count_combinations(rows: int, cols: int, min_val: int, max_val: int) -> int:
    n_values = max_val - min_val + 1
    pairs_per_cell = n_values ** 2
    return pairs_per_cell ** (rows * cols)


def human_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def human_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

def generate_range(
    min_val: int,
    max_val: int,
    rows: int,
    cols: int,
    dry_run: bool = False,
) -> dict:
    """
    Generate (or skip) one range.  Returns a summary dict for the comparison CSV.
    """
    out_path = dataset_path(min_val, max_val, rows, cols)
    total = count_combinations(rows, cols, min_val, max_val)

    print(f"\n{'─'*60}")
    print(f"  Range {min_val}–{max_val}  |  {rows}×{cols} matrix  |  {human_count(total)} matrices")
    print(f"  Output: {out_path.name}")

    if out_path.exists():
        print(f"  ✓ Already exists — skipping (delete file to regenerate)")
        # Still read summary stats so the comparison CSV is accurate
        return _read_summary(out_path, min_val, max_val, rows, cols, total)

    if dry_run:
        print(f"  [dry-run] Would generate {human_count(total)} matrices")
        return {"range": f"{min_val}-{max_val}", "total": total,
                "pct_0_ne": "?", "pct_1_ne": "?", "pct_2+_ne": "?", "status": "skipped (dry-run)"}

    DATASETS_DIR.mkdir(exist_ok=True)
    print(f"  Generating… (batched vectorised, batch={BATCH_SIZE:,}, progress every 1M rows)")
    t0 = time.time()

    count = save_batched_to_csv(
        rows=rows,
        cols=cols,
        min_val=min_val,
        max_val=max_val,
        filename=str(out_path),
        batch_size=BATCH_SIZE,
        progress_interval=1_000_000,
    )

    elapsed = time.time() - t0
    rate = count / elapsed if elapsed > 0 else 0
    print(f"  ✓ Done — {human_count(count)} matrices in {human_duration(elapsed)}"
          f"  ({human_count(int(rate))}/s)")

    return _read_summary(out_path, min_val, max_val, rows, cols, total)


def _read_summary(
    path: Path,
    min_val: int,
    max_val: int,
    rows: int,
    cols: int,
    total: int,
) -> dict:
    """Read a completed CSV and return NE distribution statistics."""
    ne_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    other = 0

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row.get("num_equilibria", 0))
            if n in ne_counts:
                ne_counts[n] += 1
            else:
                other += 1

    grand = sum(ne_counts.values()) + other or 1  # avoid div/0

    def pct(n):
        return f"{100 * ne_counts.get(n, 0) / grand:.2f}%"

    return {
        "range":     f"{min_val}–{max_val}",
        "matrix":    f"{rows}×{cols}",
        "total":     grand,
        "0_NE":      f"{ne_counts[0]:,} ({pct(0)})",
        "1_NE":      f"{ne_counts[1]:,} ({pct(1)})",
        "2_NE":      f"{ne_counts[2]:,} ({pct(2)})",
        "3_NE":      f"{ne_counts[3]:,} ({pct(3)})",
        "4_NE":      f"{ne_counts[4]:,} ({pct(4)})",
        "pct_solved":f"{100 * (grand - ne_counts[0]) / grand:.2f}%",
    }


def write_comparison(summaries: list[dict]) -> None:
    """Write all range summaries to a single comparison CSV."""
    if not summaries:
        return
    out = DATASETS_DIR / "range_comparison.csv"
    keys = list(summaries[0].keys())
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\n  📊 Comparison summary written → {out.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate game theory matrix datasets.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without writing any files.")
    args = parser.parse_args()

    rows, cols = MATRIX_SIZE
    print(f"\nGame Theory Matrix Dataset Generator")
    print(f"Matrix size : {rows}×{cols}")
    print(f"Ranges      : {RANGES}")
    if args.dry_run:
        print(f"Mode        : DRY RUN (no files written)\n")

    summaries = []
    for (min_val, max_val) in RANGES:
        summary = generate_range(min_val, max_val, rows, cols, dry_run=args.dry_run)
        summaries.append(summary)

    if not args.dry_run:
        DATASETS_DIR.mkdir(exist_ok=True)
        write_comparison(summaries)

    print(f"\n{'═'*60}")
    print(f"  All done.\n")


if __name__ == "__main__":
    main()
