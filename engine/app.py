"""
app.py — Game Theory Matrix Finder  ·  Gradio web viewer
─────────────────────────────────────────────────────────
Run locally:
    python app.py
    → Open http://localhost:7860

HuggingFace Spaces:
    Push this repo as-is.  No code changes needed — Spaces picks up app.py
    automatically.  Add a requirements.txt listing gradio, pandas, matplotlib.

Tabs
────
  01 / MATRIX EXPLORER   — enter payoffs, see Nash equilibria computed live
  02 / DATASET BROWSER   — browse any cached CSV with filters
  03 / DISTRIBUTIONS     — NE count bar chart + Solved/Unsolved donut
  04 / CROSS-RANGE       — compare NE distributions across all available ranges
  05 / CLASSIFICATION    — single-matrix classifier: type label + properties
  06 / GAME TYPE ANALYSIS — aggregate charts across all enriched datasets
"""

import sys
import functools
from io import BytesIO
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")           # headless — must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make src/ importable when running from any working directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from src.matrix_permutations import find_nash_equilibria
from src.analysis import classify_full, GAME_TYPE_DESCRIPTIONS
from src.theme import (
    MPLSTYLE, GRADIO_CSS,
    ACCENT, BG_PAGE, BG_SURFACE, BG_ALT,
    BORDER_STRONG, TEXT_PRI, TEXT_MUT, TEXT_DIM,
    PALETTE,
)

# Apply matplotlib theme globally
plt.rcParams.update(MPLSTYLE)

DATASETS_DIR  = Path(__file__).parent / "datasets"
ENRICHED_DIR  = DATASETS_DIR / "enriched"
PAGE_SIZE = 500   # rows shown per page in the dataset browser

# Ordered game-type colours (same across all charts)
GAME_TYPE_ORDER = [
    "Prisoner's Dilemma",
    "Harmony",
    "Deadlock",
    "Coordination",
    "Zero-Sum",
    "Dominant (P1 only)",
    "Dominant (P2 only)",
    "No Equilibrium",
    "Other",
]
GAME_TYPE_COLORS = {
    "Prisoner's Dilemma": "#E8610A",   # accent orange — iconic social dilemma
    "Harmony":            "#4CAF82",   # green — all good
    "Deadlock":           "#C0392B",   # red — stuck
    "Coordination":       "#3A9BD5",   # blue — align!
    "Zero-Sum":           "#9B59B6",   # purple — adversarial
    "Dominant (P1 only)": "#F39C12",   # amber
    "Dominant (P2 only)": "#E67E22",   # orange-amber
    "No Equilibrium":     "#7A7570",   # muted grey
    "Other":              "#4A4540",   # dim grey
}


# ============================================================
# Dataset cache
# ============================================================

def list_datasets() -> list[str]:
    """Return sorted list of dataset CSV filenames (not full paths)."""
    if not DATASETS_DIR.exists():
        return []
    return sorted(p.name for p in DATASETS_DIR.glob("*.csv")
                  if p.name != "range_comparison.csv")


@functools.lru_cache(maxsize=8)
def load_dataset(filename: str) -> pd.DataFrame:
    """Load a dataset CSV into a DataFrame — cached after first read."""
    path = DATASETS_DIR / filename
    print(f"[cache] Loading {filename} …")
    df = pd.read_csv(path)
    print(f"[cache] Loaded {len(df):,} rows from {filename}")
    return df


def dataset_label(filename: str) -> str:
    """Turn 'matrices_2x2_0_to_5.csv' → '2×2 · range 0–5 (1,679,616 rows)'."""
    try:
        df = load_dataset(filename)
        stem = filename.replace("matrices_", "").replace(".csv", "")
        parts = stem.split("_")          # ['2x2', '0', 'to', '5']
        size  = parts[0].replace("x", "×")
        lo, hi = parts[1], parts[3]
        return f"{size} · range {lo}–{hi}  ({len(df):,} rows)"
    except Exception:
        return filename


def list_enriched_datasets() -> list[str]:
    """Return sorted list of enriched CSV filenames."""
    if not ENRICHED_DIR.exists():
        return []
    return sorted(p.name for p in ENRICHED_DIR.glob("matrices_*.csv"))


# Columns needed for single-matrix classification display (Tab 05)
_ENRICH_FULL_COLS = None   # None = all columns

# Columns needed for aggregate charts (Tab 06) — avoids loading 21-col payoff data
_ENRICH_CHART_COLS = [
    "game_type", "num_equilibria", "all_ne_pareto_eff", "welfare_loss",
    # Feature 1 — mixed strategy
    "mixed_exists", "mixed_p", "mixed_q", "mixed_payoff_p1", "mixed_payoff_p2",
    # Feature 2 — payoff asymmetry
    "ne_has_equal_payoffs", "ne_mean_abs_diff",
]

# File-size threshold: datasets larger than this load chart-only columns
_LARGE_FILE_BYTES = 500 * 1024 * 1024   # 500 MB


@functools.lru_cache(maxsize=4)
def load_enriched_dataset(filename: str) -> pd.DataFrame:
    """
    Load an enriched dataset CSV — cached after first read.
    For files > 500 MB only the columns needed for charts are loaded,
    keeping RAM usage manageable even for the 21 GB 0-10 dataset.
    """
    path = ENRICHED_DIR / filename
    size = path.stat().st_size
    cols = _ENRICH_CHART_COLS if size > _LARGE_FILE_BYTES else _ENRICH_FULL_COLS
    print(f"[cache] Loading enriched {filename}  "
          f"({'chart cols only' if cols else 'all cols'}, "
          f"{size/1e9:.1f} GB on disk)…")
    df = pd.read_csv(path, usecols=cols)
    print(f"[cache] Loaded {len(df):,} rows, {df.memory_usage(deep=True).sum()/1e6:.0f} MB RAM")
    return df


# ============================================================
# Tab 5 — Classification (backend helpers defined first so Tab 1 can reuse)
# ============================================================

def classify_matrix(
    r0c0_p1, r0c0_p2,
    r0c1_p1, r0c1_p2,
    r1c0_p1, r1c0_p2,
    r1c1_p1, r1c1_p2,
):
    """
    Return (properties_html, game_type_html, description_html) for Tab 05.
    Reuses the same 8-input signature as compute_nash for easy preset sharing.
    """
    vals = [int(v) for v in (r0c0_p1, r0c0_p2, r0c1_p1, r0c1_p2,
                              r1c0_p1, r1c0_p2, r1c1_p1, r1c1_p2)]
    matrix = np.array([
        [[vals[0], vals[1]], [vals[2], vals[3]]],
        [[vals[4], vals[5]], [vals[6], vals[7]]],
    ], dtype=np.int32)

    ne = find_nash_equilibria(matrix)
    props, label = classify_full(matrix, ne)

    # ── Properties card ────────────────────────────────────────────────────
    def _bool_badge(v):
        color  = ACCENT if v else TEXT_DIM
        symbol = "✓" if v else "✗"
        return (f"<span style='color:{color};font-weight:600;"
                f"font-family:IBM Plex Mono,monospace'>{symbol}</span>")

    def _row(name, val_html, note=""):
        return (
            f"<tr>"
            f"<td style='padding:6px 14px;color:{TEXT_MUT};font-size:0.82rem;"
            f"font-family:IBM Plex Mono,monospace;border-bottom:1px solid {BORDER_STRONG}'>"
            f"{name}</td>"
            f"<td style='padding:6px 14px;text-align:center;"
            f"border-bottom:1px solid {BORDER_STRONG}'>{val_html}</td>"
            f"<td style='padding:6px 14px;color:{TEXT_DIM};font-size:0.78rem;"
            f"border-bottom:1px solid {BORDER_STRONG}'>{note}</td>"
            f"</tr>"
        )

    ne_str   = str(ne) if ne else "none"
    nw_str   = str(props['ne_welfare']) if props['ne_welfare'] else "—"

    # Build rows as a list so conditional rows can be appended cleanly
    prop_rows = [
        _row("P1 has dominant strategy", _bool_badge(props["p1_has_dominant"]),
             "a row weakly best in every column"),
        _row("P2 has dominant strategy", _bool_badge(props["p2_has_dominant"]),
             "a column weakly best in every row"),
        _row("Both dominant",            _bool_badge(props["both_dominant"])),
        _row("Zero-sum",                 _bool_badge(props["is_zero_sum"]),
             "p1+p2 constant across all cells"),
        _row("Symmetric",                _bool_badge(props["is_symmetric"]),
             "matrix[r,c,0] == matrix[c,r,1]"),
        _row("NE count",
             f"<span style='color:{TEXT_PRI};font-weight:600;"
             f"font-family:IBM Plex Mono,monospace'>{props['ne_count']}</span>",
             ne_str),
        _row("Any NE Pareto-dominated",  _bool_badge(props["has_pareto_dom_ne"]),
             "some other cell beats the NE for both players"),
        _row("All NE Pareto-efficient",  _bool_badge(props["all_ne_pareto_eff"])),
        _row("Max social welfare",
             f"<span style='color:{TEXT_PRI};font-family:IBM Plex Mono,monospace'>"
             f"{props['max_welfare']}</span>",
             "best possible p1+p2"),
        _row("NE welfare",
             f"<span style='color:{TEXT_PRI};font-family:IBM Plex Mono,monospace'>"
             f"{nw_str}</span>"),
        _row("Welfare loss",
             f"<span style='color:{'#C0392B' if props['welfare_loss'] > 0 else '#4CAF82'};"
             f"font-weight:600;font-family:IBM Plex Mono,monospace'>"
             f"{props['welfare_loss']}</span>",
             "max_welfare − best NE welfare"),
    ]

    # ── Payoff asymmetry rows (only when there are pure NE) ─────────────────
    if props["ne_count"] > 0:
        mean_diff_str = (f"{props['ne_mean_abs_diff']:.3f}"
                         if props["ne_mean_abs_diff"] is not None else "—")
        prop_rows += [
            _row("NE payoffs — P1",
                 f"<span style='color:{TEXT_PRI};font-family:IBM Plex Mono,monospace'>"
                 f"{props['ne_p1_payoffs']}</span>",
                 "per equilibrium"),
            _row("NE payoffs — P2",
                 f"<span style='color:{TEXT_PRI};font-family:IBM Plex Mono,monospace'>"
                 f"{props['ne_p2_payoffs']}</span>",
                 "per equilibrium"),
            _row("Payoff diff (P1−P2)",
                 f"<span style='color:{TEXT_PRI};font-family:IBM Plex Mono,monospace'>"
                 f"{props['ne_payoff_diffs']}</span>",
                 "per equilibrium"),
            _row("Any NE with equal payoffs", _bool_badge(props["ne_has_equal_payoffs"]),
                 "P1 == P2 at some NE"),
            _row("Mean |P1−P2| at NE",
                 f"<span style='color:{TEXT_PRI};font-family:IBM Plex Mono,monospace'>"
                 f"{mean_diff_str}</span>",
                 "payoff asymmetry"),
        ]

    # ── Mixed-strategy rows (only when there are no pure NE) ────────────────
    if props["ne_count"] == 0:
        if props["mixed_exists"]:
            prop_rows += [
                _row("Mixed strategy — P1 plays Row 0",
                     f"<span style='color:{ACCENT};font-weight:600;"
                     f"font-family:IBM Plex Mono,monospace'>"
                     f"p = {props['mixed_p']:.4f}</span>",
                     "probability ∈ [0, 1]"),
                _row("Mixed strategy — P2 plays Col 0",
                     f"<span style='color:{ACCENT};font-weight:600;"
                     f"font-family:IBM Plex Mono,monospace'>"
                     f"q = {props['mixed_q']:.4f}</span>",
                     "probability ∈ [0, 1]"),
                _row("Expected payoffs at mixed NE",
                     f"<span style='color:{TEXT_PRI};font-family:IBM Plex Mono,monospace'>"
                     f"({props['mixed_payoff_p1']:.3f}, {props['mixed_payoff_p2']:.3f})</span>",
                     "(P1, P2)"),
            ]
        else:
            prop_rows.append(
                _row("Mixed strategy",
                     f"<span style='color:{TEXT_DIM}'>none (degenerate)</span>",
                     "denominator = 0")
            )

    rows_html = "".join(prop_rows)

    props_html = (
        f"<div style='background:{BG_SURFACE};border:1px solid {BORDER_STRONG};"
        f"border-radius:6px;overflow:hidden;margin-top:0.5rem'>"
        f"<table style='width:100%;border-collapse:collapse'>"
        f"{rows_html}"
        f"</table></div>"
    )

    # ── Game type badge ─────────────────────────────────────────────────────
    badge_color = GAME_TYPE_COLORS.get(label, TEXT_MUT)
    game_type_html = (
        f"<div style='text-align:center;padding:1.5rem 1rem'>"
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;"
        f"color:{TEXT_DIM};text-transform:uppercase;letter-spacing:0.12em;"
        f"margin-bottom:0.6rem'>Game Type</div>"
        f"<div style='font-family:Space Grotesk,sans-serif;font-weight:700;"
        f"font-size:2rem;color:{badge_color};letter-spacing:-0.01em'>"
        f"{label}</div>"
        f"</div>"
    )

    # ── Plain-English description ───────────────────────────────────────────
    desc = GAME_TYPE_DESCRIPTIONS.get(label, "")
    desc_html = (
        f"<div style='background:{BG_ALT};border-left:3px solid {badge_color};"
        f"padding:1rem 1.2rem;border-radius:0 6px 6px 0;margin-top:0.5rem;"
        f"font-family:IBM Plex Mono,monospace;font-size:0.84rem;"
        f"color:{TEXT_MUT};line-height:1.6'>{desc}</div>"
    ) if desc else ""

    return props_html, game_type_html, desc_html


def plot_game_type_distribution(enriched_filename: str):
    """
    Return a matplotlib figure with:
      • Left  — horizontal bar chart of game type counts
      • Right — pie chart of game type shares
    Loads from datasets/enriched/.
    """
    if not enriched_filename:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5, "No enriched dataset available.\nRun  python enrich_datasets.py  first.",
                ha="center", va="center", color=TEXT_MUT, fontsize=10,
                fontfamily="monospace")
        ax.set_axis_off()
        return fig

    try:
        df = load_enriched_dataset(enriched_filename)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5, f"Could not load {enriched_filename}:\n{e}",
                ha="center", va="center", color=TEXT_MUT, fontsize=9)
        ax.set_axis_off()
        return fig

    counts = df["game_type"].value_counts()
    total  = counts.sum()

    # Reorder so known types appear in canonical order, unknowns appended
    ordered_labels = [t for t in GAME_TYPE_ORDER if t in counts.index]
    ordered_labels += [t for t in counts.index if t not in GAME_TYPE_ORDER]
    ordered_vals   = [counts[t] for t in ordered_labels]
    colors         = [GAME_TYPE_COLORS.get(t, TEXT_DIM) for t in ordered_labels]

    fig, (ax_bar, ax_pie) = plt.subplots(
        1, 2, figsize=(12, max(4, len(ordered_labels) * 0.55)),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    stem = enriched_filename.replace("matrices_","").replace(".csv","").replace("_"," ")
    fig.suptitle(stem, fontsize=10, fontweight="bold", color=TEXT_MUT,
                 fontfamily="monospace", y=1.01)

    # ── Horizontal bar chart ───────────────────────────────────────────────
    y_pos = np.arange(len(ordered_labels))
    bars  = ax_bar.barh(y_pos, ordered_vals, color=colors,
                        edgecolor=BG_PAGE, linewidth=0.8, height=0.65)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(ordered_labels, fontsize=8.5)
    ax_bar.set_xlabel("Matrix count", labelpad=8, fontsize=9)
    ax_bar.set_title("01 / GAME TYPE COUNTS", loc="left",
                     fontfamily="monospace", fontsize=9, color=TEXT_MUT)
    ax_bar.invert_yaxis()

    for bar, val in zip(bars, ordered_vals):
        pct = 100 * val / total
        ax_bar.text(bar.get_width() + total * 0.004,
                    bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%",
                    va="center", fontsize=7.5, color=TEXT_MUT)
    ax_bar.set_xlim(0, max(ordered_vals) * 1.18)

    # ── Pie chart ──────────────────────────────────────────────────────────
    # Group tiny slices (<1%) into "Other" for readability
    display_vals, display_labels, display_colors = [], [], []
    other_val = 0
    for lbl, val, col in zip(ordered_labels, ordered_vals, colors):
        if val / total < 0.01 and lbl != "Other":
            other_val += val
        else:
            display_vals.append(val)
            display_labels.append(lbl)
            display_colors.append(col)
    if other_val:
        display_vals.append(other_val)
        display_labels.append("Other (< 1%)")
        display_colors.append(TEXT_DIM)

    wedges, _ = ax_pie.pie(
        display_vals, colors=display_colors, startangle=140,
        wedgeprops={"edgecolor": BG_PAGE, "linewidth": 1.5},
    )
    ax_pie.legend(
        wedges, [f"{l}  {100*v/total:.1f}%" for l, v in zip(display_labels, display_vals)],
        loc="lower center", bbox_to_anchor=(0.5, -0.25),
        fontsize=7, ncol=2, frameon=False,
    )
    ax_pie.set_title("02 / SHARE BY TYPE", loc="left",
                     fontfamily="monospace", fontsize=9, color=TEXT_MUT)

    fig.tight_layout()
    return fig


# ============================================================
# Tab 6 — Game Type Analysis (aggregate, all enriched datasets)
# ============================================================

def plot_cross_range_game_types():
    """
    Stacked bar: game type proportions across all available enriched datasets.
    Returns a matplotlib figure.
    """
    enriched_files = list_enriched_datasets()
    if not enriched_files:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5,
                "No enriched datasets found.\nRun  python enrich_datasets.py  first.",
                ha="center", va="center", color=TEXT_MUT, fontsize=10,
                fontfamily="monospace")
        ax.set_axis_off()
        return fig

    range_data = []
    for fname in enriched_files:
        try:
            d     = load_enriched_dataset(fname)
            total = len(d)
            stem  = fname.replace("matrices_","").replace(".csv","")
            parts = stem.split("_")
            label = f"{parts[0].replace('x','×')} · {parts[1]}–{parts[3]}" if len(parts) >= 4 else stem
            vc    = d["game_type"].value_counts()
            row   = {"label": label, "total": total}
            for gt in GAME_TYPE_ORDER:
                row[gt] = vc.get(gt, 0) / total * 100
            range_data.append(row)
        except Exception as e:
            print(f"[tab6] skipping {fname}: {e}")

    if not range_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Could not load enriched datasets.",
                ha="center", va="center", color=TEXT_MUT)
        ax.set_axis_off()
        return fig

    labels = [r["label"] for r in range_data]
    x      = np.arange(len(labels))
    width  = 0.55

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.8), 6))
    bottoms = np.zeros(len(labels))

    for gt in GAME_TYPE_ORDER:
        vals = np.array([r.get(gt, 0) for r in range_data])
        bars = ax.bar(x, vals, width, bottom=bottoms,
                      label=gt,
                      color=GAME_TYPE_COLORS.get(gt, TEXT_DIM),
                      edgecolor=BG_PAGE, linewidth=0.5)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 5:
                ax.text(xi, b + v / 2, f"{v:.1f}%",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="600")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("% of matrices", labelpad=8)
    ax.set_ylim(0, 100)
    ax.set_title("01 / GAME TYPE MIX BY RANGE",
                 loc="left", fontfamily="monospace", fontsize=9, color=TEXT_MUT)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.15,
              fancybox=False, edgecolor=BORDER_STRONG)
    fig.tight_layout()
    return fig


def plot_pareto_and_welfare():
    """
    Side-by-side: (left) % Pareto-efficient NE by game type,
                  (right) welfare loss histogram across all enriched datasets.
    Returns a matplotlib figure.
    """
    enriched_files = list_enriched_datasets()
    all_dfs = []
    for fname in enriched_files:
        try:
            d = load_enriched_dataset(fname)
            all_dfs.append(d)
        except Exception:
            pass

    if not all_dfs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No enriched datasets available.",
                ha="center", va="center", color=TEXT_MUT)
        ax.set_axis_off()
        return fig

    combined = pd.concat(all_dfs, ignore_index=True)
    has_ne   = combined[combined["num_equilibria"] > 0]

    fig, (ax_par, ax_wl) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: Pareto efficiency % by game type ─────────────────────────────
    gt_pareto = (
        has_ne.groupby("game_type")["all_ne_pareto_eff"]
              .apply(lambda s: 100 * (s == True).sum() / len(s))
              .reindex(GAME_TYPE_ORDER)
              .dropna()
    )
    y_pos  = np.arange(len(gt_pareto))
    colors = [GAME_TYPE_COLORS.get(t, TEXT_DIM) for t in gt_pareto.index]
    ax_par.barh(y_pos, gt_pareto.values, color=colors,
                edgecolor=BG_PAGE, linewidth=0.8, height=0.65)
    ax_par.set_yticks(y_pos)
    ax_par.set_yticklabels(gt_pareto.index, fontsize=8.5)
    ax_par.set_xlabel("% of matrices where all NE are Pareto-efficient", fontsize=9)
    ax_par.set_title("02 / PARETO EFFICIENCY BY TYPE",
                     loc="left", fontfamily="monospace", fontsize=9, color=TEXT_MUT)
    ax_par.invert_yaxis()
    ax_par.set_xlim(0, 110)
    ax_par.axvline(100, color=TEXT_DIM, linewidth=0.7, linestyle="--")
    for i, v in enumerate(gt_pareto.values):
        ax_par.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=8, color=TEXT_MUT)

    # ── Right: welfare loss histogram ──────────────────────────────────────
    wl = has_ne["welfare_loss"].values
    max_wl = int(wl.max()) if len(wl) else 0
    bins   = np.arange(-0.5, max_wl + 1.5, 1)
    total  = len(wl)
    ax_wl.hist(wl, bins=bins, color=ACCENT, edgecolor=BG_PAGE,
               linewidth=0.8, rwidth=0.85)
    zero_pct = 100 * (wl == 0).sum() / total if total else 0
    ax_wl.set_xlabel("Welfare loss at Nash equilibrium", fontsize=9)
    ax_wl.set_ylabel("Matrix count (all enriched datasets)", fontsize=9)
    ax_wl.set_title("03 / WELFARE LOSS DISTRIBUTION",
                    loc="left", fontfamily="monospace", fontsize=9, color=TEXT_MUT)
    ax_wl.text(0.97, 0.95, f"welfare_loss = 0\n{zero_pct:.1f}% of cases",
               transform=ax_wl.transAxes, ha="right", va="top",
               fontsize=8.5, color=TEXT_MUT, fontfamily="monospace")
    if max_wl <= 20:
        ax_wl.set_xticks(range(max_wl + 1))

    fig.tight_layout()
    return fig


def plot_mixed_strategy_distribution(enriched_filename: str):
    """
    2×2 subplot figure showing mixed-strategy equilibrium distributions
    for the 'No Equilibrium' matrices in the selected enriched dataset.
    """
    def _empty(msg):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                color=TEXT_MUT, fontsize=9, fontfamily="monospace")
        ax.set_axis_off()
        return fig

    if not enriched_filename:
        return _empty("No enriched dataset selected.")

    try:
        path = ENRICHED_DIR / enriched_filename
        df = pd.read_csv(
            path,
            usecols=["mixed_exists", "mixed_p", "mixed_q",
                     "mixed_payoff_p1", "mixed_payoff_p2"],
        )
    except Exception as e:
        return _empty(f"Could not load {enriched_filename}:\n{e}")

    mixed = df[df["mixed_exists"] == True].copy()
    if mixed.empty:
        return _empty("No matrices with a valid mixed-strategy NE found.\n"
                      "(Re-enrich datasets to add mixed_* columns.)")

    mixed["mixed_welfare"] = mixed["mixed_payoff_p1"] + mixed["mixed_payoff_p2"]
    n = len(mixed)
    stem = enriched_filename.replace("matrices_", "").replace(".csv", "").replace("_", " ")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"Mixed-strategy equilibria  ·  {stem}  ·  {n:,} matrices",
        fontsize=9, fontweight="bold", color=TEXT_MUT, fontfamily="monospace",
    )

    # Top-left: histogram of mixed_p
    ax = axes[0, 0]
    ax.hist(mixed["mixed_p"], bins=40, color=ACCENT, edgecolor=BG_PAGE,
            linewidth=0.6, rwidth=0.9)
    ax.set_xlabel("p  (P1 plays Row 0)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("01 / P1 MIXING PROBABILITY", loc="left",
                 fontfamily="monospace", fontsize=9, color=TEXT_MUT)

    # Top-right: histogram of mixed_q
    ax = axes[0, 1]
    ax.hist(mixed["mixed_q"], bins=40, color=PALETTE[1], edgecolor=BG_PAGE,
            linewidth=0.6, rwidth=0.9)
    ax.set_xlabel("q  (P2 plays Col 0)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("02 / P2 MIXING PROBABILITY", loc="left",
                 fontfamily="monospace", fontsize=9, color=TEXT_MUT)

    # Bottom-left: scatter mixed_payoff_p1 vs mixed_payoff_p2
    ax = axes[1, 0]
    ax.scatter(mixed["mixed_payoff_p1"], mixed["mixed_payoff_p2"],
               s=2, alpha=0.15, color=ACCENT, rasterized=True)
    ax.set_xlabel("Expected payoff — P1", fontsize=9)
    ax.set_ylabel("Expected payoff — P2", fontsize=9)
    ax.set_title("03 / EXPECTED PAYOFFS AT MIXED NE", loc="left",
                 fontfamily="monospace", fontsize=9, color=TEXT_MUT)

    # Bottom-right: histogram of total welfare at mixed NE
    ax = axes[1, 1]
    ax.hist(mixed["mixed_welfare"], bins=40, color=PALETTE[2], edgecolor=BG_PAGE,
            linewidth=0.6, rwidth=0.9)
    ax.set_xlabel("P1 + P2  (social welfare at mixed NE)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("04 / SOCIAL WELFARE AT MIXED NE", loc="left",
                 fontfamily="monospace", fontsize=9, color=TEXT_MUT)

    fig.tight_layout()
    return fig


def plot_payoff_asymmetry(enriched_filename: str):
    """
    2×2 subplot figure exploring payoff asymmetry (|P1−P2|) at Nash equilibria.
    """
    def _empty(msg):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                color=TEXT_MUT, fontsize=9, fontfamily="monospace")
        ax.set_axis_off()
        return fig

    if not enriched_filename:
        return _empty("No enriched dataset selected.")

    try:
        path = ENRICHED_DIR / enriched_filename
        df = pd.read_csv(
            path,
            usecols=["game_type", "num_equilibria",
                     "ne_has_equal_payoffs", "ne_mean_abs_diff", "welfare_loss"],
        )
    except Exception as e:
        return _empty(f"Could not load {enriched_filename}:\n{e}")

    has_ne = df[df["num_equilibria"] > 0].copy()
    if has_ne.empty:
        return _empty("No matrices with pure Nash equilibria found.")

    has_ne["ne_mean_abs_diff"] = pd.to_numeric(has_ne["ne_mean_abs_diff"], errors="coerce")
    stem = enriched_filename.replace("matrices_", "").replace(".csv", "").replace("_", " ")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Payoff asymmetry at Nash equilibria  ·  {stem}",
        fontsize=9, fontweight="bold", color=TEXT_MUT, fontfamily="monospace",
    )

    # Top-left: % equal-payoff NE by game type
    ax = axes[0, 0]
    ordered_types = [t for t in GAME_TYPE_ORDER
                     if t in has_ne["game_type"].unique() and t != "No Equilibrium"]
    eq_pcts = []
    for gt in ordered_types:
        sub = has_ne[has_ne["game_type"] == gt]
        pct = 100 * (sub["ne_has_equal_payoffs"] == True).sum() / len(sub)
        eq_pcts.append(pct)
    colors = [GAME_TYPE_COLORS.get(t, TEXT_DIM) for t in ordered_types]
    y_pos = np.arange(len(ordered_types))
    ax.barh(y_pos, eq_pcts, color=colors, edgecolor=BG_PAGE, linewidth=0.8, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_types, fontsize=8)
    ax.set_xlabel("% matrices where any NE has P1 = P2", fontsize=9)
    ax.set_title("01 / EQUAL-PAYOFF NE % BY GAME TYPE", loc="left",
                 fontfamily="monospace", fontsize=9, color=TEXT_MUT)
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    for i, v in enumerate(eq_pcts):
        ax.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=7.5, color=TEXT_MUT)

    # Top-right: % equal-payoff NE by NE count (1–4)
    ax = axes[0, 1]
    ne_counts_present = sorted(has_ne["num_equilibria"].unique())
    eq_by_ne = []
    labels_ne = []
    for n in ne_counts_present:
        sub = has_ne[has_ne["num_equilibria"] == n]
        pct = 100 * (sub["ne_has_equal_payoffs"] == True).sum() / len(sub)
        eq_by_ne.append(pct)
        labels_ne.append(str(n))
    bar_colors = PALETTE[:len(ne_counts_present)]
    ax.bar(labels_ne, eq_by_ne, color=bar_colors, edgecolor=BG_PAGE, linewidth=0.8, width=0.65)
    ax.set_xlabel("Number of pure Nash equilibria", fontsize=9)
    ax.set_ylabel("% with any equal-payoff NE", fontsize=9)
    ax.set_title("02 / EQUAL-PAYOFF NE % BY NE COUNT", loc="left",
                 fontfamily="monospace", fontsize=9, color=TEXT_MUT)
    ax.set_ylim(0, max(eq_by_ne) * 1.2 if eq_by_ne else 1)
    for i, v in enumerate(eq_by_ne):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8, color=TEXT_MUT)

    # Bottom-left: horizontal boxplot of ne_mean_abs_diff by game type
    ax = axes[1, 0]
    box_data = []
    box_labels = []
    box_colors = []
    for gt in reversed(ordered_types):
        sub = has_ne[has_ne["game_type"] == gt]["ne_mean_abs_diff"].dropna()
        if len(sub) > 0:
            box_data.append(sub.values)
            box_labels.append(gt)
            box_colors.append(GAME_TYPE_COLORS.get(gt, TEXT_DIM))
    if box_data:
        bp = ax.boxplot(box_data, vert=False, patch_artist=True,
                        medianprops={"color": TEXT_PRI, "linewidth": 1.5},
                        whiskerprops={"color": TEXT_MUT},
                        capprops={"color": TEXT_MUT},
                        flierprops={"marker": ".", "markersize": 2,
                                    "markerfacecolor": TEXT_DIM, "alpha": 0.3})
        for patch, col in zip(bp["boxes"], box_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
            patch.set_edgecolor(BG_PAGE)
    ax.set_yticks(np.arange(1, len(box_labels) + 1))
    ax.set_yticklabels(box_labels, fontsize=8)
    ax.set_xlabel("Mean |P1 − P2| at NE", fontsize=9)
    ax.set_title("03 / PAYOFF ASYMMETRY DISTRIBUTION BY TYPE", loc="left",
                 fontfamily="monospace", fontsize=9, color=TEXT_MUT)

    # Bottom-right: scatter ne_mean_abs_diff vs welfare_loss
    ax = axes[1, 1]
    plot_df = has_ne[["ne_mean_abs_diff", "welfare_loss"]].dropna()
    ax.scatter(plot_df["welfare_loss"], plot_df["ne_mean_abs_diff"],
               s=1.5, alpha=0.2, color=ACCENT, rasterized=True)
    ax.set_xlabel("Welfare loss at NE", fontsize=9)
    ax.set_ylabel("Mean |P1 − P2| at NE", fontsize=9)
    ax.set_title("04 / ASYMMETRY vs WELFARE LOSS", loc="left",
                 fontfamily="monospace", fontsize=9, color=TEXT_MUT)

    fig.tight_layout()
    return fig


# ============================================================
# Tab 1 — Matrix Explorer
# ============================================================

def _matrix_html(payoffs: list[int], equilibria: list[tuple]) -> str:
    """Render a 2×2 payoff matrix as a themed HTML table."""
    ne_set = set(equilibria)
    cells = [(payoffs[i*2], payoffs[i*2+1]) for i in range(4)]
    positions = [(0,0),(0,1),(1,0),(1,1)]

    rows_html = []
    for r in range(2):
        row_cells = []
        for c in range(2):
            idx = r*2 + c
            p1, p2 = cells[idx]
            is_ne = (r, c) in ne_set
            accent = f"color:{ACCENT};font-weight:600;" if is_ne else f"color:{TEXT_MUT};"
            star   = " ★" if is_ne else ""
            cell = (
                f"<td style='padding:14px 22px;border:1px solid {BORDER_STRONG};"
                f"background:{BG_ALT};text-align:center;{accent}'>"
                f"({p1}, {p2}){star}</td>"
            )
            row_cells.append(cell)
        rows_html.append("<tr>" + "".join(row_cells) + "</tr>")

    header = (
        f"<tr>"
        f"<th style='padding:8px 22px;background:{BG_PAGE};color:{TEXT_DIM};"
        f"font-size:0.75rem;text-transform:uppercase;border:1px solid {BORDER_STRONG};'>"
        f"Col 0</th>"
        f"<th style='padding:8px 22px;background:{BG_PAGE};color:{TEXT_DIM};"
        f"font-size:0.75rem;text-transform:uppercase;border:1px solid {BORDER_STRONG};'>"
        f"Col 1</th>"
        f"</tr>"
    )

    table = (
        f"<table style='border-collapse:collapse;font-family:IBM Plex Mono,monospace;"
        f"font-size:1rem;margin:auto;'>"
        f"{header}{''.join(rows_html)}</table>"
    )
    return table


def compute_nash(
    r0c0_p1, r0c0_p2,
    r0c1_p1, r0c1_p2,
    r1c0_p1, r1c0_p2,
    r1c1_p1, r1c1_p2,
):
    """Gradio event handler: compute NE and return HTML table + description."""
    payoffs = [
        int(r0c0_p1), int(r0c0_p2),
        int(r0c1_p1), int(r0c1_p2),
        int(r1c0_p1), int(r1c0_p2),
        int(r1c1_p1), int(r1c1_p2),
    ]
    matrix = np.array([
        [[payoffs[0], payoffs[1]], [payoffs[2], payoffs[3]]],
        [[payoffs[4], payoffs[5]], [payoffs[6], payoffs[7]]],
    ], dtype=np.int32)

    eq = find_nash_equilibria(matrix)
    table_html = _matrix_html(payoffs, eq)

    if not eq:
        props, _ = classify_full(matrix, [])
        if props["mixed_exists"]:
            p_val = props["mixed_p"]
            q_val = props["mixed_q"]
            ep1   = props["mixed_payoff_p1"]
            ep2   = props["mixed_payoff_p2"]
            desc = (
                f"<p style='color:{TEXT_MUT};font-family:IBM Plex Mono,monospace;'>"
                f"⚠ <strong style='color:{TEXT_PRI}'>No pure-strategy Nash equilibrium.</strong><br>"
                f"Mixed-strategy NE: P1 plays Row&nbsp;0 with probability "
                f"<strong style='color:{ACCENT}'>p&nbsp;=&nbsp;{p_val:.4f}</strong>, "
                f"P2 plays Col&nbsp;0 with probability "
                f"<strong style='color:{ACCENT}'>q&nbsp;=&nbsp;{q_val:.4f}</strong>.<br>"
                f"Expected payoffs: ({ep1:.3f},&nbsp;{ep2:.3f}).</p>"
            )
        else:
            desc = (
                f"<p style='color:{TEXT_MUT};font-family:IBM Plex Mono,monospace;'>"
                f"⚠ <strong style='color:{TEXT_PRI}'>No pure-strategy Nash equilibrium.</strong> "
                f"Neither player can reach a stable outcome without mixed strategies "
                f"(mixed NE is degenerate for this matrix).</p>"
            )
    elif len(eq) == 1:
        r, c = eq[0]
        desc = (
            f"<p style='font-family:IBM Plex Mono,monospace;'>"
            f"✓ <strong style='color:{ACCENT}'>1 Nash equilibrium</strong> "
            f"at position ({r}, {c}) — "
            f"payoffs ({matrix[r,c,0]}, {matrix[r,c,1]}).</p>"
        )
    else:
        pos_str = ", ".join(f"({r},{c})" for r,c in eq)
        desc = (
            f"<p style='font-family:IBM Plex Mono,monospace;'>"
            f"✓ <strong style='color:{ACCENT}'>{len(eq)} Nash equilibria</strong> "
            f"at positions: {pos_str}.</p>"
        )

    full_html = (
        f"<div style='text-align:center;padding:1.5rem;'>"
        f"{table_html}"
        f"<div style='margin-top:1rem;'>{desc}</div>"
        f"</div>"
    )
    return full_html


# Preset examples
PRESETS = {
    "Prisoner's Dilemma": (3, 3, 0, 5, 5, 0, 1, 1),
    "Coordination Game":  (2, 2, 0, 0, 0, 0, 2, 2),
    "Battle of the Sexes":(3, 2, 0, 0, 0, 0, 2, 3),
    "Matching Pennies":   (1,-1,-1, 1,-1, 1, 1,-1),
    "All Equal (4 NE)":   (3, 3, 3, 3, 3, 3, 3, 3),
}

def load_preset(preset_name):
    if preset_name in PRESETS:
        return PRESETS[preset_name]
    return (0,)*8


# ============================================================
# Tab 2 — Dataset Browser
# ============================================================

def browse_dataset(filename: str, ne_filter: str, page: int):
    """Load dataset, apply filters, return a page of rows + summary."""
    if not filename:
        return pd.DataFrame(), "No dataset selected."

    df = load_dataset(filename)
    total = len(df)

    # Apply NE count filter
    if ne_filter != "All":
        n = int(ne_filter)
        df = df[df["num_equilibria"] == n]

    filtered = len(df)
    start = max(0, int(page) - 1) * PAGE_SIZE
    end   = start + PAGE_SIZE
    page_df = df.iloc[start:end].reset_index(drop=True)

    summary = (
        f"**{filename}** — "
        f"{total:,} total rows · "
        f"{filtered:,} matching filter · "
        f"showing rows {start+1}–{min(end, filtered):,}"
    )
    return page_df, summary


def get_dataset_stats(filename: str) -> str:
    """Return a markdown summary of NE distribution for a dataset."""
    if not filename:
        return ""
    df = load_dataset(filename)
    total = len(df)
    if total == 0:
        return "Empty dataset."

    counts = df["num_equilibria"].value_counts().sort_index()
    lines = [f"**{filename}** — {total:,} matrices\n"]
    for ne_count, n in counts.items():
        bar = "█" * int(30 * n / total)
        lines.append(f"`{ne_count} NE` {bar} {n:,} ({100*n/total:.1f}%)")
    solved = total - counts.get(0, 0)
    lines.append(f"\n**Solved:** {solved:,} ({100*solved/total:.1f}%)"
                 f"  |  **Unsolved:** {counts.get(0,0):,} ({100*counts.get(0,0)/total:.1f}%)")
    return "\n\n".join(lines)


# ============================================================
# Tab 3 — Distributions
# ============================================================

def plot_distributions(filename: str):
    """Return a matplotlib figure with NE bar chart + Solved/Unsolved donut."""
    if not filename:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Select a dataset above",
                ha="center", va="center", color=TEXT_MUT, fontsize=12)
        ax.set_axis_off()
        return fig

    df = load_dataset(filename)
    counts = df["num_equilibria"].value_counts().sort_index()
    total  = len(df)

    fig, (ax_bar, ax_donut) = plt.subplots(
        1, 2, figsize=(11, 5),
        gridspec_kw={"width_ratios": [2, 1]}
    )
    fig.suptitle(filename.replace(".csv","").replace("matrices_","").replace("_"," "),
                 fontsize=11, fontweight="bold", color=TEXT_MUT,
                 fontfamily="monospace", y=1.01)

    # ── Bar chart ──────────────────────────────────────────────────────────
    x      = list(counts.index)
    values = list(counts.values)
    bars   = ax_bar.bar(x, values, color=PALETTE[:len(x)], width=0.55,
                        edgecolor=BG_PAGE, linewidth=1.2)

    ax_bar.set_xlabel("Number of Nash Equilibria", labelpad=8)
    ax_bar.set_ylabel("Matrix Count", labelpad=8)
    ax_bar.set_title("01 / NE COUNT DISTRIBUTION", loc="left",
                     fontfamily="monospace", fontsize=9, color=TEXT_MUT)
    ax_bar.set_xticks(x)

    for bar, val in zip(bars, values):
        ax_bar.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + total*0.005,
                    f"{val/total*100:.1f}%",
                    ha="center", va="bottom", fontsize=8, color=TEXT_MUT)

    # ── Donut ──────────────────────────────────────────────────────────────
    unsolved = counts.get(0, 0)
    solved   = total - unsolved
    sizes    = [solved, unsolved]
    colors   = [ACCENT, TEXT_DIM]
    wedges, _ = ax_donut.pie(
        sizes, colors=colors, startangle=90,
        wedgeprops={"width": 0.52, "edgecolor": BG_PAGE, "linewidth": 2}
    )
    ax_donut.text(0, 0, f"{100*solved/total:.0f}%\nSolved",
                  ha="center", va="center", fontsize=11, color=TEXT_PRI,
                  fontweight="bold", fontfamily="monospace")
    legend_patches = [
        mpatches.Patch(color=ACCENT, label=f"Solved  {solved:,}"),
        mpatches.Patch(color=TEXT_DIM, label=f"Unsolved  {unsolved:,}"),
    ]
    ax_donut.legend(handles=legend_patches, loc="lower center",
                    bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax_donut.set_title("02 / SOLVED vs UNSOLVED", loc="left",
                       fontfamily="monospace", fontsize=9, color=TEXT_MUT)

    fig.tight_layout()
    return fig


# ============================================================
# Tab 4 — Cross-range comparison
# ============================================================

def plot_cross_range():
    """Grouped bar chart comparing NE distributions across all available datasets."""
    files = list_datasets()
    if not files:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5,
                "No datasets found.\nRun  python generate_datasets.py  first.",
                ha="center", va="center", color=TEXT_MUT, fontsize=11,
                fontfamily="monospace")
        ax.set_axis_off()
        return fig

    summaries = []
    for fname in files:
        try:
            df    = load_dataset(fname)
            total = len(df)
            entry = {"label": fname.replace("matrices_","").replace(".csv","")
                                     .replace("_"," "), "total": total}
            for ne_n in range(5):
                entry[f"ne_{ne_n}"] = df["num_equilibria"].value_counts().get(ne_n, 0) / total * 100
            summaries.append(entry)
        except Exception as e:
            print(f"[cross-range] skipping {fname}: {e}")

    if not summaries:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Could not load any datasets.",
                ha="center", va="center", color=TEXT_MUT)
        ax.set_axis_off()
        return fig

    labels  = [s["label"] for s in summaries]
    ne_vals = {f"ne_{i}": [s[f"ne_{i}"] for s in summaries] for i in range(5)}

    x      = np.arange(len(labels))
    width  = 0.15
    fig, ax = plt.subplots(figsize=(max(9, len(labels)*3), 5))

    for i, (key, vals) in enumerate(ne_vals.items()):
        bars = ax.bar(x + i*width, vals, width, label=f"{i} NE",
                      color=PALETTE[i % len(PALETTE)],
                      edgecolor=BG_PAGE, linewidth=0.8)

    ax.set_xlabel("Dataset (matrix size · range)", labelpad=8)
    ax.set_ylabel("% of matrices", labelpad=8)
    ax.set_title("04 / CROSS-RANGE NE DISTRIBUTION COMPARISON",
                 loc="left", fontfamily="monospace", fontsize=9, color=TEXT_MUT)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.legend(title="# NE", framealpha=0, title_fontsize=8)
    ax.set_ylim(0, 100)

    fig.tight_layout()
    return fig


# ============================================================
# Build the Gradio interface
# ============================================================

def build_app() -> gr.Blocks:
    datasets = list_datasets()
    default_ds = datasets[0] if datasets else None

    with gr.Blocks(
        css=GRADIO_CSS,
        title="Game Theory Matrix Finder",
        theme=gr.themes.Base(),
    ) as demo:

        gr.HTML("""
        <div style="padding:1.8rem 1rem 0.5rem;border-bottom:1px solid #3D2418;">
          <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;
                      font-size:1.5rem;text-transform:uppercase;
                      letter-spacing:-0.02em;color:#F0EDE8;">
            Game Theory Matrix Finder
          </div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;
                      color:#7A7570;margin-top:0.25rem;">
            Exhaustive 2-player payoff matrix analysis · pure-strategy Nash equilibria
          </div>
        </div>
        """)

        with gr.Tabs():

            # ────────────────────────────────────────────────────────────────
            # TAB 1 — Matrix Explorer
            # ────────────────────────────────────────────────────────────────
            with gr.Tab("01 / MATRIX EXPLORER"):
                gr.Markdown(
                    "> Enter any 2×2 payoff matrix below. Each cell takes two numbers: "
                    "**Player 1's payoff** and **Player 2's payoff**. Hit **Compute** "
                    "to find all pure-strategy Nash equilibria instantly.\n\n"
                    "Or load a classic game from the presets."
                )

                with gr.Row():
                    preset_dd = gr.Dropdown(
                        choices=list(PRESETS.keys()),
                        label="Load a classic game",
                        value=None,
                    )

                gr.HTML("<div style='padding:0.4rem 0;color:#4A4540;font-size:0.75rem;"
                        "font-family:IBM Plex Mono,monospace;'>"
                        "Row labels → Row 0 / Row 1 &nbsp;·&nbsp; "
                        "Column labels → Col 0 / Col 1</div>")

                with gr.Row():
                    with gr.Column():
                        gr.HTML("<div class='section-num'>TOP ROW</div>")
                        with gr.Row():
                            r0c0_p1 = gr.Number(label="(0,0) P1", value=3, precision=0)
                            r0c0_p2 = gr.Number(label="(0,0) P2", value=3, precision=0)
                        with gr.Row():
                            r0c1_p1 = gr.Number(label="(0,1) P1", value=0, precision=0)
                            r0c1_p2 = gr.Number(label="(0,1) P2", value=5, precision=0)

                    with gr.Column():
                        gr.HTML("<div class='section-num'>BOTTOM ROW</div>")
                        with gr.Row():
                            r1c0_p1 = gr.Number(label="(1,0) P1", value=5, precision=0)
                            r1c0_p2 = gr.Number(label="(1,0) P2", value=0, precision=0)
                        with gr.Row():
                            r1c1_p1 = gr.Number(label="(1,1) P1", value=1, precision=0)
                            r1c1_p2 = gr.Number(label="(1,1) P2", value=1, precision=0)

                compute_btn = gr.Button("⚡ Compute Nash Equilibria", variant="primary")
                output_html = gr.HTML()

                all_inputs = [r0c0_p1, r0c0_p2, r0c1_p1, r0c1_p2,
                              r1c0_p1, r1c0_p2, r1c1_p1, r1c1_p2]

                compute_btn.click(fn=compute_nash, inputs=all_inputs, outputs=output_html)

                # Preset loader
                def apply_preset(name):
                    return load_preset(name)

                preset_dd.change(
                    fn=apply_preset,
                    inputs=preset_dd,
                    outputs=all_inputs,
                )

                # Auto-compute on load with default (Prisoner's Dilemma)
                demo.load(fn=compute_nash, inputs=all_inputs, outputs=output_html)

            # ────────────────────────────────────────────────────────────────
            # TAB 2 — Dataset Browser
            # ────────────────────────────────────────────────────────────────
            with gr.Tab("02 / DATASET BROWSER"):
                gr.Markdown(
                    "> Browse any cached dataset directly in the browser. "
                    "Filter by Nash equilibrium count, then page through results.\n\n"
                    f"Datasets are read from `datasets/` — run `generate_datasets.py` to add more."
                )

                with gr.Row():
                    ds_dd = gr.Dropdown(
                        choices=datasets,
                        value=default_ds,
                        label="Dataset",
                        scale=3,
                    )
                    ne_dd = gr.Dropdown(
                        choices=["All", "0", "1", "2", "3", "4"],
                        value="All",
                        label="NE count filter",
                        scale=1,
                    )
                    page_num = gr.Number(
                        label=f"Page (×{PAGE_SIZE} rows)",
                        value=1, precision=0, minimum=1,
                        scale=1,
                    )

                stats_md   = gr.Markdown()
                summary_md = gr.Markdown()
                data_table = gr.Dataframe(
                    interactive=False,
                    wrap=False,
                )

                def refresh_browser(filename, ne_filter, page):
                    df, summary = browse_dataset(filename, ne_filter, page)
                    stats = get_dataset_stats(filename)
                    return df, summary, stats

                browse_btn = gr.Button("Load / Refresh", variant="primary")
                browse_btn.click(
                    fn=refresh_browser,
                    inputs=[ds_dd, ne_dd, page_num],
                    outputs=[data_table, summary_md, stats_md],
                )
                # Load on dataset change
                ds_dd.change(
                    fn=refresh_browser,
                    inputs=[ds_dd, ne_dd, page_num],
                    outputs=[data_table, summary_md, stats_md],
                )
                # Auto-load default on startup
                if default_ds:
                    demo.load(
                        fn=refresh_browser,
                        inputs=[ds_dd, ne_dd, page_num],
                        outputs=[data_table, summary_md, stats_md],
                    )

            # ────────────────────────────────────────────────────────────────
            # TAB 3 — Distributions
            # ────────────────────────────────────────────────────────────────
            with gr.Tab("03 / DISTRIBUTIONS"):
                gr.Markdown(
                    "> Distribution of Nash equilibria counts across all matrices "
                    "in the selected dataset, plus a Solved vs Unsolved breakdown."
                )

                dist_ds_dd = gr.Dropdown(
                    choices=datasets,
                    value=default_ds,
                    label="Dataset",
                )
                dist_plot = gr.Plot()

                dist_ds_dd.change(fn=plot_distributions, inputs=dist_ds_dd, outputs=dist_plot)
                if default_ds:
                    demo.load(fn=lambda: plot_distributions(default_ds), outputs=dist_plot)

            # ────────────────────────────────────────────────────────────────
            # TAB 4 — Cross-range Comparison
            # ────────────────────────────────────────────────────────────────
            with gr.Tab("04 / CROSS-RANGE"):
                gr.Markdown(
                    "> Compares NE distributions across every dataset in `datasets/`. "
                    "Run `generate_datasets.py` to add ranges — they appear here automatically."
                )

                refresh_cross_btn = gr.Button("↺ Refresh comparison", variant="secondary")
                cross_plot = gr.Plot()

                refresh_cross_btn.click(fn=plot_cross_range, outputs=cross_plot)
                demo.load(fn=plot_cross_range, outputs=cross_plot)

            # ────────────────────────────────────────────────────────────────
            # TAB 5 — Classification
            # ────────────────────────────────────────────────────────────────
            with gr.Tab("05 / CLASSIFICATION"):
                gr.Markdown(
                    "> Enter any 2×2 payoff matrix to classify its game type — "
                    "boolean structural properties, named type label, and a plain-English "
                    "explanation.  Scroll down for the game-type distribution across "
                    "enriched datasets.\n\n"
                    "Run `python enrich_datasets.py` first to enable the distribution charts."
                )

                # ── Matrix input (mirrors Tab 01) ───────────────────────────
                gr.HTML(
                    "<div class='section-num' style='margin-bottom:0.5rem'>"
                    "PAYOFF MATRIX INPUT</div>"
                )

                with gr.Row():
                    cl5_preset_dd = gr.Dropdown(
                        choices=list(PRESETS.keys()),
                        label="Load a classic game",
                        value="Prisoner's Dilemma",
                    )

                with gr.Row():
                    with gr.Column():
                        gr.HTML("<div class='section-num'>TOP ROW</div>")
                        with gr.Row():
                            c5_r0c0_p1 = gr.Number(label="(0,0) P1", value=3, precision=0)
                            c5_r0c0_p2 = gr.Number(label="(0,0) P2", value=3, precision=0)
                        with gr.Row():
                            c5_r0c1_p1 = gr.Number(label="(0,1) P1", value=0, precision=0)
                            c5_r0c1_p2 = gr.Number(label="(0,1) P2", value=5, precision=0)
                    with gr.Column():
                        gr.HTML("<div class='section-num'>BOTTOM ROW</div>")
                        with gr.Row():
                            c5_r1c0_p1 = gr.Number(label="(1,0) P1", value=5, precision=0)
                            c5_r1c0_p2 = gr.Number(label="(1,0) P2", value=0, precision=0)
                        with gr.Row():
                            c5_r1c1_p1 = gr.Number(label="(1,1) P1", value=1, precision=0)
                            c5_r1c1_p2 = gr.Number(label="(1,1) P2", value=1, precision=0)

                classify_btn = gr.Button("🔍 Classify Game", variant="primary")

                c5_all_inputs = [
                    c5_r0c0_p1, c5_r0c0_p2, c5_r0c1_p1, c5_r0c1_p2,
                    c5_r1c0_p1, c5_r1c0_p2, c5_r1c1_p1, c5_r1c1_p2,
                ]

                # ── Output area ─────────────────────────────────────────────
                c5_game_type_html = gr.HTML()    # large type label
                c5_desc_html      = gr.HTML()    # plain-English description
                c5_props_html     = gr.HTML()    # properties table

                classify_btn.click(
                    fn=classify_matrix,
                    inputs=c5_all_inputs,
                    outputs=[c5_props_html, c5_game_type_html, c5_desc_html],
                )

                # Preset loader for Tab 05
                def apply_preset_c5(name):
                    return load_preset(name)

                cl5_preset_dd.change(
                    fn=apply_preset_c5,
                    inputs=cl5_preset_dd,
                    outputs=c5_all_inputs,
                )

                # Auto-classify default on load
                demo.load(
                    fn=classify_matrix,
                    inputs=c5_all_inputs,
                    outputs=[c5_props_html, c5_game_type_html, c5_desc_html],
                )

                # ── Game type distribution ───────────────────────────────────
                gr.HTML(
                    f"<div style='margin-top:2rem;padding-top:1.5rem;"
                    f"border-top:1px solid {BORDER_STRONG}'>"
                    f"<div class='section-num'>GAME TYPE DISTRIBUTION</div>"
                    f"<p style='font-family:IBM Plex Mono,monospace;font-size:0.82rem;"
                    f"color:{TEXT_MUT};margin-top:0.4rem'>"
                    f"Distribution of game types across an enriched dataset.</p>"
                    f"</div>"
                )

                enriched_datasets = list_enriched_datasets()
                default_enriched  = enriched_datasets[0] if enriched_datasets else None

                with gr.Row():
                    enriched_dd = gr.Dropdown(
                        choices=enriched_datasets,
                        value=default_enriched,
                        label="Enriched dataset",
                    )
                    refresh_dist_btn = gr.Button("↺ Refresh", variant="secondary", scale=0)

                dist5_plot = gr.Plot()

                def _plot_dist5(fname):
                    return plot_game_type_distribution(fname)

                enriched_dd.change(fn=_plot_dist5, inputs=enriched_dd, outputs=dist5_plot)
                refresh_dist_btn.click(fn=_plot_dist5, inputs=enriched_dd, outputs=dist5_plot)

                if default_enriched:
                    demo.load(
                        fn=lambda: plot_game_type_distribution(default_enriched),
                        outputs=dist5_plot,
                    )

            # ────────────────────────────────────────────────────────────────
            # TAB 6 — Game Type Analysis
            # ────────────────────────────────────────────────────────────────
            with gr.Tab("06 / GAME TYPE ANALYSIS"):
                gr.Markdown(
                    "> Aggregate analysis across all enriched datasets — game type "
                    "mix by payoff range, Pareto efficiency, and welfare loss. "
                    "Run `python enrich_datasets.py` first to populate the charts."
                )

                refresh_gt_btn = gr.Button("↺ Refresh all charts", variant="secondary")

                gr.HTML(
                    f"<div style='margin-bottom:0.5rem'>"
                    f"<div class='section-num'>GAME TYPE MIX BY RANGE</div>"
                    f"<p style='font-family:IBM Plex Mono,monospace;font-size:0.82rem;"
                    f"color:{TEXT_MUT};margin-top:0.3rem'>"
                    f"How the proportion of each game type shifts as the payoff range widens.</p>"
                    f"</div>"
                )
                cross_gt_plot = gr.Plot()

                gr.HTML(
                    f"<div style='margin-top:1.5rem;padding-top:1.2rem;"
                    f"border-top:1px solid {BORDER_STRONG};margin-bottom:0.5rem'>"
                    f"<div class='section-num'>PARETO EFFICIENCY · WELFARE LOSS</div>"
                    f"<p style='font-family:IBM Plex Mono,monospace;font-size:0.82rem;"
                    f"color:{TEXT_MUT};margin-top:0.3rem'>"
                    f"Left: how often rational play reaches a socially optimal outcome, by game type. "
                    f"Right: distribution of welfare left on the table at Nash equilibrium.</p>"
                    f"</div>"
                )
                pareto_plot = gr.Plot()

                # ── Mixed-strategy equilibrium sub-section ───────────────────
                gr.HTML(
                    f"<div style='margin-top:1.5rem;padding-top:1.2rem;"
                    f"border-top:1px solid {BORDER_STRONG};margin-bottom:0.5rem'>"
                    f"<div class='section-num'>MIXED-STRATEGY EQUILIBRIUM ANALYSIS</div>"
                    f"<p style='font-family:IBM Plex Mono,monospace;font-size:0.82rem;"
                    f"color:{TEXT_MUT};margin-top:0.3rem'>"
                    f"For matrices with 0 pure-strategy NE, the unique mixed-strategy "
                    f"equilibrium is computed. Select an enriched dataset to see the "
                    f"distributions of mixing probabilities and expected payoffs.</p>"
                    f"</div>"
                )
                with gr.Row():
                    mixed_enrich_dd = gr.Dropdown(
                        choices=list_enriched_datasets(),
                        value=list_enriched_datasets()[-1] if list_enriched_datasets() else None,
                        label="Enriched dataset",
                    )
                mixed_plot = gr.Plot()
                mixed_enrich_dd.change(
                    fn=plot_mixed_strategy_distribution,
                    inputs=mixed_enrich_dd,
                    outputs=mixed_plot,
                )

                # ── Payoff asymmetry sub-section ─────────────────────────────
                gr.HTML(
                    f"<div style='margin-top:1.5rem;padding-top:1.2rem;"
                    f"border-top:1px solid {BORDER_STRONG};margin-bottom:0.5rem'>"
                    f"<div class='section-num'>PAYOFF ASYMMETRY AT NASH EQUILIBRIA</div>"
                    f"<p style='font-family:IBM Plex Mono,monospace;font-size:0.82rem;"
                    f"color:{TEXT_MUT};margin-top:0.3rem'>"
                    f"How often P1 and P2 receive equal payoffs at the Nash equilibrium, "
                    f"broken down by game type and NE count. Does asymmetry correlate with "
                    f"welfare loss?</p>"
                    f"</div>"
                )
                with gr.Row():
                    asym_enrich_dd = gr.Dropdown(
                        choices=list_enriched_datasets(),
                        value=list_enriched_datasets()[-1] if list_enriched_datasets() else None,
                        label="Enriched dataset",
                    )
                asym_plot = gr.Plot()
                asym_enrich_dd.change(
                    fn=plot_payoff_asymmetry,
                    inputs=asym_enrich_dd,
                    outputs=asym_plot,
                )

                def _refresh_tab6():
                    default = list_enriched_datasets()[-1] if list_enriched_datasets() else None
                    return (
                        plot_cross_range_game_types(),
                        plot_pareto_and_welfare(),
                        plot_mixed_strategy_distribution(default),
                        plot_payoff_asymmetry(default),
                    )

                refresh_gt_btn.click(
                    fn=_refresh_tab6,
                    outputs=[cross_gt_plot, pareto_plot, mixed_plot, asym_plot],
                )

                # Auto-load on page open
                _default_enrich6 = (list_enriched_datasets()[-1]
                                    if list_enriched_datasets() else None)
                demo.load(fn=plot_cross_range_game_types, outputs=cross_gt_plot)
                demo.load(fn=plot_pareto_and_welfare, outputs=pareto_plot)
                if _default_enrich6:
                    demo.load(
                        fn=lambda: plot_mixed_strategy_distribution(_default_enrich6),
                        outputs=mixed_plot,
                    )
                    demo.load(
                        fn=lambda: plot_payoff_asymmetry(_default_enrich6),
                        outputs=asym_plot,
                    )

    return demo


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    print("Starting Game Theory Matrix Finder…")
    print(f"Datasets directory: {DATASETS_DIR}")
    available = list_datasets()
    print(f"Available datasets: {available or '(none — run generate_datasets.py first)'}")
    print()

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,        # auto-opens http://localhost:7860 in your browser
        max_threads=40,
    )
