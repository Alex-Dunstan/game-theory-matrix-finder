"""
theme.py — single source of truth for the project's visual identity.
Import this in both app.py and the Jupyter notebook so every chart and UI
element uses the same palette automatically.

Usage
-----
  from src.theme import MPLSTYLE, GRADIO_CSS, ACCENT, BG_PAGE, TEXT_PRI
  import matplotlib.pyplot as plt
  plt.rcParams.update(MPLSTYLE)
"""

# Pick best available monospace font — avoids matplotlib 'findfont' warnings
# when IBM Plex Mono isn't installed on the host machine.
import matplotlib.font_manager as _fm
_available_fonts = {f.name for f in _fm.fontManager.ttflist}
_MONO_PREFS = ["IBM Plex Mono", "Menlo", "PT Mono", "Courier New", "monospace"]
_MONO_FONT = next((f for f in _MONO_PREFS if f in _available_fonts), "monospace")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

BG_PAGE       = "#0F0F0F"   # outermost page background
BG_SURFACE    = "#1A1410"   # card / panel surface
BG_ALT        = "#15110D"   # alternate rows, nested surfaces

BORDER_STRONG = "#3D2418"   # prominent dividers, chart axes
BORDER_SOFT   = "#2A2520"   # subtle separators, grid lines

ACCENT        = "#E8610A"   # vivid orange — highlights, active states
ACCENT_DIM    = "#8C3A06"   # muted orange — hover, secondary accent

TEXT_PRI      = "#F0EDE8"   # primary readable text
TEXT_MUT      = "#7A7570"   # secondary / metadata text
TEXT_DIM      = "#4A4540"   # placeholder / disabled text

# Convenience list for bar/line chart colour cycling
PALETTE = [ACCENT, "#5BA4CF", "#73C990", "#E5C07B", "#C678DD", "#56B6C2"]


# ---------------------------------------------------------------------------
# Matplotlib rcParams
# ---------------------------------------------------------------------------
# Apply with:  plt.rcParams.update(MPLSTYLE)
# Or once at module level so every subsequent plot inherits it.

MPLSTYLE: dict = {
    # Canvas
    "figure.facecolor":      BG_PAGE,
    "figure.edgecolor":      BG_PAGE,
    "savefig.facecolor":     BG_PAGE,

    # Axes
    "axes.facecolor":        BG_SURFACE,
    "axes.edgecolor":        BORDER_STRONG,
    "axes.labelcolor":       TEXT_PRI,
    "axes.titlecolor":       TEXT_PRI,
    "axes.titlepad":         14,
    "axes.grid":             True,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.prop_cycle":       __import__("cycler").cycler("color", PALETTE),

    # Grid
    "grid.color":            BORDER_SOFT,
    "grid.linewidth":        0.6,
    "grid.alpha":            0.8,

    # Ticks
    "xtick.color":           TEXT_MUT,
    "ytick.color":           TEXT_MUT,
    "xtick.labelsize":       9,
    "ytick.labelsize":       9,

    # Legend
    "legend.facecolor":      BG_ALT,
    "legend.edgecolor":      BORDER_STRONG,
    "legend.labelcolor":     TEXT_PRI,
    "legend.fontsize":       9,

    # Lines & patches
    "lines.linewidth":       1.8,
    "patch.edgecolor":       BG_PAGE,

    # Text / fonts
    "text.color":            TEXT_PRI,
    "font.family":           [_MONO_FONT],
    "font.size":             10,

    # Figure sizing defaults
    "figure.dpi":            110,
    "figure.autolayout":     True,
}


# ---------------------------------------------------------------------------
# Gradio CSS
# ---------------------------------------------------------------------------
# Injected via:  gr.Blocks(css=GRADIO_CSS)
# Uses Google Fonts for Space Grotesk (headings) + IBM Plex Mono (body).

GRADIO_CSS = """
/* ── Google Fonts ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── CSS custom properties (tokens) ───────────────────────────────────── */
:root {
  --bg-page:       #0F0F0F;
  --bg-surface:    #1A1410;
  --bg-alt:        #15110D;
  --border-strong: #3D2418;
  --border-soft:   #2A2520;
  --accent:        #E8610A;
  --accent-dim:    #8C3A06;
  --text-pri:      #F0EDE8;
  --text-mut:      #7A7570;
  --text-dim:      #4A4540;
  --radius:        6px;
}

/* ── Base reset ────────────────────────────────────────────────────────── */
body, .gradio-container {
  background-color: var(--bg-page) !important;
  color: var(--text-pri) !important;
  font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
}

/* ── Headings — Space Grotesk, uppercase, tight ───────────────────────── */
h1, h2, h3, h4, .section-label {
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: -0.02em !important;
  color: var(--text-pri) !important;
}

/* ── Numbered section labels  (e.g. "01 / EXPLORER") ─────────────────── */
.section-num {
  color: var(--accent) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.75rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.12em !important;
}

/* ── Panels / blocks ───────────────────────────────────────────────────── */
.block, .panel, .gr-box, .gr-form {
  background-color: var(--bg-surface) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: var(--radius) !important;
}

/* ── Tabs ───────────────────────────────────────────────────────────────── */
.tabs > .tab-nav {
  background-color: var(--bg-alt) !important;
  border-bottom: 1px solid var(--border-strong) !important;
}
.tabs > .tab-nav > button {
  color: var(--text-mut) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.78rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  border-bottom: 2px solid transparent !important;
  padding: 0.55rem 1.1rem !important;
  background: transparent !important;
  transition: color 0.15s, border-color 0.15s;
}
.tabs > .tab-nav > button.selected,
.tabs > .tab-nav > button:hover {
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
}

/* ── Inputs ─────────────────────────────────────────────────────────────── */
input, textarea, select, .gr-input, .gr-textbox {
  background-color: var(--bg-alt) !important;
  border: 1px solid var(--border-strong) !important;
  color: var(--text-pri) !important;
  border-radius: var(--radius) !important;
  font-family: 'IBM Plex Mono', monospace !important;
}
input:focus, textarea:focus {
  border-color: var(--accent) !important;
  outline: none !important;
  box-shadow: 0 0 0 2px rgba(232, 97, 10, 0.25) !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
button.primary, .gr-button-primary {
  background-color: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-weight: 500 !important;
  letter-spacing: 0.04em !important;
  transition: background-color 0.15s;
}
button.primary:hover {
  background-color: var(--accent-dim) !important;
}
button.secondary, .gr-button-secondary {
  background-color: transparent !important;
  color: var(--text-mut) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: var(--radius) !important;
  font-family: 'IBM Plex Mono', monospace !important;
}
button.secondary:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}

/* ── Dataframe / table ──────────────────────────────────────────────────── */
table, .gr-dataframe table {
  background-color: var(--bg-surface) !important;
  color: var(--text-pri) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.82rem !important;
  border-collapse: collapse !important;
}
th {
  background-color: var(--bg-alt) !important;
  color: var(--accent) !important;
  font-weight: 500 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
  padding: 6px 10px !important;
  border-bottom: 1px solid var(--border-strong) !important;
}
td {
  padding: 4px 10px !important;
  border-bottom: 1px solid var(--border-soft) !important;
}
tr:nth-child(even) td {
  background-color: var(--bg-alt) !important;
}
tr:hover td {
  background-color: rgba(232, 97, 10, 0.06) !important;
}

/* ── Nash-equilibrium highlight class ──────────────────────────────────── */
.ne-cell {
  color: var(--accent) !important;
  font-weight: 600 !important;
}

/* ── Dropdown ──────────────────────────────────────────────────────────── */
.gr-dropdown, .gr-select {
  background-color: var(--bg-alt) !important;
  border: 1px solid var(--border-strong) !important;
  color: var(--text-pri) !important;
  border-radius: var(--radius) !important;
}

/* ── Markdown prose ─────────────────────────────────────────────────────── */
.prose, .gr-markdown {
  color: var(--text-pri) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  line-height: 1.65 !important;
}
.prose code, code {
  background-color: var(--bg-alt) !important;
  color: var(--accent) !important;
  border-radius: 3px !important;
  padding: 1px 5px !important;
  font-size: 0.9em !important;
}

/* ── Scrollbars (WebKit) ────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-page); }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-dim); }
"""
