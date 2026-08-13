"""
theme.py — UI theme tokens for the public Hugging Face Space.
"""

BG_PAGE       = "#0F0F0F"
BG_SURFACE    = "#1A1410"
BG_ALT        = "#15110D"

BORDER_STRONG = "#3D2418"
BORDER_SOFT   = "#2A2520"

ACCENT        = "#E8610A"
ACCENT_DIM    = "#8C3A06"

TEXT_PRI      = "#F0EDE8"
TEXT_MUT      = "#7A7570"
TEXT_DIM      = "#4A4540"


GRADIO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

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

body, .gradio-container {
  background-color: var(--bg-page) !important;
  color: var(--text-pri) !important;
  font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
}

h1, h2, h3, h4, .section-label {
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: -0.02em !important;
  color: var(--text-pri) !important;
}

.section-num {
  color: var(--accent) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.75rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.12em !important;
}

.block, .panel, .gr-box, .gr-form {
  background-color: var(--bg-surface) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: var(--radius) !important;
}

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

.gr-dropdown, .gr-select {
  background-color: var(--bg-alt) !important;
  border: 1px solid var(--border-strong) !important;
  color: var(--text-pri) !important;
  border-radius: var(--radius) !important;
}

.prose, .gr-markdown {
  color: var(--text-pri) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  line-height: 1.65 !important;
}

.prose strong, .gr-markdown strong {
  color: var(--accent) !important;
}

.prose code, .gr-markdown code {
  background-color: var(--bg-alt) !important;
  color: var(--text-pri) !important;
  padding: 0.1rem 0.3rem !important;
  border-radius: 4px !important;
  border: 1px solid var(--border-soft) !important;
}

label {
  color: var(--text-mut) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.04em !important;
}
"""
