# Game Theory Matrix Finder

Exhaustive enumeration, Nash-equilibrium detection, and classification for two-player payoff matrices.

It generates configurable payoff-matrix spaces, identifies pure and mixed equilibria, classifies familiar 2x2 game types, and presents the results through a research write-up and a lightweight public classifier.

Try the [live matrix classifier on Hugging Face](https://alexdunstan-game-theory-matrix-classifier.static.hf.space).

![Game-type mix across enumerated payoff ranges](writeup/charts/cell24_07.png)

*Generated game-type distribution across selected matrix spaces. Full method and figures: [`writeup/WRITEUP.md`](writeup/WRITEUP.md).*

---

## Table of Contents

- [Security](#security)
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [API](#api)
- [Contributing](#contributing)
- [License](#license)

---

## Security

No credentials, private keys, or external service configuration are required. Do not commit generated datasets, local environments, or `.env` files.

---

## Background

Most introductory game theory begins with a few named examples. This project takes the opposite route: enumerate a whole payoff space, identify every equilibrium structure, then ask which games are common, exceptional, stable, efficient, or strategically awkward.

It covers pure and mixed equilibria, familiar 2x2 game types, Pareto efficiency, welfare loss, and payoff asymmetry. The engine is reproducible; the write-up explains the results; the [live classifier](https://alexdunstan-game-theory-matrix-classifier.static.hf.space) lets you inspect an individual matrix.

Read the illustrated [research write-up](writeup/WRITEUP.md) or the formatted [PDF companion](output/pdf/game-theory-matrix-finder-writeup.pdf).

### Repository shape

The public repository is source-only: code, tests, write-up, charts, and Hugging Face Space source. Generated datasets remain local because the largest files are many gigabytes. The live public Space is a static browser classifier built from `spaces/`.

```text
game-theory-matrix-finder/
  engine/                 generator, classifier, local app, and Python tests
  writeup/                research narrative and charts
  spaces/                 source for the static Hugging Face Space
  scripts/                allowlisted export and validation helpers
  analysis.ipynb          reproducible analysis notebook
  CITATION.cff            citation metadata
  LICENSE                 MIT license
```

---

## Install

```bash
cd engine
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Start

```bash
cd engine
python app.py
```

Then open the local address printed by the app.

### Stop

Press `Ctrl+C` in the Terminal window.

### Run

```bash
cd engine
python generate_datasets.py
python enrich_datasets.py
```

The default 0-10 run creates 214 million rows and needs substantial disk space and time.

---

## Usage

### Layout

| Folder / File | Purpose |
|---|---|
| `engine/` | matrix generation, classification, local app, and tests |
| `engine/datasets/` | local generated CSV output; excluded from Git |
| `writeup/WRITEUP.md` | illustrated research narrative |
| `writeup/charts/` | charts used by the write-up |
| `spaces/` | static public Hugging Face classifier source |
| `scripts/export_hf_space.sh` | exports the allowlisted static Space bundle |
| `scripts/validate_hf_space.sh` | tests and validates the exact Space bundle |

### Roles

| Surface | Role |
|---|---|
| Public GitHub repository | canonical public source |
| Hugging Face Space | generated static browser demo |
| Local generated datasets | reproducible research outputs; not published in Git |

### Deployment

```text
public GitHub repository -> validated static export -> Hugging Face Space
```

Build and validate the exact static bundle before upload:

```bash
./scripts/validate_hf_space.sh
```

The generated `.build/hf_space/` bundle contains only `README.md`, `.gitignore`, `index.html`, `app.mjs`, `classifier.mjs`, and `styles.css`. It contains no datasets, notebook, write-up assets, or local configuration.

### Tech Stack

Python, NumPy, pandas, matplotlib, pytest, Jupyter, JavaScript, and a static Hugging Face Space.

---

## API

| Command | Description |
|---|---|
| `python engine/generate_datasets.py` | generates configured exhaustive datasets locally |
| `python engine/enrich_datasets.py` | adds classification and equilibrium metrics to local datasets |
| `python -m pytest engine/tests` | runs the core engine tests |
| `node spaces/static/test_classifier.mjs` | runs static classifier tests |
| `./scripts/validate_hf_space.sh` | validates and exports the public Space bundle |

---

## Contributing

Small research project. Open an issue or pull request with a reproducible explanation and test coverage for behavioural changes.

---

## License

Released under the [MIT License](LICENSE). Citation metadata is in [`CITATION.cff`](CITATION.cff).
