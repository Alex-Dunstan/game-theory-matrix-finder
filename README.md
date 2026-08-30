# Game Theory Matrix Finder

What happens when you stop using a few canonical games and generate *all of them*? Game Theory Matrix Finder exhaustively builds payoff-matrix spaces, then maps their equilibrium structure.

The full 2x2, 0-10 run creates **214,358,881 games**: a few hundred million rows of strategic data to classify for pure and mixed equilibria, familiar game types, Pareto efficiency, welfare loss, and payoff asymmetry. It is part research engine, part computational stress test, and part way to see the geometry of game theory instead of just reading proofs about it.

Try the [live matrix classifier](https://huggingface.co/spaces/AlexDunstan/game-theory-matrix-classifier).

Explore the [Nash equilibria matrices dataset](https://huggingface.co/datasets/AlexDunstan/nash-equilibria-matrices) on Hugging Face.

![Game-type mix across enumerated payoff ranges](writeup/charts/cell24_07.png)

*Generated game-type distribution across selected matrix spaces. Full method and figures: [`writeup/WRITEUP.md`](writeup/WRITEUP.md).*

---

## Before you run it

> [!WARNING]
> The generator has deliberately minimal safety rails. A full run writes very large local datasets; changing the matrix size or payoff range can multiply the output fast enough to fill disk, exhaust memory, stall the machine, or crash the process. Check available storage, start small, and change settings only if you understand the scale you are asking for. The project does not cap or rescue unsafe runs for you.

---

## Table of Contents

- [Before you run it](#before-you-run-it)
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

It covers pure and mixed equilibria, familiar 2x2 game types, Pareto efficiency, welfare loss, and payoff asymmetry. The engine is reproducible; the write-up explains the results; the [live classifier](https://huggingface.co/spaces/AlexDunstan/game-theory-matrix-classifier) lets you inspect an individual matrix.

Read the illustrated [research write-up](writeup/WRITEUP.md) or the formatted [PDF companion](output/pdf/game-theory-matrix-finder-writeup.pdf).

### Repository shape

The public repository is source-only: code, tests, write-up, charts, and Hugging Face Space source. Generated datasets are excluded from Git; the public Parquet release is hosted as the [Nash equilibria matrices dataset](https://huggingface.co/datasets/AlexDunstan/nash-equilibria-matrices) on Hugging Face. The live public Space is a static browser classifier built from `spaces/`.

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

The default 0-10 run creates 214 million rows and needs substantial disk space and time. Read [Before you run it](#before-you-run-it) before changing generation settings.

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
| Hugging Face Dataset | public Parquet outputs; [browse or download](https://huggingface.co/datasets/AlexDunstan/nash-equilibria-matrices) |
| Local generated datasets | reproducible working outputs; excluded from Git |

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

## Citation

If this software supports your work, cite:

> Dunstan, A. L. (2026). *Game Theory Matrix Finder* [Computer software]. GitHub. https://github.com/Alex-Dunstan/game-theory-matrix-finder

Author: [Alex Lewis Dunstan](https://orcid.org/0009-0007-7869-809X) (ORCID: [0009-0007-7869-809X](https://orcid.org/0009-0007-7869-809X)). For machine-readable metadata, use [`CITATION.cff`](CITATION.cff). Cite the accompanying [Hugging Face dataset](https://huggingface.co/datasets/AlexDunstan/nash-equilibria-matrices) separately when using its data.

---

## License

Released under the [MIT License](LICENSE). Citation metadata is in [`CITATION.cff`](CITATION.cff).
