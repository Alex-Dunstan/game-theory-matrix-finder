---
title: Game Theory Matrix Classifier
emoji: 🎲
colorFrom: yellow
colorTo: blue
sdk: static
app_file: index.html
pinned: false
---

# Game Theory Matrix Classifier

This Space is a compact public-facing demo for exploring 2x2 payoff matrices.

It is packaged as a plain static Hugging Face Space, so the app runs client-side in the browser with no Python runtime and no Gradio dependency.

It does three things:

1. Finds pure-strategy Nash equilibria
2. Computes the mixed-strategy equilibrium when no pure equilibrium exists
3. Classifies the matrix into familiar game types

Included game-type labels:

- `Prisoner's Dilemma`
- `Harmony`
- `Deadlock`
- `Battle of the Sexes`
- `Stag Hunt`
- `Chicken`
- `Coordination`
- `Zero-Sum`
- `Dominant (P1 only)`
- `Dominant (P2 only)`
- `No Equilibrium`
- `Other`

This Space is intentionally lightweight. It does **not** include the larger local research repo, datasets, notebook, or write-up.

## Author and source

Created by [Alex Lewis Dunstan](https://orcid.org/0009-0007-7869-809X) (ORCID: [0009-0007-7869-809X](https://orcid.org/0009-0007-7869-809X)). Source code and research write-up: [Game Theory Matrix Finder](https://github.com/Alex-Dunstan/game-theory-matrix-finder).

## Citation

This Space is an interactive deployment of the canonical software project; cite the project rather than this deployment:

> Dunstan, A. L. (2026). *Game Theory Matrix Finder* [Computer software]. GitHub. https://github.com/Alex-Dunstan/game-theory-matrix-finder

Cite the [Nash Equilibria of 2x2 Normal-Form Games dataset](https://huggingface.co/datasets/AlexDunstan/nash-equilibria-matrices) separately when using its data.
