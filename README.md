# sinn-sdam — Spelke-Seeded Dense Associative Memory (v2)

A PyTorch research repository for **S-DAM**, targeting the NeurIPS 2026 Workshop
on Associative Memory and Hopfield Networks.

S-DAM extends Modern Hopfield Networks (Ramsauer et al. 2021) by seeding the
memory substrate with **four frozen-ish cognitive priors** from Elizabeth
Spelke's core-knowledge theory (Spelke & Kinzler 2007) — Objectness, Agentness,
Numerosity, Geometry — and storing all new knowledge as **geometric residuals**
relative to those priors.

## The idea in one paragraph

Human infants are born with four innate representational systems. S-DAM encodes
these as four seed vectors in a Dense Associative Memory. Instead of storing raw
inputs, it stores only the *residual* — the part of each input not explained by
the four priors. This residual encoding improves storage capacity, reduces
cross-category interference, and makes the system sensitive to the order in
which categories are introduced during training.

## Architecture

| Component | File | Role |
|---|---|---|
| Spelke Seed Layer (SSL) | `sdam/seeds.py` | orthonormal priors; `project` / `residual` / `category` |
| Orthogonal Residual Slots (ORS) | `sdam/hopfield.py` | Modern Hopfield store/retrieve over residuals |
| Surprise-Gated Write Rule (SGWR) | `sdam/model.py` | learned threshold `tau` gates writes |
| Provisional buffer + consolidation | `sdam/model.py` | hippocampal fast-intake; grows new attractors |

Key v2 properties:
- **Seeds are high-inertia, not frozen** — `nn.Parameter` trained at LR `1e-6`.
- **Below-threshold patterns are never discarded** — they enter `self.provisional`.
- `consolidate()` promotes survivors and triggers `_create_new_attractor()`.
- `retrieve_multi()` (top-k blend) and `energy_distance()` are available.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The Modern Hopfield update is implemented directly in `sdam/hopfield.py`; no
external `hopfield-layers` package is needed.

## Run

```bash
pytest tests/                              # all unit tests
python experiments/phase2_interference.py  # PRIMARY experiment — run first
python experiments/phase1_retrieval.py
python experiments/phase3_curriculum.py
```

If CLEVR is not present under `data/`, experiments automatically fall back to
synthetic features (with a printed warning) and never crash.

## Passing criteria

| Check | Requirement |
|---|---|
| `pytest tests/` | zero failures |
| Phase 2 (Lemma 2) | cross mean < 0.20 AND same mean > 0.40 AND p < 0.01 |
| Phase 1 (Lemma 1) | S-DAM ≥ Baseline + 5% at 30% corruption |
| Phase 3 (Theorem 1) | Spelke order non-decreasing, beats random at ≥ 3 stages |
| Reproducibility | Phase 2 run twice → identical JSON |

Phase 2 is the only mandatory experimental result.

## Colab

Open `notebooks/colab_runner.ipynb` and run cells in order. Phase 2 runs before
Phase 1 and Phase 3.

## Layout

```
sdam/         core model (seeds, hopfield, model, utils)
data/         CLEVR loader + synthetic fallback
experiments/  phase1 / phase2 / phase3 runners
configs/      base.yaml (all hyperparameters)
tests/        pytest suite
results/      auto-created, gitignored
results_archive/  committed experiment outputs (NOT gitignored)
notebooks/    colab_runner.ipynb
```
