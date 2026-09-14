# FabGuard AI

FabGuard replays recorded bearing vibration/audio signals, detects abnormal conditions, and produces an inspection report with traceable evidence — a small feature-based detector plus a bounded, checkpointed investigation graph, wrapped in a Streamlit demo.

**Status: design/scaffolding phase.** No models are trained and no results exist yet. This README describes what the project intends to do and how it's reproduced; it is not a claim of completed work. The full build specification lives in [`docs/DESIGN_SPEC.md`](docs/DESIGN_SPEC.md).

## What this is

- **Dataset:** [UORED-VAFCLS v5](https://data.mendeley.com/datasets/y2px5tg92h/5) — 20 bearings, 3 recorded states each, vibration + audio.
- **Detection:** healthy vs. developing/faulty, evaluated with leave-one-bearing-out cross-validation over 15 eligible bearings (5 are quarantined for a load confound; all 20 are used only in a separate confound audit).
- **Investigation:** a sequential, typed-tool graph (not an open-ended agent) that reads telemetry/audio findings, retrieves reference passages, drafts a report, optionally requests one refined search, and writes a checkpointed, resumable incident record.
- **Ablation:** a fixed one-shot pipeline vs. the adaptive graph, compared on 15 authored investigation cases under a predeclared adoption rule.
- **Demo:** Streamlit views for Replay, Incident, and Results, backed by FastAPI + PostgreSQL/pgvector.

See `docs/DESIGN_SPEC.md` for the full protocol (cohort definitions, feature/model grid, execution contract, recovery checks, retrieval/report contract, and the bounded implementation plan).

## Repository layout

```text
fabguard-ai/
├── README.md
├── pyproject.toml
├── compose.yaml
├── src/fabguard/
│   ├── data.py           # Audit, cohorts and folds
│   ├── signals.py        # Vibration/audio features and quality
│   ├── train.py          # Baselines and nested evaluation
│   ├── replay.py         # Recorded-data runner
│   ├── retrieval.py      # Documents, search and provenance
│   ├── graph.py          # Sequential nodes, routing and checkpoints
│   ├── api.py            # Incidents and approval transactions
│   ├── worker.py         # Claims, incident locking and resume
│   └── storage.py        # Database and artifact references
├── app.py                # Streamlit demo
├── configs/              # Frozen experiment settings
├── manifests/            # Identities, exclusions and folds
├── evals/                # Investigation cases and split
├── tests/                # Integrity, recovery and permissions
├── reports/              # Results, ablations and model card
└── runs/                 # Ignored experiment/model artifacts
```

## Setup (planned)

```bash
uv sync                    # or: pip install -e .
docker compose up -d       # PostgreSQL + pgvector
```

Runtime services (`api.py`, `worker.py`, `app.py`) and the data pipeline are not implemented yet — the files above are stubs marking each module's responsibility per the design spec.

## First action

Per the design spec, the first real deliverable is the raw data/audio audit: reconcile the downloaded dataset against the publisher's paper, produce a manifest (bearing ID, health state, manufacturer, load/RPM, channel mapping, sample rate, recording hash), and freeze cohort/exclusion rules before any model training. That audit, and every model or runtime result that follows, is still to be done.

## License

Code: MIT (see `LICENSE`). Dataset: UORED-VAFCLS v5 is CC BY 4.0 per its publisher — cite it separately; this repository does not redistribute the raw data.
