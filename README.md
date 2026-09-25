# FabGuard AI

FabGuard replays recorded bearing vibration and audio, scores one-second windows with a small
local detector, and produces a cited inspection report behind human ticket approval. It is a
local, software-only build: no sensor hardware, cloud deployment, or equipment-control tool.

## What is actually here

- A pinned downloader and full raw audit for
  [UORED-VAFCLS v5](https://data.mendeley.com/datasets/y2px5tg92h/5).
- Amplitude-preserving DSP, three feature policies, three model families, nested grouped tuning,
  15-fold leave-one-bearing-out (LOBO) evaluation, and a separate 20-bearing load-confound audit.
- A bounded sequential LangGraph investigation: at most two retrievals and two logical model
  calls, strict schemas, deterministic verification, and fail-closed reports.
- FastAPI, PostgreSQL/pgvector migrations, durable deliveries/checkpoints/revisions, a synchronous
  worker, idempotent approvals, and three Streamlit views.
- A 15-case workflow suite with five development and ten frozen evaluation fixtures.

The raw audit found 60/60 expected recordings, 420,000 samples each, no content duplicates, and
usable vibration/audio in every file. UORED stores RPM/load once in the first row and zero-pads the
rest; the parser handles that representation explicitly. Bearings 11–15 stay quarantined from the
headline result because their ball-fault recordings change load.

## Results

Mean ± standard deviation is across independent held-out bearings, not windows and not a
confidence interval.

| Policy / model | Balanced accuracy | AUROC | Healthy FPR | Developing recall | Faulty recall |
|---|---:|---:|---:|---:|---:|
| Vibration / Random Forest | 0.896 ± 0.179 | 1.000 ± 0.000 | 0.074 ± 0.258 | 0.811 ± 0.394 | 0.923 ± 0.227 |
| Audio / healthy reference | 0.954 ± 0.111 | 0.980 ± 0.079 | 0.011 ± 0.041 | 0.933 ± 0.258 | 0.902 ± 0.278 |
| **Fusion / Isolation Forest** | **0.955 ± 0.079** | **1.000 ± 0.000** | **0.056 ± 0.131** | **0.989 ± 0.041** | **0.944 ± 0.217** |

The deployed fusion Isolation Forest was selected from grouped inner predictions, not from a test
bearing. The separate all-20 load-only Random Forest reached 0.625 ± 0.222 balanced accuracy,
which is evidence of the operating-condition shortcut and is not combined with the primary score.
Full comparisons and every bearing fold are in [reports/model-evaluation](reports/model-evaluation).
The selected detector's manufacturer rows are descriptive only: manufacturer, bearing number, and
fault family are confounded in this release, so they do not establish manufacturer generalization.
The recorded end-to-end local training run took 3 minutes 30 seconds; its exact scope and timing are
in [TRAINING_RUN.md](reports/model-evaluation/TRAINING_RUN.md).

The offline deterministic workflow check passed 10/10 frozen cases for both fixed and adaptive
paths over three repeats. Adaptive retrieval passed no additional case, so the predeclared rule
keeps the fixed path as default. This result validates routing and safety fixtures, not LLM quality;
rerun with Groq for a model-based ablation.

## Reproduce

Python 3.12 is the tested runtime. Raw data, run directories, WAV listening copies, and trained
models are intentionally ignored by Git.

```bash
uv sync --extra dev --python 3.12

# About 835 MB: 60 pinned CSV files from Mendeley Data v5.
uv run fabguard-download

uv run fabguard-audit data/raw/csv \
  --output manifests/uored_vafcls_v5.json \
  --report-dir reports/data-audit --strict

uv run fabguard-train \
  --manifest manifests/uored_vafcls_v5.json \
  --config configs/experiment.json \
  --run-dir runs/uored-v5-seed17 \
  --report-dir reports/model-evaluation

uv run fabguard-evaluate --provider offline --output-dir runs/investigation-eval
uv run pytest -q
```

The investigation worker defaults to Groq. Set `FABGUARD_LLM_API_KEY` in `.env` and keep
`FABGUARD_LLM_MODEL=openai/gpt-oss-20b`; the worker then uses Groq for planner and report calls.
The adapter uses strict JSON Schema output, a per-call timeout, and one transient retry. Run
`fabguard-evaluate --provider groq` only when intentionally spending provider quota. The explicit
`offline` provider remains available for deterministic tests and local recovery checks; it is never
presented as a trained language model.

Runtime setup:

```bash
cp .env.example .env                    # set three distinct, long role tokens
docker compose up -d postgres
docker compose exec -T postgres psql -U fabguard -d fabguard < migrations/001_runtime.sql
docker compose exec -T postgres psql -U fabguard -d fabguard < migrations/002_references.sql
uv run fabguard-index-references --database-url "$FABGUARD_DATABASE_URL"

uv run fabguard-api                     # terminal 1
uv run fabguard-worker                  # terminal 2; add --adaptive only after an ablation passes
uv run streamlit run app.py             # terminal 3
```

`fabguard-replay <opaque-recording-id>` creates an on-disk evidence artifact. Add `--submit` after
setting `FABGUARD_API_URL` and the producer token in the environment to ingest it idempotently.
Database credentials and bearer tokens are never accepted on the command line. The UI reads and
approves incidents through the API; rerenders do not launch investigations.

## Limits

These are short, selected laboratory recordings with accelerated degradation, only 20 independent
bearings, and load/manufacturer/fault-family confounding. Cross-validation does not create machine
diversity. Transfer to other equipment is unverified. Snapshot replay does not establish remaining
useful life, continuous deterioration, root cause, or field false alarms per hour. The application
can draft an inspection ticket; only an authenticated human can approve it.

Code is MIT licensed. UORED-VAFCLS v5 is CC BY 4.0 and is downloaded from its publisher rather
than redistributed here.
