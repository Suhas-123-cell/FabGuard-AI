# FabGuard AI

FabGuard replays recorded bearing vibration and audio, scores one-second windows with a small
local detector, and produces a cited inspection report behind human ticket approval. It can also
monitor a live stream from any sensor or microphone a capture tool can read, against that
machine's own baseline. It is a local build with no cloud deployment or equipment control.

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
- `fabguard-monitor`: live or recorded single-channel monitoring with per-machine commissioning,
  whose alarms feed the same investigation path as replays.

The raw audit found 60/60 expected recordings, 420,000 samples each, no content duplicates, and
usable vibration/audio in every file. UORED stores RPM/load once in the first row and zero-pads the
rest; the parser handles that representation explicitly. Bearings 11–15 stay quarantined from the
headline result because their ball-fault recordings change load.

## Results

Mean ± standard deviation is across independent held-out bearings, not windows and not a
confidence interval. The table is window-level. Each row is the best model family for its feature
policy on these same outer folds, so it is optimistic; all nine policy/family rows are in
[primary_results.csv](reports/model-evaluation/primary_results.csv). AUROC ranks one held-out
bearing's own 19 healthy windows against its fault windows, so 1.000 means separable inside a
bearing, not that one threshold works across bearings. Healthy FPR rests on 15 healthy recordings.
Fusion and audio-only are not distinguishable on balanced accuracy.

| Policy / model | Balanced accuracy | AUROC | Healthy FPR | Developing recall | Faulty recall |
|---|---:|---:|---:|---:|---:|
| Vibration / Random Forest | 0.896 ± 0.179 | 1.000 ± 0.000 | 0.074 ± 0.258 | 0.811 ± 0.394 | 0.923 ± 0.227 |
| Audio / healthy reference | 0.954 ± 0.111 | 0.980 ± 0.079 | 0.011 ± 0.041 | 0.933 ± 0.258 | 0.902 ± 0.278 |
| **Fusion / Isolation Forest** | **0.955 ± 0.079** | **1.000 ± 0.000** | **0.056 ± 0.131** | **0.989 ± 0.041** | **0.944 ± 0.217** |

Alerts need persistence: a recording alarms only when 3 of any 5 consecutive windows (about two
seconds) are abnormal. These numbers are recording-level. The rule was added after the out-of-fold
predictions existed and was compared against them, so it is a chosen rule, not a blind one.
`fabguard-train` writes every rule for every policy/family to
[alert_rule_sensitivity.csv](reports/model-evaluation/alert_rule_sensitivity.csv). Out of fold, the
old single-window rule alarmed 3 of 15 healthy recordings (20%); 3-of-5 alarms 2 of 15 (13%) and
misses 1 of 15 faulty recordings (bearing 19). The false alarms sit on whole bearings (6, 9, 16),
so persistence helps only a little; a per-machine baseline (`calibrated_threshold`) is the intended
fix, but it needs at least 120 healthy windows per machine and could not be validated on UORED,
which has 19 per bearing. It can only raise the threshold, never lower it.

The deployed fusion Isolation Forest was selected from grouped inner predictions, not from a test
bearing. That policy/family choice is not nested inside the outer folds, so its headline row is
slightly optimistic. The separate all-20 load-only Random Forest reached 0.625 ± 0.222 balanced accuracy,
which is evidence of the operating-condition shortcut and is not combined with the primary score.
Full comparisons and every bearing fold are in [reports/model-evaluation](reports/model-evaluation).
The selected detector's manufacturer rows are descriptive only: manufacturer, bearing number, and
fault family are confounded in this release, so they do not establish manufacturer generalization.
The recorded end-to-end local training run took 3 minutes 30 seconds; its exact scope and timing are
in [TRAINING_RUN.md](reports/model-evaluation/TRAINING_RUN.md).

The offline workflow check passed 10/10 frozen cases for both fixed and adaptive paths over three
repeats, so the predeclared rule keeps the fixed path as default. The offline provider replays
scripted answers chosen per case, so adaptive cannot gain a case offline by construction: this
validates routing, fail-closed handling and the injection detector, not LLM quality or whether
adaptive retrieval helps. That question is still open until a provider run:

```bash
uv run fabguard-evaluate --provider groq --pause-seconds 2 \
  --input-cost-per-million <usd> --output-cost-per-million <usd>
```

A provider run refuses to start without prices, alternates variant order, scores the model's own
draft as well as the verified report, and stops after three provider failures instead of scoring
outages as case results.

## Other rigs

The UORED fusion model cannot run on other rigs (no microphone, different sensors and sample
rates), so these checks reuse the *method*: the same vibration features, healthy-only models and
3-of-5 alert rule. Nothing was tuned on their outcomes.

**Paderborn KAt (32 bearings, 64 kHz, six folds each holding out one healthy bearing).** The
method does not transfer across bearings. Window balanced accuracy was 0.54-0.59 (0.5 is chance).
At the recording level the healthy reference never alarmed on a healthy recording but caught only
20% of damaged ones; Isolation Forest caught 32% while alarming on 17% of healthy recordings, all
from one bearing (K002). Calibrating on that bearing's own healthy measurements removed those
false alarms but cut its fold's recall from 0.88 to 0 for artificial damage and 0.75 to 0.38 for
real damage.

A follow-up added after that result (`vibration_relative`: crest factor, kurtosis and each band's
share of 0-10 kHz energy, so sensor gain cancels) did worse: balanced accuracy 0.50-0.53 and
10-17% of damaged recordings caught. Sensor gain was not the reason for the poor transfer; a model
fitted on other bearings does not carry over with these features. See
`reports/paderborn-evaluation/`.

**NASA/IMS run-to-failure (3 tests, 4 bearings each, 20 kHz snapshots every 10 minutes).** Each
bearing got its own model fitted on its own early snapshots, as if commissioning a new machine.
Three commissioning rules were compared; the last two were added after the first was seen to fail.
Figures are for the healthy reference, counted over recorded time, on the other bearings before a
failing bearing on the same shaft entered its final alarm (after that they see its vibration):

| Commissioning rule | False alarms/day | Time in alarm | Failing bearings in alarm at the end |
|---|---|---|---|
| first 24 h | 1.78 | 21% | 4/4 |
| first 72 h | 0.10 | 0.6% | 4/4: last 75 h (test 2), 59 h (test 3), 50 h and 158 h (test 1) |
| first 24 h, re-fit after any stop of 12 h or more | 1.69 | 22% | 4/4 |

Isolation Forest was worse on false alarms in every rule (0.91/day with 72 h). The 24-hour rule
failed on test 3, where the machine's normal level drifted after the first day, and on test 1,
where the rig stopped for days. Re-fitting after a stop only helped test 1, and re-fitting can
also learn damage that is already present as normal. The 72-hour rule worked best but was picked
after seeing these tests, uses 44% of test 2's short life as baseline, and still left unattributed
alarm episodes on test 1's failing bearings, so it needs a check on data it has not seen. See
`reports/ims-evaluation/`.

## Monitor your own machine

No industrial hardware is needed to try this. `fabguard-monitor` reads a WAV file or raw 16-bit
mono PCM on stdin, resamples to 42 kHz, and takes the first two minutes as the machine's healthy
baseline. After that it prints one JSON line per half second, and an `alarm` line with an evidence
artifact when 3 of 5 windows are abnormal. A desk fan and a laptop microphone are enough:

```bash
# macOS microphone (Linux: arecord -q -f S16_LE -r 48000 -c 1 -t raw)
ffmpeg -loglevel error -f avfoundation -i ":0" -ac 1 -ar 48000 -f s16le - \
  | uv run fabguard-monitor - --rate 48000 --machine desk-fan

uv run fabguard-monitor recording.wav --channel 0      # a recorded file instead
```

Let the fan run normally until the `commissioned` line, then change it: a fold of paper taped to
the guard so the blades flick it, or a different speed. Keep fingers clear of the blades.

This shows the software path, not detection quality. The baseline is assumed healthy and is never
checked; two minutes is far shorter than the 72 hours that worked on IMS, so expect false alarms as
room noise or temperature drifts; and a microphone hears the room as well as the machine. Alarms
can be submitted for an investigation with `--modality vibration --submit`, because investigations
read vibration evidence: feed it a vibration sensor (an accelerometer through an audio interface,
or a contact microphone), not a room microphone.

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

# Other rigs: about 5.5 GB and 1.1 GB of downloads, processed into data/.
uv run fabguard-paderborn
uv run fabguard-ims
```

The investigation worker defaults to Groq. Set `FABGUARD_LLM_API_KEY` in `.env` and keep
`FABGUARD_LLM_MODEL=openai/gpt-oss-20b`; the worker then uses Groq for planner and report calls.
The adapter uses strict JSON Schema output, a per-call timeout, and one transient retry. Run
`fabguard-evaluate --provider groq` only when intentionally spending provider quota. The explicit
`offline` provider remains available for deterministic tests and local recovery checks; it is never
presented as a trained language model.

Runtime setup with Docker (after training, so `runs/` holds the model):

```bash
cp .env.example .env                    # set three distinct, long role tokens
docker compose up --build               # Postgres, migrations, API :8000, worker, UI :8501
ffmpeg ... -f s16le - | docker compose run --rm -T monitor - --rate 48000 --modality vibration --submit
```

Or run each process directly:

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
diversity. On two other public rigs the method transferred poorly (see Other rigs); it is not
validated on industrial equipment. Snapshot replay does not establish remaining
useful life, continuous deterioration, root cause, or field false alarms per hour. Live monitoring
fits each machine on its own unchecked baseline and has no validation on real equipment. The
72-hour commissioning rule from IMS is still unconfirmed: XJTU-SY cannot test it, because its 15
runs last 42 minutes to 42 hours and every tested bearing fails, leaving no healthy bearing to
count false alarms on. A check needs multi-day recordings that include bearings that stay healthy.
The application can draft an inspection ticket; only an authenticated human can approve it.

Code is MIT licensed. UORED-VAFCLS v5 is CC BY 4.0 and is downloaded from its publisher rather
than redistributed here.
