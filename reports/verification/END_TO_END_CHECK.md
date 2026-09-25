# End-to-end verification

Verified locally on 2026-09-26 against the pinned UORED-VAFCLS v5 dataset and the local PostgreSQL
runtime. The Groq adapter was not exercised because no provider API key was configured; its results
must not be inferred from the offline fixture.

## Data and training

- Strict raw-data audit: 60 recordings found, 15 primary bearings, 5 load-quarantined bearings,
  zero unparsed files, and zero audit issues.
- Fresh training run: 209.592 seconds (3 minutes 30 seconds), 1,140 paired windows, 20 bearings,
  and model version `fabguard-ac565a24957e5bb0`.
- Selected fusion Isolation Forest: 15-bearing LOBO balanced accuracy 0.955 ± 0.079, AUROC 1.000,
  developing recall 0.989, faulty recall 0.944, and healthy false-positive rate 0.056.
- Confound check: all-20 load-only Random Forest balanced accuracy 0.625 ± 0.222. This weaker,
  load-only result stayed outside the deployed model selection.
- Artifact integrity: the model SHA-256 recomputed from the saved model matched the value in
  `runs/uored-v5-20260926/run.json`.

## Automated checks

- Ruff: passed for `src` and `tests`.
- Unit and integration suite: 67 passed; the PostgreSQL recovery case is skipped only when its
  explicit database URL is absent.
- Live PostgreSQL checkpoint recovery: passed. The graph paused after durable initial retrieval,
  resumed with the same thread, ran initial retrieval exactly once, and reached review.
- Offline workflow evaluation: all 10 frozen held-out authored cases passed for fixed and adaptive
  routes across three repeats; unsafe and unsupported-claim counts were both zero. This checks
  routing and enforcement only because the offline provider is deterministic.

## Runtime flow

A newly scored recorded replay was submitted as the producer, processed once by the worker through
the PostgreSQL reference index, and saved as a ready-for-review report. Its trace was quality,
telemetry, audio, initial retrieval, planner, verifier, and review, with one retrieval and one
offline model call.

The producer received HTTP 403 when attempting to read the incident. The analyst received HTTP 200.
The reviewer created one internal ticket; an identical repeated approval returned the same ticket
with `created: false`, confirming idempotency.
