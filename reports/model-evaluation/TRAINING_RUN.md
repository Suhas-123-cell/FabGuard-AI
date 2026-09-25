# Recorded training run

The final local training run started at **2026-09-26 01:11:45 IST** and finished at
**01:15:15 IST**, taking **209.592 seconds (3 minutes 30 seconds)** end to end.

- Source revision: `1d2bf644aa2e30edcc27c5d46ce5a6e0218a7388`
- Dataset: UORED-VAFCLS v5, 60 recordings, 1,140 paired one-second windows
- Primary cohort: 45 recordings, 855 windows, 15 independent held-out bearings
- The local run directory contains the model digest, configuration, split, metrics, benchmark,
  copied `uv.lock`, and a copy of this report set. It is intentionally ignored because the model
  and intermediate feature/prediction artifacts are reproducible but large.

The elapsed duration includes feature extraction, nested 15-bearing LOBO comparison, the 20-bearing
load-only confound audit, deployment-model fit, benchmarking, and report generation. The saved
model SHA-256 matched its run metadata after the run. It is a local ARM-laptop measurement, not a
hardware requirement or a cloud-training estimate.
