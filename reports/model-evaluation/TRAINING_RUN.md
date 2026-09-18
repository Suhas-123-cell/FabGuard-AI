# Recorded training run

The final local training run started at **2026-09-18 14:30:43 IST** and finished at
**14:33:51 IST**, taking **187.724 seconds (3 minutes 8 seconds)** end to end.

- Source revision: `715f9818a5a6f83ce2b770e7c73c00f150e357ca`
- Dataset: UORED-VAFCLS v5, 60 recordings, 1,140 paired one-second windows
- Primary cohort: 45 recordings, 855 windows, 15 independent held-out bearings
- The local run directory contains the model digest, configuration, split, metrics, benchmark,
  copied `uv.lock`, and a copy of this report set. It is intentionally ignored because the model
  and intermediate feature/prediction artifacts are reproducible but large.

The elapsed duration includes feature extraction, nested 15-bearing LOBO comparison, the 20-bearing
load-only confound audit, deployment-model fit, benchmarking, and report generation. It is a local
ARM-laptop measurement, not a hardware requirement or a cloud-training estimate.
