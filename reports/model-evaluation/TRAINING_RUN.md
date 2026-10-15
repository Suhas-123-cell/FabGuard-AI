# Recorded training run

The final training run started at **2026-10-15 12:03:59 IST** and finished at
**12:23:09 IST**, taking **1,149.444 seconds (19 minutes 9 seconds)** end to end.

- Source revision: `a4f73907687790dd641d43e99ff18302cf6d4db3`
- Machine: Google Colab CPU runtime (`x86_64`, Python 3.13.16); the exact CPU model was not recorded
- Dataset: UORED-VAFCLS v5, 60 recordings, 1,140 paired one-second windows
- Primary cohort: 45 recordings, 855 windows, 15 independent held-out bearings
- Model version `fabguard-ac565a24957e5bb0`, SHA-256
  `09022f10f59744f1a09039379189115780fa24c884721df77e48d543521f770f`
- The run directory contains the model digest, configuration, split, metrics, benchmark,
  copied `uv.lock`, and a copy of this report set. It is intentionally ignored because the model
  and intermediate feature/prediction artifacts are reproducible but large.

The elapsed duration includes feature extraction, nested 15-bearing LOBO comparison, the 20-bearing
load-only confound audit, deployment-model fit, benchmarking, and report generation. The saved
model SHA-256 matched its run metadata after the run. Every evaluation CSV here is byte-identical
to a local ARM-laptop run of the same code; only the model card's runtime measurement depends on
the machine. It is a Colab measurement, not a hardware requirement.
