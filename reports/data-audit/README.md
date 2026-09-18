# UORED-VAFCLS v5 data audit

Audited 60 recordings from 20 bearings. All 60 vibration and 60 audio channels passed the frozen technical checks. The primary cohort contains 45 recordings from 15 bearings; 15 recordings from bearings 11–15 remain in the separate load-confound audit. No recording was excluded for being difficult to classify.

The release encodes RPM and load as a scalar in the first CSV row followed by zero padding. `recording_quality.csv` reports the recovered scalar values. `confound_table.csv` makes the ball-fault load shift explicit. Plots and normalized listening copies are deterministic quality checks, not outcome-based case selection.
