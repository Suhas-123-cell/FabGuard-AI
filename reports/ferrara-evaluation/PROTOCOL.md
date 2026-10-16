# Ferrara check of the 72-hour commissioning rule: protocol

Written on 2026-10-16, before any Ferrara data was downloaded or opened. The point is to test the
72-hour rule that was chosen after seeing the IMS results on data that played no part in choosing it.
Anything decided after looking at the data will be marked as a deviation in the results.

## Data

University of Ferrara run-to-failure dataset of self-aligning double-row ball bearings (SKF 1205 ETN9),
Mendeley Data `htk59pp5wx` (part 1: E1-E3) and its part 2 (E4-E6), CC BY 4.0. Each trial ran
continuously at 40 Hz until the vibration peak reached 20 g, recording 5 s at 25.6 kHz every 5 minutes
(E4: hourly for its first 453 files). Durations from the data paper: E1 408 h, E2 166 h, E3 199 h,
E4 502 h, E5 144 h, E6 43 h. E2 and E3 are downloaded first; the others are added when available.

## Method (unchanged from IMS)

- Features: the IMS vibration features (`ims.COLUMNS`), computed on each whole 5 s record. The
  10-21 kHz band is dropped, as on IMS.
- Models: healthy reference and Isolation Forest (`ims.MODELS`), threshold at the 99th percentile
  of the baseline's own scores, causal 3-of-5 alarm.
- Baselines: `first_24h` and `first_72h`, counted in recorded time. The rigs ran without stopping,
  so the restart rule is not tested. E6 (43 h) is too short for 72 h and is reported under 24 h only.
- Time: from file timestamps if the files carry them, otherwise from the documented cadence.

## When damage starts (fixed now, independent of the detector)

The baseline RMS level is the median RMS over each trial's first 24 hours, the same for both
baseline rules. Damage starts at the first record after which the trailing one-hour (12-record)
median RMS stays at or above twice that level until the end of the trial.

## Metrics, per trial, baseline rule and model

1. False alarms per day and the fraction of time in alarm, from the end of the baseline until
   damage starts.
2. Whether the bearing is in alarm at the end, and how many hours before the end (and relative to
   the damage start) its final alarm began.

If files include the support-bearing accelerometers, those bearings are scored like the healthy IMS
bearings: false alarms counted until the test bearing's damage starts.

## What counts as confirming the rule (healthy reference, `first_72h`)

A trial is eligible if it has at least 24 hours of monitoring between the end of the 72-hour
baseline and the damage start. All four must hold:

- pooled false alarms are at most 0.5 per day across eligible trials (IMS: 0.10);
- every eligible bearing is in alarm at the end;
- `first_72h` has fewer false alarms per day than `first_24h` on the same eligible trials;
- no trial longer than 96 hours has its damage start inside the 72-hour baseline, which would mean
  the rule learned damage as normal.

Failing any condition is reported as the rule not being confirmed.
