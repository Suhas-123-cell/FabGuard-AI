# FabGuard model card

## Delivered detector

The deployable model is a `isolation_forest` using `fusion` features,
selected with grouped inner predictions on the 15 eligible training bearings. It emits a
window-level anomaly score and threshold decision; it does not diagnose a fault family or root
cause. Model version: `fabguard-ac565a24957e5bb0`.

## Primary bearing-separated result

The primary cohort contained 45 recordings and
855 overlapping one-second windows from 15 independent bearings. Across
15 outer leave-one-bearing-out folds, the matching policy/family achieved balanced
accuracy 0.955 ± 0.079,
AUROC 1.000 ± 0.000, average precision
1.000 ± 0.000, healthy
false-positive rate 0.056 ±
0.131, developing recall
0.989 ± 0.041, and faulty
recall 0.944 ± 0.217. Standard
deviations describe bearing-fold variability; they are not confidence intervals.

Candidates used grouped inner-fold predictions. Each candidate threshold came from its inner
held-out healthy-score quantile (0.95, 0.975, or 0.99); the selected candidate was then refit on
the outer-training bearings. No outer held-out bearing set its own threshold.

All audio channels passed the frozen technical audit, and paired audio features contribute to the
selected fusion model. Vibration-only and audio-only results remain in `primary_results.csv`.

## Descriptive manufacturer subgroup

`manufacturer_subgroups.csv` reports the delivered detector's held-out-bearing metrics by the
published bearing manufacturer: FAFNIR 203KD (10 bearings), NSK 6203ZZ (5 bearings).
These rows are descriptive only. Manufacturer, bearing number, and fault family do not have the
overlap required to claim manufacturer-independent performance.

## Confound audit

The separately labeled 20-bearing load-only Random Forest reached mean balanced accuracy
0.625 ±
0.222. This is not combined with the primary result;
it demonstrates that operating condition contains label information for the ball-fault bearings.

## Runtime measurement

On `arm` with Python 3.12.13, one thread, batch size one,
20 warm-up calls, and 200 measured calls, model-only
latency was P50 3.862 ms and P95 3.990 ms.
The measured model-only RSS delta was 278,528 bytes.
For a one-second in-memory window, preprocessing plus model latency was P50
8.618 ms and P95
9.052 ms over 100 repetitions, with
an RSS delta of 0 bytes. The one-second acquisition window still
dominates alert delay. These are laptop measurements, not physical edge-device benchmarks.

## Limits

UORED contains short, selected laboratory recordings, accelerated degradation, only 20
independent bearings, and load/manufacturer/fault-family confounding. Cross-validation does not
create equipment diversity. Transfer to other machines is unverified. Snapshot replay does not
demonstrate remaining useful life, continuous deterioration, or field false alarms per hour.
