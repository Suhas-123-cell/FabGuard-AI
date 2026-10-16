# Reproduction on a second machine

The committed results were produced on an ARM laptop. On 2026-10-16 the check was rerun from a
fresh clone and a fresh download of E1-E3 on a Google Colab CPU runtime (Linux `x86_64`) with
`fabguard-ferrara`, and the output folder was compared file by file with the committed one.

- Source revision: `0ba78e2`, the head of `main` when it was cloned (the results zip does not
  record the revision); the Python version and CPU model were not recorded
- Data: Mendeley Data `htk59pp5wx` part 1, 4,917 + 1,985 + 2,386 records

| File | Result |
|---|---|
| `PROTOCOL.md` | byte-identical |
| `summary.json` | byte-identical: not confirmed, 0.878 and 6.50 false alarms per day |
| `per_trial.csv` | byte-identical |
| `scores.csv.gz` | same 30,240 rows; `hours` and every `alarm` identical |

The gzip files differ as bytes because gzip stores a timestamp, and their contents differ only in
the last floating-point digits: `vibration__rms` by at most 1.3×10⁻¹⁵, `threshold` by at most
3.1×10⁻¹⁴ and `anomaly_score` by at most 7.4×10⁻¹³ (relative 6.7×10⁻¹⁴). That is rounding from a
different CPU and maths library, far too small to move any record across its threshold, so the
verdict does not depend on the machine.
