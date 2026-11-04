# vehicle-ml

Part of [vehicle-platform](https://github.com/vehicle-platform), an SDV-style diagnostics, telemetry and rider-assistance platform for motorcycles (first vehicle: Honda CL250).

Offline model **training** (Python): context classification (first model: road type/surface/riding event), the combined anomaly model (early fusion: CAN + vibration + acoustic + thermal + context), rider identity/style/fatigue (Phase 2). Outputs are **exported** in the format appropriate for the target node: TFLite Micro or ESP-DL (for MCUs), ONNX/TFLite (for the Raspi).

**Status:** the offline data layer (session store, session split, windows, features, normalisation and the dataset CLI), the D-083 stage-1 classical physics estimators and the stage-2 synthetic evaluation (operating-point holdout and physics-weighted hybrid) are implemented. Stage 2 on real rides and HIL fault injection is still to come; no neural model is trained yet.

## Data layer

The data layer reads a `vehicle-server` data directory and writes a reproducible
feature dataset. It never writes into the server data directory and never uses
position columns (`gps_latitude`, `gps_longitude`, ...): the store drops them on
load (D-076, invariant 7).

```sh
uv run vehicle-ml dataset \
  --data-dir /path/to/vehicle-server/data \
  --out ./out/run-001 \
  --kinds telemetry,imu \
  --channels all \
  --window-ms 2000 --stride-ms 1000 --max-gap-ms 500 \
  --seed 0
```

`--channels` takes `all` or a comma-separated list. `all` skips storage metadata and
per-signal `*_age_ms`/`*_valid`, `can_*`, `unanswered_did_count` and the tester
statistics (`step_gap_*`, `rtt_*`); name one explicitly to keep it. `--bands` adds FFT band
energy, e.g. `--bands ax_g:0-5,5-20,ay_g:5-15` (Hz).

The command writes four files:

| file | contents |
|---|---|
| `features.parquet` | one row per window: `session_id`, `split`, `start_ms`, `end_ms` and `<channel>.<stat>` features |
| `split.json` | seed, fractions and the session → split assignment |
| `scaler.json` | means/stds fitted on the **train** split only, plus constant-feature names |
| `dataset.md` | dataset card: splits, channels, per-session modality availability, auto-excluded columns, time base, dropped windows and `defs_version` from each session's `report.json` |

Rules enforced by the code:

- Windows belong to one session; a window is dropped when its samples span more
  than `--max-gap-ms` or a requested channel misses more than 20% of the expected
  samples; with `--bands`, a window whose sample rate is uneven (>5%) is dropped as
  `rate`. Dropped windows are counted by reason in the dataset card (a session
  without usable frames counts as `no_data`).
- The split is a stable hash of `(seed, session_id)`, so adding a session never
  moves an existing one. Rows are never split randomly (CLAUDE.md rule 1).
- Kinds are aligned on `device_time_ms` by nearest-previous sample within
  `--max-gap-ms`; values are never interpolated across a gap. The time base of a
  session is its kind with the most rows (ties: order of `KINDS`).
- The scaler refuses to fit on rows outside the train sessions and fits feature
  columns only (not `session_id`, `start_ms`, `end_ms`, `split`).
- Tests use a synthetic data directory that mirrors the server's Parquet layout;
  no real ride data, GPS positions, VINs or keys are in the repository.

## Stage 1 (D-083)

Stage 1 is a classical physics-informed detector compared with a baseline **without
physics**. It is evaluated only on synthetic sessions with known injected faults; no
number in an evaluation report is evidence for a real ride, and no output is used by a
vehicle node.

| estimator | identifies |
|---|---|
| cooling (`physics/cooling.py`) | first-order `dT/dt = a * q(rpm) - h * (T - T_amb)` with `q = clip(rpm/6000, 0, 1)`; per-window `a`, `h`, `T_amb`, time constant, derivative residual and fit quality. Gap-free stretches only; too-short or unexcited data is refused as not identifiable. |
| charging (`physics/charging.py`) | piecewise-linear `V = min(base + slope * rpm, plateau)`; plateau voltage, cut-in RPM, below-cut-in slope, residual and fit quality. The 14.0-15.0 V plateau in the idea note is CONFIRM, never a hard limit. |
| gearing (`physics/gearing.py`) | per-gear median `gearbox_ratio_measured` learned from train healthy sessions (gear 0 ignored); per-window slip residual. `gear_estimate` stays matched to the defs constants; the table is a learned check, not a new source of truth. |
| RLS (`physics/rls.py`) | recursive least squares with a forgetting factor and covariance reset guard; the online cooling variant matches the batch least-squares answer at `forgetting = 1`. |

The baseline (`vehicle_ml.baseline`) uses the same ML-8 windows and the same session
split: a PCA reconstruction-error detector (numpy SVD) and a robust per-feature
median/MAD z-score. Both are fitted on train sessions only. An isolation forest or a
nonlinear autoencoder would need a new runtime dependency and a licence decision, so it
is not part of this stage.

```sh
uv run vehicle-ml evaluate \
  --data-dir /path/to/vehicle-server/data \
  --out ./out/eval-001 \
  --config configs/physics_cl250.toml \
  --seed 0
```

The evaluator fits thresholds per model on **validation healthy** sessions for a target
false-alarm rate. `--target-false-alarms-per-hour` defaults to `1.0` per hour of healthy
riding. On the test split it reports false alarms per hour, detection delay after onset,
missed faults, and the modality behind each alarm per fault kind. It writes
`evaluation.json` and `evaluation.md` with the split, seed and defs version. Ground truth
is read from an optional `faults.json` next to the data directory (the ML-11 synthetic
helper writes it, with an `expected_modality` per session); without that file the report
contains false alarms only.

### Attribution (rule 3)

Every feature column has a source (`did.<NAME>`, `derived.<name>`, `sensor.<name>`) and a
modality: `can` for DIDs and derived signals, `imu` for `sensor.imu_*`, `other` for the
rest. Both are derived from the pinned defs tables, never from a hand-written list. The
baseline windows include IMU FFT band features (equal-width bands from 0 Hz to the
Nyquist frequency given by the defs IMU `sample_period_ms`). Each alarm records:

- physics: modality `can` and the estimator group (`cooling`, `charging`,
  `longitudinal`) from `[groups]` in the role config;
- PCA baseline: the modality with the largest ratio of its window's summed squared
  reconstruction error to its mean summed squared residual on the healthy training
  windows. The ratio is used because the raw sum favours the modality with more features
  (the IMU has six axes times statistics and FFT bands); the means are fitted once from
  the same healthy training windows as the baseline, never from validation or test. A
  modality with a zero or non-finite training mean is skipped;
- an alarm window with no attributable modality (every modality skipped, or a
  non-finite window) is reported as `unattributed`;
- robust-z baseline: the modality of the feature with the largest |z|; |z| is already
  scale-free per feature, so it is not normalised again.

The report has one row per model and fault kind: expected modality, detections, alarm
windows per modality, the modality of the first alarm after onset and whether it equals
the expected one. A model that does not alarm is reported as a miss. A fault session
whose windows were all unscored by a baseline is reported as "not evaluated" for that
baseline, not as a miss.

The PCA ratio is relative to the in-sample healthy-training mean: it is a relative
attribution, not a calibrated score, and healthy out-of-sample windows can exceed 1 for
every modality. In the seed stability table a seed's first-alarm modality reports the
detected sessions only; partial detection within a seed (some sessions missed) is not
shown there.

The synthetic `wheel_vibration` scenario is the rule-3 weak-CAN case: from onset a
narrow-band line appears in `imu_accel_z` at 4x the rear-wheel rotation frequency, while
every CAN and derived column stays identical to the healthy session with the same seed.
The CAN-only physics models are expected to miss it, and they are not tuned to catch it.

### Sessions without IMU

`build_dataset` still picks a time base per session, so a session with no IMU file gets
a telemetry base and no IMU feature values. The baseline scaler and both baseline
detectors are fitted only on healthy training windows that have every baseline feature.
A window missing any baseline feature gets no baseline score (NaN) and is reported as
**"not scored (missing modality)"**; it is never a detection, a miss or a false alarm.
The CAN-only physics models still score it. `evaluation.json` / `evaluation.md` list the
sessions and windows per split in `baseline_unscored` with the missing modality, and the
dataset card's modality availability table marks the session's `imu` column false.

Two further coverage rules apply. A feature column with no finite value in the healthy
training windows (for example every healthy train session lacks IMU) is excluded from
the baselines, which then use the remaining modalities only; the excluded feature names
and their modalities are listed in `baseline_excluded_features` /
`baseline_excluded_modalities` and in the report's "Baseline coverage" section. If a
baseline has no scored validation healthy window, its threshold cannot be chosen: it is
reported as **"not calibrated"** (`null` in `evaluation.json`), every window of that
model is treated as unscored (its fault sessions are "not evaluated", no false-alarm
episodes or hours), and the physics models are still evaluated. Seed aggregates omit a
rate that has nothing to divide by (no evaluated session of a fault kind, no scored
healthy hour) instead of averaging in a zero.

### Several split seeds

`vehicle-ml evaluate --seeds 0,1,2` runs the session split and the whole evaluation once
per seed (`--seed N` stays the single-seed form; the two are exclusive). A seed changes
only the split, never the data. Per-seed reports go to `<out>/seed-<n>/`; the top-level
`evaluation.json` / `evaluation.md` hold every per-seed result plus the mean, min and max
of each metric over the seeds that produced it. A fault kind that is in the test split
for only some seeds is aggregated over those seeds (the report states how many).

The aggregate also has an **attribution stability** table: for every model and fault
kind it lists the first-alarm modality per seed, how many seeds match
`expected_modality`, and the matching fraction. A seed with no alarm is `no alarm`, an
all-unscored seed is `not evaluated`, and a seed whose test sessions disagree is
`mixed`; none of these is a match. The fraction denominator is every seed that had the
fault kind in its test split.

Role config and defs bridge:

- `configs/physics_cl250.toml` maps library roles (`coolant_temp`, `engine_speed`, ...)
  to `did.<NAME>` or `derived.<name>` vehicle-defs signals.
- `src/vehicle_ml/defs.py` imports `vehicle_defs` from the pinned
  `external/vehicle-defs/gen/python` submodule; it is not a runtime dependency in
  `pyproject.toml`.
- `src/vehicle_ml/data/naming.py` copies the server's `<signal>_<unit>` telemetry column
  rule, and `physics/config.py` validates every role against the pinned defs on load.
  Library code contains role names only, never defs signal names.

Deliberately not modelled in stage 1 (rule 7): the battery equivalent circuit (needs
current and 1 kHz voltage) and air/fuel trims (trim DIDs are not verified, D-068).

**No safety use:** these detectors are offline evidence only. Nothing here outputs to a
node, and no threshold here may loosen a safety ceiling (invariant 3).

## Stage 2 (D-083, synthetic)

Stage 2 adds an operating-point holdout and a physics-weighted hybrid score. It is still
evaluated only on synthetic sessions; no number here is evidence for a real ride, and
nothing feeds a vehicle node.

### Operating-point proxies

The role map has no ambient-temperature signal and the vehicle has no direct load signal
in the roles. The holdout therefore uses two **proxies** taken from the recorded session
itself: median coolant temperature for ambient temperature, and median engine speed for
load. The optional `[operating_point]` table in `configs/physics_cl250.toml` maps the axes
to those existing roles; `session_operating_points` reports the median and valid sample
count per session, and leaves an axis `None` when the session has no valid sample. It
never reads `faults.json` or the synthetic spec. A session with `None` on the holdout axis
is excluded and listed in the report.

### Holdout split and generalisation report

`split_by_band` puts every session with a proxy value inside the closed band into
`test_holdout` and splits the remaining sessions into train/val/test with the same stable
hash as `split_sessions`. Train and val never contain a session from the band; the code
refuses an empty holdout, or a train/val part without healthy sessions. `--seeds` keeps
the holdout membership fixed and changes only the remaining split.

Thresholds are always fitted on validation healthy sessions, never on the holdout. With
`--holdout AXIS:LOW:HIGH`, `evaluation.json` gains a `generalisation` object and
`evaluation.md` a *Generalisation (synthetic)* section: per model false alarms per hour,
detection delay, misses and modality are reported for the in-distribution `test` part and
for `test_holdout`, with deltas between the reported numbers. Neither the fit nor the
threshold uses holdout data.

### Physics-weighted hybrid

Per window, `hybrid = w * physics_score + (1 - w) * baseline_pca_score`. Each score is
normalised by its train-healthy median/MAD before mixing, so the two are on one robust
scale. The fixed grid is `(0.25, 0.5, 0.75, 1.0)`; **`w = 0` is never allowed** because
the physics residual must always be part of the score. Tuning uses the validation split
only: for each candidate it thresholds at the target validation false-alarm rate, then
picks the weight with the most detected validation fault sessions; ties go to the smaller
mean detection delay, then the larger `w`. If validation has no fault session the report
uses `w = 0.5` and marks it `not tuned`. A window the physics model did not score stays
unscored for the hybrid too (never baseline-only), and the report counts those windows.
When the physics weighted term dominates an alarm, its modality is the physics group's;
otherwise the baseline attribution is used, and the dominant term is reported per alarm.
The hybrid model and its section appear only when `--hybrid` is given; the default output
is unchanged.

```sh
uv run vehicle-ml evaluate \
  --data-dir /path/to/vehicle-server/data \
  --out ./out/eval-holdout \
  --config configs/physics_cl250.toml \
  --holdout temperature:30:45 \
  --hybrid
```

Add `--seeds 0,1,2` to repeat the evaluation with the same holdout and aggregate the
per-seed metrics. Stage 2 on real rides and HIL fault injection is still to come; it needs
ride data and the HIL bench, and it does not change any safety threshold.

### Limitations of the synthetic stage 2

- **The temperature proxy is not ambient temperature.** Median coolant temperature rises
  with load, so a coolant band also selects a load range. The synthetic scenario adds
  low-load warm sessions to train/val/test so the cold holdout differs in temperature only;
  without them the physics charging model was scored below its training RPM range and
  every cold "detection" came from the charging group. A real-ride holdout needs the same
  check: compare the load proxy of the holdout with the training range before reading the
  report.
- **Cooling faults move the proxy.** `fan_failure` and `thermostat_stuck_open` change the
  coolant temperature, so a faulty session can enter or leave a temperature band because of
  its fault.
- **Holdout detections are only meaningful next to the holdout false-alarm rate.** When a
  model alarms often on healthy holdout sessions, an alarm after the onset of a fault
  session counts as a detection even if it has nothing to do with the fault. Read
  `detected` together with `false_alarms.per_hour` of the same part.
- On the synthetic cold holdout the physics model does not generalise better than the
  baseline: its cooling group alarms on healthy cold sessions, whose coolant temperatures
  the warm-only fit has never seen. This is reported, not tuned away; no generalisation
  claim follows from it (methodology rule 4).

## Build

```sh
git submodule update --init --recursive   # external/vehicle-defs, pinned to v0.20.0
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

- Architecture and decisions: [vehicle-defs/docs](https://github.com/vehicle-platform/vehicle-defs/tree/main/docs) (`ARCHITECTURE.md`, `DECISIONS.md`)
- Signals, CAN IDs and DIDs come only from [vehicle-defs](https://github.com/vehicle-platform/vehicle-defs) (git submodule pinned to a tag)
- Scope rules for contributors and Claude Code: [`CLAUDE.md`](CLAUDE.md)

## License

MIT, see [LICENSE](LICENSE) (D-036).
