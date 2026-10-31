# vehicle-ml

Part of [vehicle-platform](https://github.com/vehicle-platform), an SDV-style diagnostics, telemetry and rider-assistance platform for motorcycles (first vehicle: Honda CL250).

Offline model **training** (Python): context classification (first model: road type/surface/riding event), the combined anomaly model (early fusion: CAN + vibration + acoustic + thermal + context), rider identity/style/fatigue (Phase 2). Outputs are **exported** in the format appropriate for the target node: TFLite Micro or ESP-DL (for MCUs), ONNX/TFLite (for the Raspi).

**Status:** the offline data layer (session store, session split, windows, features, normalisation and the dataset CLI) and the D-083 stage-1 classical physics estimators are implemented; no neural model is trained yet.

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
- PCA baseline: the modality holding the largest share of the window's squared
  reconstruction error (per-feature residuals summed per modality);
- robust-z baseline: the modality of the feature with the largest |z|.

The report has one row per model and fault kind: expected modality, detections, alarm
windows per modality, the modality of the first alarm after onset and whether it equals
the expected one. A model that does not alarm is reported as a miss.

The synthetic `wheel_vibration` scenario is the rule-3 weak-CAN case: from onset a
narrow-band line appears in `imu_accel_z` at 4x the rear-wheel rotation frequency, while
every CAN and derived column stays identical to the healthy session with the same seed.
The CAN-only physics models are expected to miss it, and they are not tuned to catch it.

### Several split seeds

`vehicle-ml evaluate --seeds 0,1,2` runs the session split and the whole evaluation once
per seed (`--seed N` stays the single-seed form; the two are exclusive). A seed changes
only the split, never the data. Per-seed reports go to `<out>/seed-<n>/`; the top-level
`evaluation.json` / `evaluation.md` hold every per-seed result plus the mean, min and max
of each metric over the seeds that produced it. A fault kind that is in the test split
for only some seeds is aggregated over those seeds (the report states how many).

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
