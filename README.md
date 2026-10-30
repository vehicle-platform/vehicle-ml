# vehicle-ml

Part of [vehicle-platform](https://github.com/vehicle-platform), an SDV-style diagnostics, telemetry and rider-assistance platform for motorcycles (first vehicle: Honda CL250).

Offline model **training** (Python): context classification (first model: road type/surface/riding event), the combined anomaly model (early fusion: CAN + vibration + acoustic + thermal + context), rider identity/style/fatigue (Phase 2). Outputs are **exported** in the format appropriate for the target node: TFLite Micro or ESP-DL (for MCUs), ONNX/TFLite (for the Raspi).

**Status:** the offline data layer (session store, session split, windows, features, normalisation and the dataset CLI) is implemented; no model is trained yet.

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
