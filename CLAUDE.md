# CLAUDE.md — vehicle-ml

@.claude/PLATFORM-RULES.md

## What this repo is

Offline model **training** (Python): context classification (first model: road type/surface/riding event), the combined anomaly model (early fusion: CAN + vibration + acoustic + thermal + context), rider identity/style/fatigue (Phase 2). Outputs are **exported** in the format appropriate for the target node: TFLite Micro or ESP-DL (for MCUs), ONNX/TFLite (for the Raspi).

## What this repo is NOT

- Inference is NOT here. Inference lives in the relevant node's `features/` folder (context/anomaly network → rt-core, anomaly model → linux-node, voice command → connectivity-node).
- No voice command model is trained (ESP-SR MultiNet is ready-made). No model for lane departure warning (classic CV is used).
- No model makes safety decisions. ML only personalizes; it never sets the ceiling.

## Methodology rules (non-negotiable for thesis validity)

1. The train/test split is **session-based**. Data from the same ride cannot be in both training and test. Random row splitting is FORBIDDEN.
2. CWRU/MaFaulDa/MIMII are only methodology references, NOT direct training data (domain gap). A transfer attempt is reported as "attempted, result was X."
3. Which modality catches which fault is reported separately (the risk of CAN masking vibration/acoustic signals). At least one fault scenario must be of the kind that leaves a weak trace on CAN.
4. No generalization claim is made: organic faults that weren't tested are reported as "unknown."
5. MCU-targeted models are reported together with their memory budget (ARCHITECTURE §2 RAM/flash). A model that exceeds the budget is not delivered.

## Dependencies

`external/vehicle-defs` (signal names), data comes from `vehicle-server` (Parquet).

## Build

`uv` + `ruff` + `pytest`. Experiment tracking is kept lightweight (trackio or MLflow). Raw data does not go into git.

## Context

`../vehicle-defs/docs/hardware-architecture.md` §5b.0, §5b.9 · `../vehicle-defs/docs/phase0-data-collection-plan.md`.
