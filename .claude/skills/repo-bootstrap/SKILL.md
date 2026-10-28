---
name: repo-bootstrap
description: Scaffold a vehicle-platform repo with the standard layout, build system, tests, CI and the vehicle-defs submodule. Use when starting code in a repo that only has README/CLAUDE.md, e.g. "/repo-bootstrap vehicle-rt-core", "set up vehicle-hil-bench host".
argument-hint: <repo-name> [part]
---

<!-- Synced from vehicle-workspace/.claude/skills/repo-bootstrap/SKILL.md by scripts/sync_claude.py. Do not edit here; edit the source and re-run the sync. -->

# Repo bootstrap

Read the target repo's `CLAUDE.md` first; its scope and module list drive the skeleton. Decisions: D-003 (submodule + gen), D-007 (toolchains), D-011 (English), D-012 (RTOS).

## Common to every repo

- `README.md` (English): purpose, place in the platform (link to the org), build/test commands, license line "MIT, see LICENSE (D-036)". Add the MIT `LICENSE` ("Copyright (c) 2026 The vehicle-platform authors").
- Version `0.1.0` in the build file (CMake `project(VERSION)`, `pyproject.toml`, `pubspec.yaml`); releases follow PLATFORM-RULES "Releases" (D-036).
- `.gitignore` suited to the toolchain, `.editorconfig`.
- Submodule, using a **relative URL** so it survives the move from the personal account to the `vehicle-platform` org:
  ```bash
  git submodule add ../vehicle-defs.git external/vehicle-defs
  git -C external/vehicle-defs checkout <tag>   # always a tag (the one in manifest.yaml), never main
  ```
  (vehicle-defs itself and vehicle-mobile skip the submodule unless needed.)
- CI: `.github/workflows/ci.yml` with `submodules: recursive` checkout, build + L0 tests + static analysis, and a step that fails unless the defs pin is a semver release tag `vX.Y.Z` (fetch the tags first, the checkout is shallow; copy the step from vehicle-rt-core `ci.yml`).
- Keep the skeleton minimal: empty module folders get a short `README.md` stating responsibility; no speculative code.

## Type: STM32 firmware (rt-core = STM32H755 Cortex-M7, safety-node = its Cortex-M4 (D-072), hil-bench/simulator)

```
CMakeLists.txt              # add_subdirectory for cubemx, src; option(BUILD_HOST_TESTS)
CMakePresets.json           # presets: debug, release, host-tests
cmake/arm-none-eabi.cmake   # toolchain file (-mcpu per chip, -ffunction-sections, --specs=nano.specs)
cubemx/<board>.ioc          # CubeMX project; generated Core/ Drivers/ live here (Makefile/CMake generator)
src/hal/                    # thin wrappers over HAL: can, imu, gps, gpio, time
src/services/               # signal pool, com (gen pack/unpack + E2E), diag (UDS server), timebase, log
src/features/<module>/      # one folder per feature from CLAUDE.md; no cross-includes
src/app/main_app.c          # task/loop setup, called from cubemx USER CODE
tests/host/                 # Unity (FetchContent), ctest; tests of HAL-free logic
external/vehicle-defs/ # submodule; include gen/c/<node>/ only
.clang-format, .clang-tidy, cppcheck suppressions (MISRA addon)
```
- Compile flags: `-Wall -Wextra -Werror -Wshadow -Wconversion`, no heap on safety nodes (`-Wl,--wrap=malloc` to catch use, or no heap region).
- CI: arm-none-eabi-gcc build of all presets + host tests + cppcheck. Renode (L1) is added once a firmware image boots.
- If the chip/board is not decided yet, ask; do not guess the `.ioc`. rt-core and safety-node share one dual-core NUCLEO-H755ZI-Q project (CM7/CM4 parts, D-072); each repo builds only its own core's image.

## Type: ESP-IDF (connectivity-node)

`CMakeLists.txt`, `sdkconfig.defaults` (target esp32s3), `main/`, `components/<module>/` (bridge, voice (esp-sr), wifi_sync, ble, platform_can), `test/` (Unity host tests where possible), `external/vehicle-defs`. CI: `espressif/esp-idf-ci-action`.

## Type: Python (hil-bench/host, linux-node, server, ml, mcp, defs tools)

`uv init --package` → `pyproject.toml` (Python ≥3.11, ruff + pytest + mypy in dev group), `src/<package>/`, `tests/`, `uv.lock`. CI: `astral-sh/setup-uv`, `uv run ruff check`, `uv run pytest`. CAN code uses `python-can` + `cantools`, and tests run on `vcan` or python-can's virtual bus, never real hardware.

## Type: vehicle-defs

```
uds/vehicle_cl250.yaml  dbc/cl250.dbc (skeleton)  dbc/platform.dbc  vss/overlay.vspec  uds/dids.yaml
tools/codegen/ (uv package: cantools generate_c_source wrapper with per-node filtering, E2E protect/check generator, DID table generator, vss-tools export)
gen/c/<node>/  gen/python/  gen/vss/
Makefile (gen, check)   CHANGELOG.md   docs/
```
CI: parse DBCs strictly, run `make gen` and fail if `git diff --exit-code gen/` shows drift. Seed `platform.dbc` with node definitions, the attribute definitions from `/signal-change`, and heartbeat messages `0x081-0x085`.

## Type: Flutter (vehicle-mobile)

Flutter (D-022). Port the legacy `HondaCl250_Telemetry/mobile_app/flutter_app` per D-023 instead of `flutter create` from scratch.

## Finish

Run the build and tests once and show the result. Update the repo's `CLAUDE.md` build section with the real commands. Suggest a commit message (`chore: bootstrap repository skeleton`), but commit only if asked.
