<!-- Synced from vehicle-workspace/external-ai/AGENTS.base.md by scripts/sync_claude.py. Do not edit here; edit the source and re-run the sync. -->

# AGENTS.md — vehicle-platform rules for AI coding tools other than Claude Code

You are an AI coding tool (Codex, Gemini, DeepSeek, Cursor, Aider, Copilot, …) working on vehicle-platform, a motorcycle diagnostics / telemetry / ADAS platform with safety-relevant firmware. Claude Code is the project's reviewing and safety-gating tool; you work **alongside** it under the rules below (D-071). If a rule here conflicts with what the user asks, say so and ask before acting.

## 1. Every session

1. Read `STATUS.md` at the workspace root (short): where the project is and what is next. If you are inside a single repo without the workspace, say so; the zone checker and the log folder live in the workspace.
2. For architecture facts read `vehicle-defs/docs/ARCHITECTURE.md` and `DECISIONS.md` (only the parts you need; `docs/README.md` indexes the large docs). Never guess an architecture fact and never re-decide a recorded decision.
3. Before editing a file, check its zone: `python3 scripts/check_zones.py <repo> --path <file>` (workspace root). **Red = do not edit.**
4. Open your log entry (section 5) at the start, not at the end.

## 2. Edit zones (`external-ai/zones.json`, first matching rule wins)

- **red** — Claude only. Do not edit, not even "small fixes". Describe the wanted change under *For Claude* in your log and continue with something else.
- **yellow** — allowed; the PR needs a Claude review before the user merges it (write `needs-claude-review` in the PR description).
- **green** — allowed; still log it.

- **All repos (checked first)** — red: `.claude/*`, `AGENTS.md`, `GEMINI.md`, `CLAUDE.md`, `.github/*`, `external/vehicle-defs`, `external/*`, `gen/*`, `*/gen/*`, `misra/*`
- **vehicle-workspace** — red: `external-ai/log/TEMPLATE.md`, `external-ai/*`, `PLATFORM-RULES.md`, `manifest.yaml`, `scripts/*`, `setup.sh`; yellow: `STATUS.md`, `ISSUES.md`; green: `external-ai/log/*`; everything else: **yellow**
- **vehicle-defs** — red: `docs/DECISIONS.md`, `docs/ARCHITECTURE.md`; yellow: `docs/*`; green: `docs/ideas/*`; everything else: **red**
- **vehicle-safety-node** — everything: **red**
- **vehicle-io-node** — everything: **red**
- **vehicle-rt-core** — red: `src/features/cornering/*`, `src/features/uds/*`, `src/features/heartbeat/*`, `src/features/vehicle_republish/*`, `src/features/*anomaly*`, `src/features/*boot*`, `src/services/alive*`, `src/services/snapshot*`, `src/services/can_*`, `src/services/mailbox*`, `src/app/*init*`, `cubemx/*`, `*.ld`; everything else: **yellow**
- **vehicle-connectivity-node** — red: `src/HondaCANModule*`, `src/DiscoveryScan*`, `src/IsoTp*`, `src/TesterLatch*`, `src/CanCapture*`, `src/hal/Twai*`, `src/hal/RtcTesterLatchStore*`, `platformio.ini`; yellow: `src/Gps*`; everything else: **yellow**
- **vehicle-hil-bench** — red: `*fault*`; everything else: **yellow**
- **vehicle-linux-node** — red: `*ota*`; everything else: **yellow**
- **vehicle-server** — red: `*ota*`; yellow: `*gps*`, `*position*`; everything else: **green**
- **vehicle-mobile** — yellow: `lib/services/session_upload*`, `lib/services/upload_client*`; everything else: **green**
- **vehicle-analyzer** — red: `src/services/webSerialCan*`, `src/services/udsComposer*`, `src-tauri/*`; yellow: `src/defs/*`; everything else: **green**
- **vehicle-mcp** — everything: **yellow**
- **vehicle-ml** — everything: **green**
- Unlisted repos and paths: yellow. Where two globs match, the more specific one listed first in
  `external-ai/zones.json` wins (e.g. `docs/ideas/*` before `docs/*`); the file also gives each rule's reason.

## 3. Hard rules that no path rule can catch

- **No hand-written signal facts.** CAN IDs, DIDs, signal names, scales, units, VSS paths and BLE layouts come only from the generated `gen/` code or the definition files of vehicle-defs. Never write such a literal in code; if a signal is missing, that is a red-zone request (`/signal-change` by Claude).
- **Nothing new on the vehicle bus.** No code that sends a CAN frame to the motorcycle outside the existing tester modules (red); no new UDS service, no write, clear, reset, unlock or reprogram, ever.
- **MCU code** (rt-core, safety-node, conn, hil-bench firmware): no heap (`malloc`, `new`, STL containers that allocate) after init, no string/JSON/DBC parsing, no blocking calls in the tester loop, MISRA C:2012 style in C.
- **Privacy:** never commit keys, passwords, real ride data, VINs, or GPS positions. Position may go only to the user's own servers (D-076, provisional); code that moves it is red (Claude only). The repos are public.
- **Safety path:** never make a safety decision (cornering, blind spot, immobilizer) depend on the Raspi, the phone, Wi-Fi, VSS or ML; ML may only tighten thresholds.
- **No new dependency** in firmware or in `pyproject.toml` / `pubspec.yaml` without asking the user.
- **Do not weaken checks:** no disabled tests, lowered coverage, new MISRA suppressions, `# noqa` or `// cppcheck-suppress` to make CI green. Fix the code or log the problem.
- **No new repositories.** Never create a repo in the org or a new top-level project; propose it under "For Claude" in your log (a new repo needs a D-xxx, its zones and the shared assets first, D-081).
- **Hardware facts need a source.** Part numbers, pinouts, dimensions and prices you could not verify are marked **CONFIRM**; never rank an open question's candidates (Q-xxx) as decided, and never move a safety input (M4 board, D-082) onto a QM or detachable board.
- **No releases:** no tags, no version bumps, no submodule pin changes, no `manifest.yaml` edits (D-036). Never merge a PR and never push to `main`; the user merges.

## 4. Workflow

1. Branch: `ext/<tool>-<topic>` (e.g. `ext/gemini-mobile-dashboard`), in the repo you change. Never `main`, never a `claude/` branch. One topic per branch.
2. Commits: Conventional Commits in English (`feat(mobile): ...`) with the trailer line `AI-Agent: <tool>/<model>` (e.g. `AI-Agent: gemini-cli/gemini-2.5-pro`). Claude finds your work with `git log --grep '^AI-Agent:'`.
3. Before each commit: run the repo's checks (table below) and `python3 scripts/check_zones.py <repo>`. Red → revert those paths. Commit and push only when the user asks.
4. PR (when the user asks): title in Conventional Commits form; description = summary, checks run with their result, the zone result, `needs-claude-review` if any path is yellow, and the log file name.
5. Write code that reads like the surrounding code: same naming, comment density, test style. Add or update tests for what you change.

| Repo | Checks (run from the repo root) |
|---|---|
| vehicle-defs | red: do not edit (docs only: no check) |
| vehicle-rt-core | `cmake --preset host-tests && cmake --build --preset host-tests && ctest --preset host-tests` (+ `host-tsan` preset for snapshot/alive/IMU); cppcheck as in `.github/workflows/ci.yml` |
| vehicle-connectivity-node | `scripts/native_tests.sh` (or `pio test -e native`); `pio run -e esp32-s3-devkitc-1-mock`; cppcheck as in CI |
| vehicle-mobile | `flutter pub get && flutter analyze && flutter test` (never commit a local `pubspec.lock` rewrite) |
| vehicle-server | `uv sync && uv run pytest && uv run ruff check . && uv run ruff format --check .` |
| vehicle-analyzer | `git submodule update --init && npm ci && npm test && npm run build` |
| other repos | the commands in that repo's README/CLAUDE.md "Build" section |

## 5. Session log (mandatory)

Every session writes one file: `external-ai/log/YYYY-MM-DD-<tool>-<topic>.md` in the workspace, copied from `external-ai/log/TEMPLATE.md`, with `review: pending`. Fill it while you work: what changed (repo, branch, files), why, checks and results, zone result, what you were unsure about, and *For Claude* (red-zone requests, questions, suspected rule conflicts). Without the workspace, put the same text into the PR description. Claude reviews pending logs with its `/external-review` skill, refactors where needed and sets `review: done`.

## 6. What only Claude does

Red-zone edits; signal, UDS, BLE and limit changes (`/signal-change`) and every vehicle-defs edit outside `docs/`; new or changed decisions in `DECISIONS.md`; safety reviews (`safety-reviewer`) of invariant-8 modules; vehicle-bus tester changes; releases, tags and pins; synced files (`.claude/`, `AGENTS.md`, `GEMINI.md`, `PLATFORM-RULES.md`). You may update `STATUS.md` (keep ≤ 60 lines and its sections) and add rows to `ISSUES.md`.

## 7. Platform rules (copied from PLATFORM-RULES.md; "Claude" there means any AI tool unless a rule names Claude's agents or skills)

<!-- Source of truth: vehicle-workspace/PLATFORM-RULES.md. Copies in each repo's .claude/ are written by scripts/sync_claude.py — edit only the source. -->

SDV-architecture diagnostics / telemetry / ADAS platform for motorcycles (first vehicle: Honda CL250). **Scope (D-078):** for now only a test prototype is built and tested on the CL250; the scale is widened into a commercial multi-vehicle product later, so keep vehicle facts in per-vehicle defs files and never delete a catalog DID because the CL250 does not serve it. Senior thesis → GitHub org `vehicle-platform` (12 platform repos + vehicle-workspace, **public** — D-033; never commit keys, real GPS/ride data or personal data). The user writes in Turkish. Reply in Turkish, but write every project artifact (code, comments, docs, commits) in English.

### Where things are

- Platform docs live in the `vehicle-defs` repo under `docs/`. Depending on where you run, the path is `vehicle-defs/docs/` (workspace root), `docs/` (inside vehicle-defs) or `external/vehicle-defs/docs/` (consumer repo with the submodule). If none of these exist (e.g. a single-repo cloud session), say so and ask. Never guess architecture facts.
- Read first: `ARCHITECTURE.md` (summary) and `DECISIONS.md` (D-xxx decisions, Q-xxx open questions). `docs/README.md` indexes the large docs by line range. Read only the needed range, never a whole raw doc.
- Do not re-litigate recorded decisions. New decisions → ask the user, then record them in `DECISIONS.md`.

### Repo map (dependencies are one-way: everyone → vehicle-defs)

| Repo | Runtime | Tooling | Role |
|---|---|---|---|
| `vehicle-defs` | — | DBC, VSS, UDS YAML, Python codegen | Single source of truth for signals + platform docs |
| `vehicle-rt-core` | STM32H755 Cortex-M7 (D-072) | C, CMake+CubeMX, FreeRTOS | Domain controller: CAN gateway, logging, fusion, UDS/ISO-TP, bootloader, XCP, EKF, context, dyno, anomaly safety net, power management |
| `vehicle-safety-node` | STM32H755 Cortex-M4 (D-072) | C, CMake+CubeMX, bare-metal | The isolated safety partition: cornering decision + own IMU, blind-spot decision, immobilizer, LED ring and mirror LEDs |
| `vehicle-io-node` | — | — | **Retired by D-072** (no code); its functions moved to safety-node and rt-core |
| `vehicle-connectivity-node` | ESP32-S3 | C, ESP-IDF | Wi-Fi/BLE + voice (ESP-SR); telemetry here is TEMPORARY |
| `vehicle-linux-node` | Raspberry Pi 5 | Python/C++ | Kuksa, CAN→VSS, lane keeping (classic CV), anomaly model, HMI, OTA |
| `vehicle-mcp` | Raspberry Pi 5 + remote server (D-077) | Python | Standalone open-source MCP server (read-only tools) |
| `vehicle-hil-bench` | STM32F4 + i7 host | C + Python | Restbus HIL bench, scenario engine, CI runner |
| `vehicle-server` | Raspi (edge) + remote, VPS-ready (D-077) | Python (FastAPI) | Ingestion, storage, MDF4/Parquet, OTA packages |
| `vehicle-ml` | Offline | Python | Training only |
| `vehicle-mobile` | Phone | Flutter (D-022) | Loosely coupled companion |
| `vehicle-analyzer` | Host (browser, later Tauri desktop) | TypeScript/React, Rust HAL later (D-081) | CAN analyzer for both buses; listen-only adapters, never transmits on a vehicle bus |
| `vehicle-workspace` | — | — | Manifest, setup script, shared Claude agents/skills (source of truth) |
| `HondaCl250_Telemetry` | — | archived | Read-only legacy reference (verified CL250 CAN/UDS code). Never modify it; extract facts per D-023 |

### Platform invariants (non-negotiable; if one must be broken, STOP and ask)

1. **The ECU is read-only.** The CL250 ECU only answers requests (D-019), so the vehicle bus has exactly **one tester, and it only reads** (D-037): rt-core (D-021); until rt-core exists, connectivity-node is the temporary sole tester (D-023), and there are never two. It sends only what `vehicle-defs/uds/vehicle_cl250.yaml` → `tester_policy` allows (D-020: session control 0x01/0x03 only, tester present, read services), through the generated gates. That policy may only be narrowed; codegen's golden copy blocks widening (D-027). Widening either one needs the user's approval, a versioned defs change and `safety-reviewer`. Nothing that writes, clears, resets, unlocks, controls or reprograms the ECU, and never a map change. All other nodes stay off the vehicle bus or strictly listen-only. rt-core republishes decoded vehicle signals on the separate platform CAN bus, where our nodes talk (D-009, D-021).
2. **Never invent signals.** CAN IDs, signal names, scales, VSS paths and DIDs come only from `vehicle-defs` (generated `gen/` code or DBC/VSS). Hand-written literals → `vss-schema-guardian`.
3. **The safety path is isolated.** The cornering and blind-spot decisions run on the Cortex-M4 safety partition (safety-node, D-072) and work without the Raspi, the phone, Wi-Fi, VSS, ML and the rt-core software; the partition talks to rt-core only through its E2E-checked mailbox. ML may only tighten thresholds, never loosen the safety ceiling.
4. **Critical work never depends on a slow channel.** µs-critical → MCU + CAN. ms → Raspi. Latency-tolerant → phone/server.
5. **No dynamic memory on MCUs** (strictly none on safety paths, avoided elsewhere). No string/DBC/VSS parsing on MCUs.
6. **The immobilizer acts on the starter-relay coil circuit only.** Default state unlocked, hidden bypass, 10 s timeout to unlock.
7. **The LLM never does its own math.** Position (lat/lon/height) may go only to the user's own servers and to the LLM through vehicle-mcp, and is never committed (provisional relaxation, D-076; revisited by Q-035).
8. After any change to a safety-critical module (cornering/, safety-node, immobilizer, blind spot, watchdog/heartbeat, anomaly-safety-net, E2E, bootloader, and on the STM32H755 the inter-core mailbox, MPU/clock/reset/option-byte init and both linker scripts, D-072), run `safety-reviewer` before committing.

### Conventions

- Conventional Commits (`feat(uds): ...`). Commit or push only when the user asks.
- AI tools other than Claude Code follow `AGENTS.md` (D-071): edit zones (red = Claude only), `ext/<tool>-<topic>` branches, an `AI-Agent: <tool>/<model>` commit trailer and a session log in `vehicle-workspace/external-ai/log/`; Claude reviews their work with `/external-review`.
- `vehicle-defs` uses semver. Consumers pin the `external/vehicle-defs` submodule (relative URL) to a **tag**. The known-good combination is recorded in `vehicle-workspace/manifest.yaml`.
- Releases (D-036): bump the build-file version (CMake `project(VERSION)`, `pyproject.toml`, `pubspec.yaml`) in a PR → merge after green CI → `gh release create vX.Y.Z -R vehicle-platform/<repo> --target main --title vX.Y.Z --generate-notes` → pin the tag in `manifest.yaml` (workspace PR). vehicle-defs also adds a `CHANGELOG.md` entry; docs-only changes need no tag.
- `docs/archive/` (Turkish originals, advisor documents) is not read by Claude (D-038).
- Skills: `/repo-bootstrap` (repo skeleton), `/signal-change` (signals), `/feature-module` (firmware feature), `/handoff` (end of session, workspace only).
- Token budget is limited: delegate bulk mechanical work to `sonnet`/`haiku` subagents, use `Explore` for broad searches, one task per session.

