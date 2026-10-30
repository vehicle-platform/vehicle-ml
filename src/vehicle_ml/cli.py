"""CLI for the vehicle-ml data layer."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from vehicle_ml.data.store import KINDS, SessionStore
from vehicle_ml.dataset import build_dataset, write_dataset


def _parse_bands(text: str | None) -> dict[str, list[tuple[float, float]]]:
    if not text:
        return {}
    bands: dict[str, list[tuple[float, float]]] = {}
    current: str | None = None
    for item in text.split(","):
        if ":" in item:
            channel, ranges = item.split(":", 1)
            if not channel or not ranges:
                raise ValueError(f"invalid band {item!r}; expected channel:low-high")
            current = channel
            bands.setdefault(current, [])
        else:
            if current is None:
                raise ValueError(f"invalid band {item!r}; expected channel:low-high")
            ranges = item
        for value in ranges.split(","):
            if "-" not in value:
                raise ValueError(f"invalid band {value!r}; expected low-high")
            low, high = value.split("-", 1)
            bands[current].append((float(low), float(high)))
    return bands


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vehicle-ml")
    commands = parser.add_subparsers(dest="command", required=True)

    dataset = commands.add_parser("dataset", help="build a windowed feature dataset")
    dataset.add_argument("--data-dir", required=True)
    dataset.add_argument("--out", required=True)
    dataset.add_argument("--kinds", default="telemetry,imu")
    dataset.add_argument("--channels", default="all")
    dataset.add_argument("--window-ms", type=int, default=2_000)
    dataset.add_argument("--stride-ms", type=int, default=1_000)
    dataset.add_argument("--max-gap-ms", type=int, default=500)
    dataset.add_argument("--seed", type=int, default=0)
    dataset.add_argument("--bands", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        kinds = [kind.strip() for kind in args.kinds.split(",") if kind.strip()]
        unknown = [kind for kind in kinds if kind not in KINDS]
        if unknown:
            raise ValueError(f"unknown kinds: {', '.join(unknown)}")
        channels = (
            None
            if args.channels == "all"
            else [name.strip() for name in args.channels.split(",") if name.strip()]
        )
        if channels == []:
            raise ValueError("--channels must name at least one channel or be 'all'")
        bands = _parse_bands(args.bands)
        store = SessionStore(args.data_dir)
        result = build_dataset(
            args.data_dir,
            args.out,
            kinds=kinds,
            channels=channels,
            window_ms=args.window_ms,
            stride_ms=args.stride_ms,
            max_gap_ms=args.max_gap_ms,
            seed=args.seed,
            bands=bands,
        )
        write_dataset(result, args.out, store.sessions(), kinds)
    except ValueError as exc:
        print(f"vehicle-ml: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"vehicle-ml: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        f"wrote {len(result.features)} windows to {args.out} "
        f"({len(result.split['train'])} train sessions)"
    )
    return 0
