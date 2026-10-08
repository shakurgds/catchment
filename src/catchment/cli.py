"""Command-line entry point: ``catchment forecast`` and ``catchment cropland``."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from .pipeline import run_forecast


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="catchment", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fc = sub.add_parser("forecast", help="Make daily and weekly GFS rainfall maps")
    fc.add_argument("--config", type=Path, default=Path("config/somalia.yaml"))
    fc.add_argument("--run", default="latest", help="GFS cycle, e.g. 2026-10-04T00, or 'latest'")
    fc.add_argument("--start", type=date.fromisoformat, help="First day (YYYY-MM-DD); default: next Monday")
    fc.add_argument("--out", type=Path, default=Path("outputs"))
    fc.add_argument("--cache", type=Path, default=Path("cache"))
    fc.add_argument("-v", "--verbose", action="store_true")

    cl = sub.add_parser("cropland", help="Cropland map from Google's Satellite Embedding (Earth Engine)")
    cl.add_argument("step", choices=["samples", "classify", "stats", "reference", "assess"])
    cl.add_argument("--config", type=Path, default=Path("config/cropland_somalia.yaml"))
    cl.add_argument("--year", type=int, help="Embedding year; default: year in the config")
    cl.add_argument("--project", help="Google Cloud project registered for Earth Engine")
    cl.add_argument("--reference", type=Path, help="assess: labelled points CSV (plotid, lon, lat, crop)")
    cl.add_argument("--wait", action="store_true", help="Wait for the Earth Engine tasks to finish")
    cl.add_argument("--out", type=Path, default=Path("outputs"))
    cl.add_argument("--cache", type=Path, default=Path("cache"))
    cl.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "cropland":
        from .cropland import run_step

        out = run_step(args.step, args.config, args.year, args.out, args.cache, args.project, args.reference, args.wait)
    else:
        out = run_forecast(args.config, args.run, args.start, args.out, args.cache)
    print(out)


if __name__ == "__main__":
    main()
