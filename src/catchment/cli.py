"""Command-line entry point: ``catchment forecast`` and ``catchment water``."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from .pipeline import run_forecast
from .water import run_water


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

    wa = sub.add_parser("water", help="Check water points and ponds for open water with Sentinel-2")
    wa.add_argument("--config", type=Path, default=Path("config/water.yaml"))
    wa.add_argument("--points", type=Path, help="Water points file (CSV with lon/lat, GeoJSON, GPKG, SHP)")
    wa.add_argument("--date", type=date.fromisoformat, help="Last day of the period (YYYY-MM-DD); default: yesterday")
    wa.add_argument("--out", type=Path, default=Path("outputs"))
    wa.add_argument("--cache", type=Path, default=Path("cache"))
    wa.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    if not args.verbose:  # GDAL reports every HTTP retry; our own log says when a scene is skipped
        logging.getLogger("rasterio").setLevel(logging.ERROR)
    if args.command == "water":
        out = run_water(args.config, args.date, args.points, args.out, args.cache)
    else:
        out = run_forecast(args.config, args.run, args.start, args.out, args.cache)
    print(out)


if __name__ == "__main__":
    main()
