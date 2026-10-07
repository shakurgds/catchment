"""End-to-end run: GFS cycle -> daily/total fields -> maps, stats, caption, manifest."""

from __future__ import annotations

import csv
import json
import logging
import platform
from dataclasses import asdict
from datetime import date, datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import shapely
import yaml

from . import __version__
from .boundaries import Layers, load_layers
from .gfs import Downloader, Field, latest_run, read_field
from .periods import Period, build_periods, default_start, range_text
from .plotting import Context, render_map, smooth

log = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def period_field(cum: dict[int, Field], p: Period) -> Field:
    end = cum[p.end_fh]
    start = cum[p.start_fh].values if p.start_fh else 0.0
    return Field(lats=end.lats, lons=end.lons, values=np.clip(end.values - start, 0, None))


def area_mean(field: Field, geom, upsample: int) -> float:
    f = smooth(field, upsample)
    lon2d, lat2d = np.meshgrid(f.lons, f.lats)
    vals = f.values[shapely.contains_xy(geom, lon2d, lat2d)]
    return float(vals.mean()) if vals.size else 0.0


def headings(p: Period, dailies: list[Period], start: date, days: int) -> dict:
    week = range_text(start, days)
    if p.kind == "daily":
        d = date.fromisoformat(p.key)
        n = [x.key for x in dailies].index(p.key) + 1
        return {"headline": f"{d:%A} {d.day} {d:%B}", "subhead": f"24-hour rainfall  ·  day {n} of {days}  ·  {week}"}
    return {"headline": week.rsplit(" ", 1)[0], "subhead": f"{days}-day rainfall total  ·  {p.label.split(': ', 1)[-1]}"}


def region_stats(field: Field, layers: Layers, upsample: int) -> list[dict]:
    """Mean and max rainfall per admin-1 region on the interpolated grid."""
    f = smooth(field, upsample)
    lon2d, lat2d = np.meshgrid(f.lons, f.lats)
    rows = []
    for _, r in layers.regions.iterrows():
        mask = shapely.contains_xy(r.geometry, lon2d, lat2d)
        vals = f.values[mask]
        rows.append({
            "region": r["name"],
            "mean_mm": round(float(vals.mean()), 1) if vals.size else None,
            "max_mm": round(float(vals.max()), 1) if vals.size else None,
        })
    return rows


def run_forecast(config_path: Path, run_arg: str, start: date | None, out_root: Path, cache: Path) -> Path:
    cfg = load_config(config_path)
    gcfg = cfg["gfs"]
    pcfg = cfg.get("period", {})
    days, offset = int(pcfg.get("days", 7)), int(pcfg.get("utc_offset_hours", 0))
    dl = Downloader(cache, gcfg["base_urls"])

    if run_arg == "latest":
        # The furthest hour needed depends on the run, so probe with a generous bound.
        run = latest_run(dl, max_fh=24 * (days + 8), cycles=tuple(gcfg.get("cycles", [0, 6, 12, 18])))
    else:
        from .gfs import parse_run

        run = parse_run(run_arg)
    start = start or default_start(run)
    periods = build_periods(run, start, days, offset)
    log.info("Run %s, %d periods from %s", run.isoformat(), len(periods), start)

    bbox = tuple(cfg["bbox"])
    pad = (bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1)
    hours = sorted({h for p in periods for h in (p.start_fh, p.end_fh) if h})
    cum, gfs_prov = {}, []
    for fh in hours:
        path, meta = dl.fetch_cumulative_apcp(run, fh)
        cum[fh] = read_field(path, pad)
        gfs_prov.append(meta)

    layers = load_layers(cfg, cache, bbox)

    out_dir = out_root / run.strftime("%Y%m%d%HZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    upsample = int(cfg["classes"].get("upsample", 1))
    rename = cfg.get("labels", {}).get("region_names", {})

    fields = {p.key: period_field(cum, p) for p in periods}
    stats = {p.key: region_stats(fields[p.key], layers, upsample) for p in periods}
    means = {p.key: area_mean(fields[p.key], layers.country.geometry.iloc[0], upsample) for p in periods}
    dailies = [p for p in periods if p.kind == "daily"]

    products, stats_rows = [], []
    for p in periods:
        ranked = sorted((r for r in stats[p.key] if r["mean_mm"] is not None), key=lambda r: -r["mean_mm"])
        ctx = Context(
            timeline=[(date.fromisoformat(d.key).strftime("%a"), means[d.key], p.kind != "daily" or d.key == p.key) for d in dailies],
            top_regions=[(rename.get(r["region"], r["region"]), r["mean_mm"]) for r in ranked[:6]],
            **headings(p, dailies, start, days),
        )
        png = out_dir / f"{'daily' if p.kind == 'daily' else 'total'}_{p.key.removeprefix('total_')}.png"
        render_map(fields[p.key], layers, cfg, ctx, run, png)
        products.append({**asdict(p), "png": png.name})
        stats_rows.extend({"period": p.key, **row} for row in stats[p.key])
        log.info("Wrote %s", png)

    with open(out_dir / "region_stats.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["period", "region", "mean_mm", "max_mm"])
        writer.writeheader()
        writer.writerows(stats_rows)

    caption = cfg.get("caption", "").format(cycle=run.strftime("%HZ"), run_date=f"{run.day} {run:%B %Y}")
    (out_dir / "caption.txt").write_text(f"{caption}\n\n{cfg.get('hashtags', '')}\n".strip() + "\n")

    manifest = {
        "catchment_version": __version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reproduce": f"catchment forecast --config {config_path} --run {run:%Y-%m-%dT%H} --start {start.isoformat()}",
        "run": run.isoformat(),
        "start": start.isoformat(),
        "days": days,
        "utc_offset_hours": offset,
        "config": {"path": str(config_path), "content": cfg},
        "gfs_inputs": gfs_prov,
        "static_inputs": layers.provenance,
        "products": products,
        "environment": {
            "python": platform.python_version(),
            **{pkg: version(pkg) for pkg in ("numpy", "scipy", "matplotlib", "geopandas", "shapely", "pygrib")},
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out_dir
