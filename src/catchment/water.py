"""Are water points and seasonal ponds filling?  Sentinel-2 water index per point.

For every berkad, dam, pond or other water point in the inventory, the most
recent cloud-free Sentinel-2 view in each period is checked for open water
(MNDWI and NDWI, see :func:`catchment.sentinel2.footprint_stats`).  Two periods
are compared, the latest ``window_days`` and the same length before it, so
each point gets a status now (water / dry / no clear view) and a change
(filled, rising, steady, falling, dried up, still dry).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml
from shapely.geometry import Point
from shapely.ops import transform as shp_transform

from . import __version__
from .boundaries import load_layers
from .plotting import Context, PointClass, render_points_map
from .sentinel2 import Observation, SceneIndex, SceneReader, mgrs_tile, neighbour_zone, to_utm, zone_edge_distance

log = logging.getLogger(__name__)

STATUS = [
    PointClass("water", "Holding water", "#1f5fa8"),
    PointClass("dry", "Dry", "#c8742b"),
    PointClass("no_view", "No clear view (cloud)", "#8d95a0", hollow=True),
]
CHANGE = [
    PointClass("filled", "Filled", "#1f5fa8"),
    PointClass("rising", "Rising", "#6fb1e0"),
    PointClass("steady", "Steady", "#8796ab"),
    PointClass("falling", "Falling", "#f0a860"),
    PointClass("dried", "Dried up", "#b2470f"),
    PointClass("still_dry", "Still dry", "#dccfb4"),
    PointClass("unknown", "No comparison", "#8d95a0", hollow=True),
]
LONS = ("lon", "longitude", "long", "x", "lng")
LATS = ("lat", "latitude", "y")


# --- Config and inputs ------------------------------------------------------


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: Path) -> dict:
    """YAML config; ``extends: other.yaml`` (relative to this file) is loaded first."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    if cfg.get("extends"):
        cfg = deep_merge(load_config(path.parent / cfg.pop("extends")), cfg)
    return cfg


def _pick(columns, wanted: str | None, candidates: tuple[str, ...], what: str) -> str:
    if wanted:
        if wanted not in columns:
            raise ValueError(f"Column '{wanted}' ({what}) is not in the points file; columns are {list(columns)}")
        return wanted
    lower = {c.lower(): c for c in columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    raise ValueError(f"Could not find a {what} column; name it in water.points (columns are {list(columns)})")


def load_points(pcfg: dict) -> gpd.GeoDataFrame:
    """Water points as a GeoDataFrame with columns id, name, type, radius_m, geometry (EPSG:4326).

    CSV/TSV files need longitude and latitude columns; any vector format
    GeoPandas reads (GeoJSON, GeoPackage, shapefile ...) works as is.  Polygon
    features (e.g. mapped pond outlines) are used as their own footprint.
    """
    path = Path(pcfg["file"])
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else None, engine="python", encoding="utf-8-sig")
        lon = _pick(df.columns, pcfg.get("lon_field"), LONS, "longitude")
        lat = _pick(df.columns, pcfg.get("lat_field"), LATS, "latitude")
        df[lon], df[lat] = pd.to_numeric(df[lon], errors="coerce"), pd.to_numeric(df[lat], errors="coerce")
        bad = df[lon].isna() | df[lat].isna()
        if bad.any():
            log.warning("Skipping %d rows without valid coordinates", int(bad.sum()))
            df = df[~bad]
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs=pcfg.get("crs", "EPSG:4326"))
    else:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(pcfg.get("crs", "EPSG:4326"))
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].to_crs("EPSG:4326")
    gdf = gdf.explode(index_parts=False) if (gdf.geom_type == "MultiPoint").any() else gdf

    def col(key: str, default):
        field = pcfg.get(key)
        if field and field not in gdf.columns:
            log.warning("Column '%s' (%s) is not in the points file; using defaults", field, key)
            field = None
        return gdf[field].astype(str) if field else default

    out = gpd.GeoDataFrame(
        {
            "id": col("id_field", pd.Series([str(i + 1) for i in range(len(gdf))], index=gdf.index)),
            "name": col("name_field", ""),
            "type": col("type_field", pcfg.get("default_type", "water point")),
        },
        geometry=gdf.geometry.values,
        crs="EPSG:4326",
    ).reset_index(drop=True)
    out["type"] = out["type"].fillna("").str.strip()
    radius = {str(k).lower(): float(v) for k, v in (pcfg.get("radius_m") or {}).items()}
    default = radius.get("default", 30.0)
    out["radius_m"] = [radius.get(t.lower(), default) for t in out["type"]]
    if out["id"].duplicated().any():
        raise ValueError("Point ids are not unique; fix the id column or leave id_field empty to number rows")
    return out


@dataclass(frozen=True)
class Window:
    name: str  # "now" or "before"
    start: date
    end: date

    def text(self) -> str:
        return span_text(self.start, self.end)


def span_text(a: date, b: date) -> str:
    if (a.year, a.month) == (b.year, b.month):
        return f"{a.day}–{b.day} {b:%B %Y}"
    if a.year == b.year:
        return f"{a.day} {a:%b} – {b.day} {b:%b %Y}"
    return f"{a.day} {a:%b %Y} – {b.day} {b:%b %Y}"


def build_windows(end: date, days: int) -> tuple[Window, Window]:
    now = Window("now", end - timedelta(days=days - 1), end)
    before = Window("before", now.start - timedelta(days=days), now.start - timedelta(days=1))
    return now, before


# --- Status and change ------------------------------------------------------


def pick(obs: list[Observation], min_clear: float, min_water_px: int) -> tuple[str, Observation | None]:
    """Status from a period's observations (any order): the newest view clear
    enough to judge decides; failing that, water seen through broken cloud
    still counts as water."""
    obs = sorted(obs, key=lambda o: (o.day, o.scene), reverse=True)
    for o in obs:
        if o.clear_fraction >= min_clear:
            return ("water" if o.water_px >= min_water_px else "dry"), o
    for o in obs:
        if o.clear_px and o.water_px >= min_water_px:
            return "water", o
    return "no_view", None


def change(now: str, before: str, area_now: float, area_before: float, rel: float) -> str:
    if "no_view" in (now, before):
        return "unknown"
    if now == "water" and before == "dry":
        return "filled"
    if now == "dry" and before == "water":
        return "dried"
    if now == "dry":
        return "still_dry"
    if area_now > area_before * (1 + rel) and area_now - area_before >= 300:
        return "rising"
    if area_now < area_before * (1 - rel) and area_before - area_now >= 300:
        return "falling"
    return "steady"


# --- Reading Sentinel-2 -----------------------------------------------------


GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_HTTP_RETRY_CODES": "ALL",
    "VSI_CACHE": "TRUE",
    # Do not remember failed requests between opens, so a retry really retries.
    "CPL_VSIL_CURL_NON_CACHED": "/vsicurl/https://sentinel-cogs.s3.us-west-2.amazonaws.com/",
}


def gdal_env(wcfg: dict) -> dict:
    env = {**GDAL_ENV, "GDAL_CACHEMAX": int(wcfg.get("gdal_cache_mb", 512))}
    # Behind a TLS-inspecting proxy GDAL needs the same CA bundle as requests.
    ca = os.environ.get("CURL_CA_BUNDLE") or os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    if ca:
        env["CURL_CA_BUNDLE"] = ca
    return env


def process_tile(tile: str, pts: gpd.GeoDataFrame, windows: tuple[Window, Window], index: SceneIndex,
                 wcfg: dict) -> tuple[dict, list[dict]]:
    """Read every scene of one tile, newest first, until each point has a clear
    view in both periods.  Returns {(point id, window name): [Observation]} and
    the scenes that were read."""
    import rasterio

    min_clear = float(wcfg.get("min_clear_fraction", 0.6))
    max_cloud = float(wcfg.get("max_scene_cloud", 95))
    first, last = min(w.start for w in windows), max(w.end for w in windows)
    scenes = [s for s in index.scenes(tile, first, last) if s.cloud_cover is None or s.cloud_cover <= max_cloud]
    obs: dict[tuple[str, str], list[Observation]] = {}
    done: set[tuple[str, str]] = set()
    used = []
    footprints: dict[int, list] = {}  # epsg -> footprints in that UTM zone, aligned with pts

    with rasterio.Env(**gdal_env(wcfg)):
        for scene in scenes:
            win = next(w for w in windows if w.start <= scene.day <= w.end)
            pending = [i for i, pid in enumerate(pts["id"]) if (pid, win.name) not in done]
            if not pending:
                continue
            remaining = list(pending)  # points of this scene not read yet
            n_read, tries = 0, int(wcfg.get("read_retries", 4))
            for attempt in range(tries):
                try:
                    with SceneReader(scene) as reader:
                        if reader.epsg not in footprints:
                            tr = to_utm(reader.epsg).transform
                            footprints[reader.epsg] = [
                                shp_transform(tr, g).buffer(r) if isinstance(g, Point) else shp_transform(tr, g)
                                for g, r in zip(pts.geometry, pts["radius_m"])
                            ]
                        fps = footprints[reader.epsg]
                        while remaining:
                            i = remaining[0]
                            o = reader.read(fps[i], wcfg)
                            remaining.pop(0)
                            if o is None:
                                continue
                            n_read += 1
                            key = (pts["id"].iloc[i], win.name)
                            obs.setdefault(key, []).append(o)
                            if o.clear_fraction >= min_clear:
                                done.add(key)
                    break
                except Exception as exc:  # noqa: BLE001 - network hiccups; one bad scene must not stop the run
                    if attempt == tries - 1:
                        log.warning("Skipping %d points in scene %s: %s", len(remaining), scene.id, exc)
                    else:
                        log.debug("Retrying scene %s after %s", scene.id, exc)
                        time.sleep(2 ** (attempt + 1))
            if not n_read:
                continue
            used.append({"scene": scene.id, "date": scene.day.isoformat(), "cloud_cover": scene.cloud_cover,
                         "period": win.name, "points_read": n_read, "url": scene.url})
            log.debug("%s: %d points read, %d/%d resolved", scene.id, n_read, len(done), 2 * len(pts))
    return obs, used


def read_tiles(points: gpd.GeoDataFrame, windows: tuple[Window, Window], index: SceneIndex,
               wcfg: dict) -> tuple[dict, list[dict]]:
    """Run :func:`process_tile` for every tile of ``points`` in parallel."""
    obs: dict = {}
    used: list[dict] = []
    tiles = sorted(points["tile"].unique())
    log.info("%d points on %d Sentinel-2 tiles", len(points), len(tiles))
    with ThreadPoolExecutor(max_workers=int(wcfg.get("workers", 8))) as pool:
        futures = {pool.submit(process_tile, t, points[points["tile"] == t].reset_index(drop=True), windows, index,
                               wcfg): t for t in tiles}
        for k, fut in enumerate(as_completed(futures), 1):
            tile_obs, tile_used = fut.result()
            obs.update(tile_obs)
            used += tile_used
            log.info("Tile %s done (%d/%d), %d scenes read", futures[fut], k, len(tiles), len(tile_used))
    return obs, used


# --- Outputs ----------------------------------------------------------------


def region_summary(points: gpd.GeoDataFrame) -> list[dict]:
    rows = []
    for region, g in points.groupby("region", sort=True, dropna=False):
        seen = int((g["status"] != "no_view").sum())
        water = int((g["status"] == "water").sum())
        rows.append({
            "region": region if isinstance(region, str) else "",
            "points": len(g),
            "seen": seen,
            "water": water,
            "dry": int((g["status"] == "dry").sum()),
            "no_view": int((g["status"] == "no_view").sum()),
            "pct_water": round(100 * water / seen, 1) if seen else None,
            "filled": int((g["change"] == "filled").sum()),
            "rising": int((g["change"] == "rising").sum()),
            "falling": int((g["change"] == "falling").sum()),
            "dried": int((g["change"] == "dried").sum()),
        })
    return rows


def pct(points: gpd.GeoDataFrame, column: str) -> float:
    seen = points[column] != "no_view"
    return 100 * float((points.loc[seen, column] == "water").mean()) if seen.any() else 0.0


def run_water(config_path: Path, end: date | None, points_file: Path | None, out_root: Path, cache: Path) -> Path:
    cfg = load_config(config_path)
    wcfg = cfg.get("water", {})
    pcfg = dict(wcfg.get("points", {}))
    if points_file:
        pcfg["file"] = str(points_file)
    if not pcfg.get("file"):
        raise ValueError("No points file: pass --points or set water.points.file in the config")
    points = load_points(pcfg)
    log.info("%d water points from %s", len(points), pcfg["file"])

    # Sentinel-2 data reach the bucket within about a day, so end yesterday by default.
    end = end or (datetime.now(timezone.utc).date() - timedelta(days=1))
    days = int(wcfg.get("window_days", 15))
    windows = build_windows(end, days)
    now, before = windows
    log.info("Now: %s, before: %s", now.text(), before.text())

    centre = points.geometry.apply(lambda g: g if isinstance(g, Point) else g.representative_point())
    points["lon"], points["lat"] = centre.x.round(6), centre.y.round(6)
    points["tile"] = [mgrs_tile(x, y) for x, y in zip(points["lon"], points["lat"])]
    index = SceneIndex(cache, wcfg.get("bucket_url", "https://sentinel-cogs.s3.us-west-2.amazonaws.com"))
    obs, scenes_used = read_tiles(points, windows, index, wcfg)

    # Near a UTM zone edge the point's own tile may not exist or not reach it;
    # try the overlapping tile of the neighbouring zone for points with no data.
    blind = [i for i, (pid, lon) in enumerate(zip(points["id"], points["lon"]))
             if zone_edge_distance(lon) < 1.0
             and not any(o.valid_px for w in windows for o in obs.get((pid, w.name), []))]
    if blind:
        log.info("%d points near a UTM zone edge had no data; trying the neighbouring zone's tiles", len(blind))
        for i in blind:
            points.loc[i, "tile"] = mgrs_tile(points.loc[i, "lon"], points.loc[i, "lat"],
                                              zone=neighbour_zone(points.loc[i, "lon"]))
        retry_obs, retry_used = read_tiles(points.loc[blind], windows, index, wcfg)
        obs.update(retry_obs)
        scenes_used += retry_used

    min_clear = float(wcfg.get("min_clear_fraction", 0.6))
    min_water = int(wcfg.get("min_water_pixels", 1))
    rel = float(wcfg.get("area_change", 0.3))
    rows = []
    for pid in points["id"]:
        row = {}
        for w in windows:
            status, o = pick(obs.get((pid, w.name), []), min_clear, min_water)
            row[f"status_{w.name}"] = status
            row[f"date_{w.name}"] = o.day.isoformat() if o else ""
            row[f"scene_{w.name}"] = o.scene if o else ""
            row[f"clear_pct_{w.name}"] = round(100 * o.clear_fraction) if o else None
            row[f"water_m2_{w.name}"] = o.water_px * 100 if o else None
            row[f"mndwi_max_{w.name}"] = round(o.max_mndwi, 3) if o and o.max_mndwi is not None else None
            row[f"ndwi_max_{w.name}"] = round(o.max_ndwi, 3) if o and o.max_ndwi is not None else None
        row["change"] = change(row["status_now"], row["status_before"], row["water_m2_now"] or 0,
                               row["water_m2_before"] or 0, rel)
        rows.append(row)
    points = pd.concat([points, pd.DataFrame(rows)], axis=1)
    points["status"] = points["status_now"]

    layers = load_layers(cfg, cache, tuple(cfg["bbox"]))
    regions = layers.regions[["name", "geometry"]].rename(columns={"name": "region"})
    joined = gpd.sjoin(gpd.GeoDataFrame(geometry=gpd.points_from_xy(points["lon"], points["lat"]), crs="EPSG:4326"), regions,
                       how="left", predicate="within")
    points["region"] = joined[~joined.index.duplicated()]["region"].reindex(points.index).fillna("")

    out_dir = out_root / f"water_{end:%Y%m%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rename = cfg.get("labels", {}).get("region_names", {})
    summary = region_summary(points)
    for r in summary:
        r["region"] = rename.get(r["region"], r["region"]) or "(outside the admin-1 regions)"

    columns = ["id", "name", "type", "region", "lon", "lat", "tile", "radius_m", "status_now", "status_before",
               "change"] + [f"{k}_{w}" for w in ("now", "before") for k in
                            ("date", "scene", "clear_pct", "water_m2", "mndwi_max", "ndwi_max")]
    points["region"] = points["region"].map(lambda r: rename.get(r, r))
    points[columns].to_csv(out_dir / "water_points.csv", index=False)
    points[columns + ["geometry"]].to_file(out_dir / "water_points.geojson", driver="GeoJSON")
    with open(out_dir / "region_summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    # Maps
    seen_now = int((points["status_now"] != "no_view").sum())
    n_water = int((points["status_now"] == "water").sum())
    counts = points["change"].value_counts()
    timeline = [("Before", pct(points, "status_before"), False), ("Now", pct(points, "status_now"), True)]
    named = [r for r in summary if not r["region"].startswith("(")]
    ranked = sorted((r for r in named if r["pct_water"] is not None and r["seen"] >= 3), key=lambda r: -r["pct_water"])
    source = ("Copernicus Sentinel-2 L2A (ESA), AWS Open Data  ·  MNDWI & NDWI, cloud-masked with SCL  ·  "
              "geoBoundaries, Natural Earth")
    mcfg = {**cfg, "kicker": wcfg.get("kicker", "SURFACE WATER")}
    maps = []
    ctx = Context(
        timeline=timeline,
        top_regions=[(r["region"], r["pct_water"]) for r in ranked[:6]],
        headline=now.text(),
        subhead=f"{n_water:,} of {seen_now:,} water points seen clearly hold open water  ·  latest clear Sentinel-2 view",
    )
    render_points_map(points, "status_now", STATUS, layers, mcfg, ctx, source, out_dir / "water_status.png",
                      chart=("MOST POINTS WITH WATER", "% of points seen clearly"))
    maps.append("water_status.png")

    filling = int(counts.get("filled", 0) + counts.get("rising", 0))
    drying = int(counts.get("dried", 0) + counts.get("falling", 0))
    by_fill = sorted((r for r in named if r["filled"] + r["rising"]),
                     key=lambda r: -(r["filled"] + r["rising"]))
    ctx = Context(
        timeline=timeline,
        top_regions=[(r["region"], float(r["filled"] + r["rising"])) for r in by_fill[:6]],
        headline="Filling or drying?",
        subhead=f"{now.text()} against {before.text()}  ·  {filling:,} filling, {drying:,} drying",
    )
    render_points_map(points, "change", CHANGE, layers, mcfg, ctx, source, out_dir / "water_change.png",
                      chart=("MOST POINTS FILLING", "points filled or rising"))
    maps.append("water_change.png")
    log.info("Wrote maps to %s", out_dir)

    top = ", ".join(f"{r['region']} {r['pct_water']:.0f}%" for r in ranked[:3]) or "none yet"
    caption = wcfg.get("caption", "").format(
        period=now.text(), before=before.text(), water=f"{n_water:,}", seen=f"{seen_now:,}", total=f"{len(points):,}",
        filling=f"{filling:,}", drying=f"{drying:,}", top=top,
    )
    (out_dir / "caption.txt").write_text(f"{caption}\n\n{wcfg.get('hashtags', '')}\n".strip() + "\n")

    points_path = Path(pcfg["file"])
    manifest = {
        "catchment_version": __version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reproduce": f"catchment water --config {config_path} --points {pcfg['file']} --date {end.isoformat()}",
        "periods": {w.name: {"start": w.start.isoformat(), "end": w.end.isoformat()} for w in windows},
        "config": {"path": str(config_path), "content": cfg},
        "points": {"file": str(points_path), "count": len(points),
                   "sha256": hashlib.sha256(points_path.read_bytes()).hexdigest() if points_path.exists() else None},
        "sentinel2_scenes": sorted(scenes_used, key=lambda s: (s["scene"], s["period"])),
        "static_inputs": layers.provenance,
        "products": maps + ["water_points.csv", "water_points.geojson", "region_summary.csv", "caption.txt"],
        "environment": {
            "python": platform.python_version(),
            **{pkg: version(pkg) for pkg in ("numpy", "matplotlib", "geopandas", "shapely", "rasterio", "pyproj")},
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out_dir

