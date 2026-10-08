"""Sentinel-2 L2A scenes from the AWS Open Data bucket, read point by point.

Scenes are found by listing ``s3://sentinel-cogs`` directly (one prefix per
MGRS tile and month), so no search API is needed.  Each scene folder holds
cloud-optimised GeoTIFFs and a STAC item with the cloud cover and the
scale/offset that turn digital numbers into reflectance.  Only the few pixels
around each water point are read: GDAL fetches the COG blocks that cover the
window and nothing else.
"""

from __future__ import annotations

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date
from functools import cache
from pathlib import Path

import numpy as np
import requests
from pyproj import Transformer

log = logging.getLogger(__name__)

BUCKET_URL = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"
PREFIX = "sentinel-s2-l2a-cogs"
SCENE_RE = re.compile(r"^(S2[A-D])_(\d{2}[C-X][A-Z]{2})_(\d{8})_(\d+)_L2A$")

# Scene classification (SCL) codes.  Cloud, cloud shadow and cirrus hide the
# ground; 0 (no data) and 1 (saturated/defective) are unusable.
SCL_HIDDEN = (3, 8, 9, 10)
SCL_INVALID = (0, 1)

LAT_BANDS = "CDEFGHJKLMNPQRSTUVWX"
COL_SETS = ("STUVWXYZ", "ABCDEFGH", "JKLMNPQR")  # indexed by zone % 3
ROW_LETTERS = "ABCDEFGHJKLMNPQRSTUV"


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


@cache
def to_utm(epsg: int) -> Transformer:
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def mgrs_tile(lon: float, lat: float, zone: int | None = None) -> str:
    """MGRS 100 km square holding a point, e.g. ``38NNJ`` (Sentinel-2 tile id).

    A Sentinel-2 tile starts at the north-west corner of its 100 km square and
    runs 109.8 km east and south, so it always covers its whole square.
    ``zone`` forces a neighbouring UTM zone (see :func:`neighbour_zone`).
    """
    if not -80 <= lat < 84:
        raise ValueError(f"latitude {lat} is outside the UTM/MGRS range")
    zone = zone or int((lon + 180) // 6) + 1
    band = LAT_BANDS[int((lat + 80) // 8)]
    e, n = to_utm((32600 if lat >= 0 else 32700) + zone).transform(lon, lat)
    col_index = int(e // 100_000) - 1
    if not 0 <= col_index < 8:
        raise ValueError(f"({lon}, {lat}) is too far from UTM zone {zone}")
    col = COL_SETS[zone % 3][col_index]
    row = ROW_LETTERS[(int(n // 100_000) + (5 if zone % 2 == 0 else 0)) % 20]
    return f"{zone:02d}{band}{col}{row}"


def neighbour_zone(lon: float) -> int:
    """The adjacent UTM zone on the side of the nearer zone edge.

    Near a zone edge the point's own square can be a sliver that Sentinel-2
    does not produce; the overlapping tile of the next zone covers it instead.
    """
    zone = int((lon + 180) // 6) + 1
    west = (zone - 1) * 6 - 180
    return zone - 1 if lon - west < 3 else zone + 1


def zone_edge_distance(lon: float) -> float:
    """Degrees of longitude to the nearest UTM zone edge."""
    off = (lon + 180) % 6
    return min(off, 6 - off)


@dataclass
class Scene:
    id: str  # e.g. S2C_38NNJ_20260930_0_L2A
    tile: str
    day: date
    url: str  # scene folder, ending in "/"
    cloud_cover: float | None = None
    epsg: int | None = None
    scale: float = 0.0001
    offset: float = 0.0
    item: dict = field(default_factory=dict, repr=False)

    def band(self, name: str) -> str:
        return f"{self.url}{name}.tif"


def tile_prefix(tile: str, year: int, month: int) -> str:
    return f"{PREFIX}/{int(tile[:2])}/{tile[2]}/{tile[3:5]}/{year}/{month}/"


def parse_listing(xml_text: str) -> tuple[list[str], str | None]:
    """Common prefixes and the continuation token from an S3 ListObjectsV2 reply."""
    root = ET.fromstring(xml_text)
    ns = {"s3": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
    q = (lambda t: f"s3:{t}") if ns else (lambda t: t)
    prefixes = [p.text for p in root.iterfind(f"{q('CommonPrefixes')}/{q('Prefix')}", ns)]
    token = root.findtext(q("NextContinuationToken"), default=None, namespaces=ns)
    return prefixes, token


def months(start: date, end: date) -> list[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def latest_versions(names: list[str]) -> list[str]:
    """Keep the highest processing sequence (``_0_``, ``_1_`` ...) per satellite and day."""
    best: dict[tuple[str, str, str], tuple[int, str]] = {}
    for name in names:
        m = SCENE_RE.match(name)
        if not m:
            continue
        key, seq = (m[1], m[2], m[3]), int(m[4])
        if key not in best or seq > best[key][0]:
            best[key] = (seq, name)
    return sorted(name for _, name in best.values())


def reflectance_offset(props: dict, raster_band: dict) -> float:
    """Additive offset that turns scaled digital numbers into reflectance.

    Since processing baseline 04.00 (January 2022) ESA adds 1000 to every
    digital number, i.e. a -0.1 reflectance offset.  Earth Search removes it
    when it converts scenes to COGs and says so with
    ``earthsearch:boa_offset_applied``, yet still lists the -0.1 offset in
    ``raster:bands``, so the flag must be checked first.
    """
    if props.get("earthsearch:boa_offset_applied"):
        return 0.0
    if "offset" in raster_band:
        return float(raster_band["offset"])
    return -0.1 if str(props.get("s2:processing_baseline", "0")) >= "04.00" else 0.0


class SceneIndex:
    """Lists scenes per tile and caches each scene's STAC item on disk."""

    def __init__(self, cache: Path, base_url: str = BUCKET_URL, timeout: int = 60):
        self.cache = Path(cache) / "sentinel2"
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, url: str, params: dict | None = None, tries: int = 5) -> requests.Response:
        """GET with retries.  Busy moments can produce a stray 404/503 even
        for keys that exist, so every failure is retried with a backoff."""
        for attempt in range(tries):
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                if attempt == tries - 1:
                    raise
                log.debug("Retrying %s after %s", url, exc)
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def _list(self, prefix: str) -> list[str]:
        names, token = [], None
        while True:
            params = {"list-type": "2", "prefix": prefix, "delimiter": "/"}
            if token:
                params["continuation-token"] = token
            resp = self._get(f"{self.base}/", params)
            prefixes, token = parse_listing(resp.text)
            names += [p.rstrip("/").rsplit("/", 1)[-1] for p in prefixes]
            if not token:
                return names

    def _item(self, scene_id: str, url: str) -> dict:
        path = self.cache / "items" / f"{scene_id}.json"
        if not path.exists():
            resp = self._get(f"{url}{scene_id}.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(resp.text)
        return json.loads(path.read_text())

    def scenes(self, tile: str, start: date, end: date) -> list[Scene]:
        """Scenes of ``tile`` acquired from ``start`` to ``end`` inclusive, newest first."""
        found = []
        for y, m in months(start, end):
            prefix = tile_prefix(tile, y, m)
            for name in latest_versions(self._list(prefix)):
                day = date(int(name[10:14]), int(name[14:16]), int(name[16:18]))
                if start <= day <= end:
                    found.append(Scene(id=name, tile=tile, day=day, url=f"{self.base}/{prefix}{name}/"))
        for s in found:
            item = self._item(s.id, s.url)
            props = item.get("properties", {})
            s.item = item
            s.cloud_cover = props.get("eo:cloud_cover")
            s.epsg = props.get("proj:epsg") or props.get("proj:code", "EPSG:0").split(":")[-1]
            s.epsg = int(s.epsg) if s.epsg else None
            rb = (item.get("assets", {}).get("green", {}).get("raster:bands") or [{}])[0]
            s.scale = float(rb.get("scale", 0.0001))
            s.offset = reflectance_offset(props, rb)
        return sorted(found, key=lambda s: (s.day, s.id), reverse=True)


# --- Per-footprint water statistics -----------------------------------------


@dataclass
class Observation:
    scene: str
    day: date
    footprint_px: int  # 10 m pixels inside the footprint
    valid_px: int  # of those, with data (inside the orbit swath, not defective)
    clear_px: int  # of those, cloud-free and valid
    water_px: int  # clear pixels classed as water
    max_mndwi: float | None
    max_ndwi: float | None

    @property
    def clear_fraction(self) -> float:
        return self.clear_px / self.footprint_px if self.footprint_px else 0.0


def normalised_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / (a + b)
    return np.where(np.isfinite(out), out, np.nan)


def footprint_stats(
    green: np.ndarray,
    nir: np.ndarray,
    swir: np.ndarray,
    scl: np.ndarray,
    footprint: np.ndarray,
    scale: float,
    offset: float,
    mndwi_min: float,
    ndwi_min: float,
) -> tuple[int, int, int, int, float | None, float | None]:
    """Count valid, clear and water pixels inside ``footprint`` (all arrays on the 10 m grid).

    Raw digital numbers are converted to reflectance first; 0 is no data.
    A pixel is water when both MNDWI (green vs SWIR) and NDWI (green vs NIR)
    pass their thresholds: MNDWI separates water from soil and built-up land,
    NDWI keeps the 10 m detail that small berkads need.
    """
    valid = (green > 0) & (nir > 0) & (swir > 0) & ~np.isin(scl, SCL_INVALID)
    clear = footprint & valid & ~np.isin(scl, SCL_HIDDEN)
    g, n, s = (x.astype("float64") * scale + offset for x in (green, nir, swir))
    mndwi = normalised_difference(g, s)
    ndwi = normalised_difference(g, n)
    water = clear & (mndwi > mndwi_min) & (ndwi > ndwi_min)
    max_m = float(np.nanmax(mndwi[clear])) if clear.any() and np.isfinite(mndwi[clear]).any() else None
    max_n = float(np.nanmax(ndwi[clear])) if clear.any() and np.isfinite(ndwi[clear]).any() else None
    return int(footprint.sum()), int((footprint & valid).sum()), int(clear.sum()), int(water.sum()), max_m, max_n


def upsample2(a: np.ndarray) -> np.ndarray:
    """20 m pixels onto the 10 m grid (each pixel becomes 2 x 2)."""
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)


class SceneReader:
    """Open bands of one scene and read small windows around footprints."""

    BANDS = {"green": "B03", "nir": "B08", "swir": "B11", "scl": "SCL"}

    def __init__(self, scene: Scene):
        import rasterio

        self.scene = scene
        def path(band: str) -> str:
            p = scene.band(band)
            return f"/vsicurl/{p}" if p.startswith(("http://", "https://")) else p

        self.ds = {k: rasterio.open(path(v)) for k, v in self.BANDS.items()}
        ref = self.ds["green"]
        self.transform, self.width, self.height = ref.transform, ref.width, ref.height
        self.epsg = ref.crs.to_epsg()

    def close(self) -> None:
        for ds in self.ds.values():
            ds.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def window(self, bounds: tuple[float, float, float, float]):
        """10 m pixel window covering ``bounds``, snapped to even offsets so that
        the 20 m bands (SWIR, SCL) line up exactly; ``None`` if off the tile."""
        from rasterio.windows import Window

        x0, y0 = self.transform.c, self.transform.f
        res = self.transform.a
        c0 = int(np.floor((bounds[0] - x0) / res)) // 2 * 2
        c1 = -(-int(np.ceil((bounds[2] - x0) / res)) // 2) * 2
        r0 = int(np.floor((y0 - bounds[3]) / res)) // 2 * 2
        r1 = -(-int(np.ceil((y0 - bounds[1]) / res)) // 2) * 2
        c0, r0 = max(c0, 0), max(r0, 0)
        c1, r1 = min(c1, self.width - self.width % 2), min(r1, self.height - self.height % 2)
        if c1 <= c0 or r1 <= r0:
            return None
        return Window(c0, r0, c1 - c0, r1 - r0)

    def read(self, geom_utm, cfg: dict) -> Observation | None:
        from rasterio.features import geometry_mask
        from rasterio.windows import Window

        win = self.window(geom_utm.bounds)
        if win is None:
            return None
        half = Window(win.col_off // 2, win.row_off // 2, win.width // 2, win.height // 2)
        green = self.ds["green"].read(1, window=win)
        nir = self.ds["nir"].read(1, window=win)
        swir = upsample2(self.ds["swir"].read(1, window=half))
        scl = upsample2(self.ds["scl"].read(1, window=half))
        mask = geometry_mask([geom_utm], out_shape=green.shape, transform=self.ds["green"].window_transform(win),
                             invert=True, all_touched=True)
        fp, valid, clear, water, max_m, max_n = footprint_stats(
            green, nir, swir, scl, mask, self.scene.scale, self.scene.offset,
            float(cfg.get("mndwi_min", 0.0)), float(cfg.get("ndwi_min", 0.0)),
        )
        if fp == 0:
            return None
        return Observation(self.scene.id, self.scene.day, fp, valid, clear, water, max_m, max_n)
