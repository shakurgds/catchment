"""Static vector layers: borders, regions, rivers and upstream basins.

Every layer is downloaded once into the cache.  When the config pins a
``sha256`` the file is verified, so a rerun months later either uses exactly
the same geometry or fails loudly.
"""

from __future__ import annotations

import hashlib
import logging
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import Point, box
from shapely.ops import unary_union

log = logging.getLogger(__name__)


@dataclass
class Layers:
    country: gpd.GeoDataFrame  # Somalia outline (single row)
    regions: gpd.GeoDataFrame  # admin-1, column "name"
    neighbours: gpd.GeoDataFrame  # other countries, column "name"
    rivers: gpd.GeoDataFrame
    basins: gpd.GeoDataFrame  # column "name"; may be empty
    ocean: gpd.GeoDataFrame
    provenance: list[dict]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str, dest: Path, sha256: str | None = None, timeout: int = 300) -> dict:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading %s", url)
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    digest = _sha256(dest)
    if sha256 and digest != sha256:
        raise RuntimeError(
            f"{dest} has sha256 {digest} but the config pins {sha256}. The upstream file changed; "
            "review it and update the pin if the new version is acceptable."
        )
    return {"url": url, "file": str(dest), "sha256": digest}


def upstream_ids(next_down: dict[int, int], outlet: int) -> set[int]:
    """All HydroBASINS ids draining into ``outlet`` (inclusive)."""
    children: dict[int, list[int]] = defaultdict(list)
    for basin, down in next_down.items():
        children[down].append(basin)
    seen, queue = {outlet}, deque([outlet])
    while queue:
        for child in children[queue.popleft()]:
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return seen


def _read_hydrobasins(zip_path: Path) -> gpd.GeoDataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        shp = next(n for n in zf.namelist() if n.endswith(".shp"))
    return gpd.read_file(f"zip://{zip_path}!{shp}")


def build_basins(cfg: dict, cache: Path, country, provenance: list[dict]) -> gpd.GeoDataFrame:
    """Trace each configured outlet upstream through HydroBASINS."""
    if not cfg or not cfg.get("enabled", True):
        return gpd.GeoDataFrame({"name": []}, geometry=[], crs="EPSG:4326")

    key = hashlib.sha256(repr((cfg.get("url"), cfg.get("outlets"))).encode()).hexdigest()[:12]
    derived = cache / "static" / f"basins_{key}.geojson"
    if cfg.get("file"):  # user-supplied polygons win
        provenance.append({"layer": "basins", "file": cfg["file"], "sha256": _sha256(Path(cfg["file"]))})
        return gpd.read_file(cfg["file"]).to_crs("EPSG:4326")

    try:
        zip_path = cache / "static" / Path(cfg["url"]).name
        provenance.append({"layer": "hydrobasins", **fetch(cfg["url"], zip_path, cfg.get("sha256"))})
    except (requests.RequestException, RuntimeError) as exc:
        if cfg.get("required"):
            raise
        log.warning("Basins skipped: %s", exc)
        return gpd.GeoDataFrame({"name": []}, geometry=[], crs="EPSG:4326")

    if not derived.exists():
        hb = _read_hydrobasins(zip_path)
        next_down = dict(zip(hb["HYBAS_ID"].astype(int), hb["NEXT_DOWN"].astype(int)))
        rows = []
        for item in cfg["outlets"]:
            pt = Point(item["lon"], item["lat"])
            hit = hb[hb.contains(pt)]
            if hit.empty:
                raise ValueError(f"Outlet {item['name']} ({pt}) is not inside any HydroBASINS polygon")
            ids = upstream_ids(next_down, int(hit.iloc[0]["HYBAS_ID"]))
            rows.append({"name": item["name"], "geometry": unary_union(hb[hb["HYBAS_ID"].isin(ids)].geometry)})
        gpd.GeoDataFrame(rows, crs="EPSG:4326").to_file(derived, driver="GeoJSON")

    basins = gpd.read_file(derived)
    if cfg.get("upstream_only", True):  # keep only the part outside Somalia
        basins["geometry"] = basins.geometry.difference(country)
    return basins


def load_layers(cfg: dict, cache: Path, bbox: tuple[float, float, float, float]) -> Layers:
    cache = Path(cache)
    static = cache / "static"
    prov: list[dict] = []
    src = cfg["sources"]

    def get(layer: str) -> gpd.GeoDataFrame:
        spec = src[layer]
        dest = static / spec.get("filename", Path(spec["url"]).name)
        prov.append({"layer": layer, **fetch(spec["url"], dest, spec.get("sha256"))})
        return gpd.read_file(dest).to_crs("EPSG:4326")

    west, south, east, north = bbox
    frame = box(west - 2, south - 2, east + 2, north + 2)

    country_gdf = get("country")
    country = unary_union(country_gdf.geometry)

    regions = get("regions").rename(columns={src["regions"].get("name_field", "shapeName"): "name"})
    regions = regions[["name", "geometry"]]

    world = get("world")
    world = world[world.intersects(frame)].copy()
    name_field = src["world"].get("name_field", "ADMIN")
    exclude = set(src["world"].get("exclude", []))
    neighbours = world[~world[name_field].isin(exclude)].rename(columns={name_field: "name"})[["name", "geometry"]]
    neighbours["geometry"] = neighbours.geometry.difference(country)

    land = unary_union(list(world.geometry) + [country])
    ocean = gpd.GeoDataFrame(geometry=[frame.difference(land)], crs="EPSG:4326")

    rivers = get("rivers")
    keep = set(src["rivers"].get("names", []))
    if keep:
        rivers = rivers[rivers[src["rivers"].get("name_field", "name")].isin(keep)]
    rivers = rivers[rivers.intersects(frame)]

    basins = build_basins(cfg.get("basins", {}), cache, country, prov)

    return Layers(
        country=gpd.GeoDataFrame(geometry=[country], crs="EPSG:4326"),
        regions=regions,
        neighbours=neighbours,
        rivers=rivers,
        basins=basins,
        ocean=ocean,
        provenance=prov,
    )
