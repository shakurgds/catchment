from datetime import date

import numpy as np
import pytest

from catchment.sentinel2 import (
    Observation,
    Scene,
    footprint_stats,
    latest_versions,
    mgrs_tile,
    neighbour_zone,
    parse_listing,
    reflectance_offset,
    tile_prefix,
)
from catchment.water import build_windows, change, load_points, pick, span_text


@pytest.mark.parametrize(
    "lon, lat, tile",
    [
        (45.4939, 3.12222, "38NNJ"),  # centroid of a real 38NNJ scene
        (47.50, 6.00, "38NQM"),
        (47.00, 8.80, "38PQQ"),
        (48.60, 5.30, "39NTF"),
        (49.18, 11.28, "39PUN"),  # Bosaaso
        (42.54, -0.36, "38MKE"),  # Kismaayo, southern hemisphere
        (41.60, -1.50, "37MGU"),
    ],
)
def test_mgrs_tile(lon, lat, tile):
    assert mgrs_tile(lon, lat) == tile


def test_neighbour_zone_tile():
    # lon 42.0 is the west edge of zone 38; its zone-38 square is a sliver
    # Sentinel-2 does not produce, so the zone-37 tile is used instead.
    assert mgrs_tile(42.0, -0.2) == "38MJE"
    assert neighbour_zone(42.0) == 37
    assert mgrs_tile(42.0, -0.2, zone=37) == "37MHV"
    assert neighbour_zone(47.9) == 39


def test_tile_prefix():
    assert tile_prefix("38NNJ", 2026, 9) == "sentinel-s2-l2a-cogs/38/N/NJ/2026/9/"
    assert tile_prefix("07XAB", 2026, 12) == "sentinel-s2-l2a-cogs/7/X/AB/2026/12/"


def test_parse_listing_and_versions():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>sentinel-cogs</Name>
<IsTruncated>true</IsTruncated><NextContinuationToken>abc==</NextContinuationToken>
<CommonPrefixes><Prefix>p/S2A_38NQM_20260912_0_L2A/</Prefix></CommonPrefixes>
<CommonPrefixes><Prefix>p/S2A_38NQM_20260912_2_L2A/</Prefix></CommonPrefixes>
<CommonPrefixes><Prefix>p/S2B_38NQM_20260915_0_L2A/</Prefix></CommonPrefixes>
<CommonPrefixes><Prefix>p/not-a-scene/</Prefix></CommonPrefixes></ListBucketResult>"""
    prefixes, token = parse_listing(xml)
    assert token == "abc=="
    names = [p.rstrip("/").rsplit("/", 1)[-1] for p in prefixes]
    assert latest_versions(names) == ["S2A_38NQM_20260912_2_L2A", "S2B_38NQM_20260915_0_L2A"]


def test_reflectance_offset():
    band = {"scale": 0.0001, "offset": -0.1}
    assert reflectance_offset({"earthsearch:boa_offset_applied": True}, band) == 0.0
    assert reflectance_offset({"earthsearch:boa_offset_applied": False}, band) == -0.1
    assert reflectance_offset({"s2:processing_baseline": "05.10"}, {}) == -0.1
    assert reflectance_offset({"s2:processing_baseline": "03.01"}, {}) == 0.0


def _bands(shape=(6, 6)):
    """Dry soil everywhere: bright SWIR, so MNDWI and NDWI are negative."""
    green = np.full(shape, 900, "uint16")
    nir = np.full(shape, 2000, "uint16")
    swir = np.full(shape, 2600, "uint16")
    scl = np.full(shape, 5, "uint8")
    return green, nir, swir, scl


def test_footprint_stats_water_cloud_and_nodata():
    green, nir, swir, scl = _bands()
    green[2:4, 2:4], nir[2:4, 2:4], swir[2:4, 2:4] = 700, 300, 150  # a small pond
    scl[0, :] = 9  # cloud along the top row
    green[5, 5] = 0  # no data
    fp = np.ones(green.shape, bool)
    total, valid, clear, water, max_m, max_n = footprint_stats(green, nir, swir, scl, fp, 0.0001, 0.0, 0.0, 0.0)
    assert (total, valid, clear, water) == (36, 35, 29, 4)
    assert max_m == pytest.approx((700 - 150) / (700 + 150))
    assert max_n == pytest.approx((700 - 300) / (700 + 300))


def test_footprint_stats_offset_matters():
    # With the ESA +1000 offset still in the numbers, dark water looks like soil
    # unless the -0.1 offset is applied.
    green, nir, swir, scl = (np.full((2, 2), v, t) for v, t in
                             ((1600, "uint16"), (1350, "uint16"), (1150, "uint16"), (6, "uint8")))
    fp = np.ones((2, 2), bool)
    assert footprint_stats(green, nir, swir, scl, fp, 0.0001, -0.1, 0.0, 0.0)[3] == 4
    assert footprint_stats(green, nir, swir, scl, fp, 0.0001, -0.1, 0.0, 0.5)[3] == 0


def _obs(day, clear, water, scene=None, fp=20):
    return Observation(scene or f"S2A_38NQM_{day:%Y%m%d}_0_L2A", day, fp, fp, clear, water, None, None)


def test_pick_newest_clear_view():
    obs = [_obs(date(2026, 10, 1), 20, 0), _obs(date(2026, 10, 4), 18, 3), _obs(date(2026, 10, 6), 2, 0)]
    status, o = pick(obs, 0.6, 1)
    assert status == "water" and o.day == date(2026, 10, 4)


def test_pick_water_through_broken_cloud_and_no_view():
    assert pick([_obs(date(2026, 10, 6), 5, 2)], 0.6, 1)[0] == "water"
    assert pick([_obs(date(2026, 10, 6), 5, 0)], 0.6, 1) == ("no_view", None)
    assert pick([], 0.6, 1) == ("no_view", None)


@pytest.mark.parametrize(
    "now, before, a_now, a_before, expected",
    [
        ("water", "dry", 300, 0, "filled"),
        ("dry", "water", 0, 500, "dried"),
        ("dry", "dry", 0, 0, "still_dry"),
        ("water", "water", 2000, 1000, "rising"),
        ("water", "water", 1000, 2000, "falling"),
        ("water", "water", 1100, 1000, "steady"),
        ("water", "water", 300, 100, "steady"),  # +200 % but only 2 pixels: noise
        ("water", "water", 400, 100, "rising"),
        ("water", "no_view", 500, 0, "unknown"),
    ],
)
def test_change(now, before, a_now, a_before, expected):
    assert change(now, before, a_now, a_before, 0.3) == expected


def test_windows():
    now, before = build_windows(date(2026, 10, 6), 15)
    assert (now.start, now.end) == (date(2026, 9, 22), date(2026, 10, 6))
    assert (before.start, before.end) == (date(2026, 9, 7), date(2026, 9, 21))
    assert now.text() == "22 Sep – 6 Oct 2026"
    assert before.text() == "7–21 September 2026"
    assert span_text(date(2025, 12, 25), date(2026, 1, 8)) == "25 Dec 2025 – 8 Jan 2026"


def test_load_points_csv(tmp_path):
    f = tmp_path / "pts.csv"
    f.write_text("ID,Name,Type,Longitude,Latitude\nb1,Xero,Berkad,45.1,6.2\nd1,Dam,dam,44.0,4.0\nx,,berkad,,\n")
    pts = load_points({"file": str(f), "id_field": "ID", "name_field": "Name", "type_field": "Type",
                       "radius_m": {"default": 30, "berkad": 25, "dam": 60}})
    assert list(pts["id"]) == ["b1", "d1"]
    assert list(pts["radius_m"]) == [25, 60]
    assert (pts.geometry.x.iloc[0], pts.geometry.y.iloc[0]) == (45.1, 6.2)


def test_load_points_numbers_rows_and_rejects_duplicate_ids(tmp_path):
    f = tmp_path / "pts.csv"
    f.write_text("lon,lat\n45,6\n46,7\n")
    pts = load_points({"file": str(f), "id_field": "id", "name_field": "name"})  # columns absent: defaults
    assert list(pts["id"]) == ["1", "2"] and list(pts["type"]) == ["water point"] * 2
    f.write_text("id,lon,lat\na,45,6\na,46,7\n")
    with pytest.raises(ValueError, match="not unique"):
        load_points({"file": str(f), "id_field": "id"})


def test_load_points_polygons(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    f = tmp_path / "ponds.geojson"
    gpd.GeoDataFrame({"pid": ["p1"]}, geometry=[box(45.0, 6.0, 45.001, 6.001)], crs="EPSG:4326").to_file(f)
    pts = load_points({"file": str(f), "id_field": "pid"})
    assert pts.geometry.iloc[0].geom_type == "Polygon"


def test_process_tile_offline(tmp_path):
    """Two synthetic scenes on a local 'tile': the older is clear and dry, the
    newer shows a pond at the point.  The newer one decides 'now'."""
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import Point

    from catchment.water import process_tile

    epsg, x0, y0 = 32638, 600000.0, 700020.0
    lon, lat = 45.9, 6.33  # inside the synthetic grid below
    from pyproj import Transformer

    px, py = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
    x0, y0 = px - 400, py + 400  # 80 x 80 pixels at 10 m around the point

    def write(folder, pond: bool):
        folder.mkdir(parents=True)
        g, n, s, c = _bands((80, 80))
        if pond:
            g[38:42, 38:42], n[38:42, 38:42] = 700, 300
            s[38:42, 38:42] = 150  # 20 m pixels 19-20 after the [::2, ::2] below
        for name, arr, res in (("B03", g, 10), ("B08", n, 10), ("B11", s[::2, ::2], 20), ("SCL", c[::2, ::2], 20)):
            with rasterio.open(folder / f"{name}.tif", "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0],
                               count=1, dtype=arr.dtype, crs=f"EPSG:{epsg}", transform=from_origin(x0, y0, res, res),
                               nodata=0) as ds:
                ds.write(arr, 1)

    old, new = tmp_path / "old", tmp_path / "new"
    write(old, pond=False)
    write(new, pond=True)

    class FakeIndex:
        def scenes(self, tile, start, end):
            return [Scene("S2A_38NQN_20261005_0_L2A", tile, date(2026, 10, 5), f"{new}/", 10.0, epsg, 0.0001, 0.0),
                    Scene("S2B_38NQN_20260920_0_L2A", tile, date(2026, 9, 20), f"{old}/", 10.0, epsg, 0.0001, 0.0)]

    pts = gpd.GeoDataFrame({"id": ["p1"], "radius_m": [25.0]}, geometry=[Point(lon, lat)], crs="EPSG:4326")
    windows = build_windows(date(2026, 10, 6), 15)
    obs, used = process_tile("38NQN", pts, windows, FakeIndex(), {})
    assert [u["period"] for u in used] == ["now", "before"]
    now_status, o = pick(obs[("p1", "now")], 0.6, 1)
    before_status, _ = pick(obs[("p1", "before")], 0.6, 1)
    assert (now_status, before_status) == ("water", "dry")
    assert o.water_px == 16 and o.clear_px == o.valid_px
