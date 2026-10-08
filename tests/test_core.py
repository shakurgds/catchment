from datetime import date, datetime, timezone

import numpy as np
import pytest

from catchment.boundaries import upstream_ids
from catchment.gfs import Field, find_cumulative_apcp, parse_idx, parse_run, valid_forecast_hour
from catchment.periods import build_periods, default_start, range_text
from catchment.pipeline import period_field

RUN = datetime(2026, 10, 4, 0, tzinfo=timezone.utc)

IDX = """\
595:425900000:d=2026100400:PRATE:surface:186-192 hour ave fcst:
596:421790356:d=2026100400:APCP:surface:186-192 hour acc fcst:
597:422171728:d=2026100400:APCP:surface:0-8 day acc fcst:
598:422600000:d=2026100400:ACPCP:surface:186-192 hour acc fcst:
"""


def test_find_cumulative_apcp_day_units():
    rec = find_cumulative_apcp(parse_idx(IDX), 192)
    assert rec.number == 597
    assert (rec.start, rec.end) == (422171728, 422600000)


def test_find_cumulative_apcp_hour_units_last_record():
    idx = "596:427126665:d=2026100400:APCP:surface:24-27 hour acc fcst:\n597:427441577:d=2026100400:APCP:surface:0-27 hour acc fcst:\n"
    rec = find_cumulative_apcp(parse_idx(idx), 27)
    assert rec.start == 427441577 and rec.end is None


def test_missing_record_raises():
    with pytest.raises(LookupError):
        find_cumulative_apcp(parse_idx(IDX), 48)


def test_valid_forecast_hours():
    assert valid_forecast_hour(97)
    assert not valid_forecast_hour(121)
    assert valid_forecast_hour(189)
    assert not valid_forecast_hour(246)
    assert valid_forecast_hour(252)


def test_parse_run():
    assert parse_run("2026-10-04T00") == RUN
    assert parse_run("2026100400") == RUN
    with pytest.raises(ValueError):
        parse_run("2026-10-04T03")


def test_periods_match_posted_maps():
    start = default_start(RUN)
    assert start == date(2026, 10, 5)
    periods = build_periods(RUN, start, 7)
    assert [p.kind for p in periods] == ["daily"] * 7 + ["weekly"]
    wed = periods[2]
    assert wed.label == "Wednesday 7 October 2026"
    assert (wed.start_fh, wed.end_fh) == (72, 96)
    assert periods[-1].label == "Weekly Total: Monday 5 October to Sunday 11 October 2026"
    assert (periods[-1].start_fh, periods[-1].end_fh) == (24, 192)
    assert range_text(start, 7) == "5-11 October 2026"


def test_periods_east_africa_time():
    periods = build_periods(RUN, date(2026, 10, 5), 7, utc_offset_hours=3)
    assert (periods[0].start_fh, periods[-1].end_fh) == (21, 189)


def test_period_before_run_rejected():
    with pytest.raises(ValueError):
        build_periods(RUN, date(2026, 10, 3), 2)


def test_daily_fields_sum_to_weekly_total():
    rng = np.random.default_rng(0)
    lats, lons = np.arange(3.0), np.arange(4.0)
    cum, total = {}, np.zeros((3, 4))
    for fh in range(24, 193, 24):
        total = total + (rng.random((3, 4)) * 10 if fh > 24 else rng.random((3, 4)) * 5)
        cum[fh] = Field(lats, lons, total.copy())
    periods = build_periods(RUN, date(2026, 10, 5), 7)
    dailies = sum(period_field(cum, p).values for p in periods[:-1])
    np.testing.assert_allclose(dailies, period_field(cum, periods[-1]).values)


def test_upstream_ids():
    # 1 <- 2 <- 3, 4 -> 2, 5 drains elsewhere
    next_down = {1: 0, 2: 1, 3: 2, 4: 2, 5: 9}
    assert upstream_ids(next_down, 2) == {2, 3, 4}
    assert upstream_ids(next_down, 1) == {1, 2, 3, 4}


def test_build_basins_from_hydrobasins_zip(tmp_path):
    import zipfile

    import geopandas as gpd
    from shapely.geometry import box

    from catchment.boundaries import build_basins

    # Three stacked cells draining south: 3 -> 2 -> 1 (outlet), plus 4 elsewhere.
    hb = gpd.GeoDataFrame(
        {"HYBAS_ID": [1, 2, 3, 4], "NEXT_DOWN": [0, 1, 2, 0]},
        geometry=[box(0, 0, 1, 1), box(0, 1, 1, 2), box(0, 2, 1, 3), box(5, 5, 6, 6)],
        crs="EPSG:4326",
    )
    shp_dir = tmp_path / "hb"
    shp_dir.mkdir()
    hb.to_file(shp_dir / "hybas_test.shp")
    zip_path = tmp_path / "cache" / "static" / "hybas_test.zip"
    zip_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in shp_dir.iterdir():
            zf.write(f, f.name)

    cfg = {"url": "https://example.invalid/hybas_test.zip", "outlets": [{"name": "Test", "lon": 0.5, "lat": 1.5}]}
    prov = []
    basins = build_basins(cfg, tmp_path / "cache", box(0, 0, 1, 1.5), prov)
    assert list(basins["name"]) == ["Test"]
    # Cells 2 and 3 are upstream of the outlet; the part inside the "country" is removed.
    assert basins.geometry.iloc[0].bounds == pytest.approx((0, 1.5, 1, 3))
    assert prov[0]["layer"] == "hydrobasins"


def test_animation_frames_and_timing(tmp_path):
    from PIL import Image

    from catchment.animate import write_gif, write_mp4

    pngs = []
    for i, colour in enumerate(["red", "green", "blue"]):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (101, 120), colour).save(p)  # odd width: MP4 must crop to even
        pngs.append(p)
    gif = Image.open(write_gif(pngs, tmp_path / "a.gif", width=50, hold_s=1.0, final_hold_s=2.0))
    durations = []
    for i in range(gif.n_frames):
        gif.seek(i)
        durations.append(gif.info["duration"])
    assert durations == [1000, 1000, 2000]

    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(write_mp4(pngs, tmp_path / "a.mp4", fps=10, hold_s=1.0, final_hold_s=2.0)))
    meta = next(reader)
    assert meta["size"] == (100, 120)
    assert sum(1 for _ in reader) == 40


def test_olofsson_perfect_map_returns_mapped_area():
    from catchment.accuracy import olofsson

    r = olofsson([[300, 0], [0, 100]], [9000.0, 1000.0])
    crop = r["classes"]["crop"]
    assert crop["estimated_area"] == pytest.approx(1000.0)
    assert crop["area_se"] == pytest.approx(0.0)
    assert r["overall_accuracy"] == pytest.approx(1.0)


def test_olofsson_hand_computed():
    from catchment.accuracy import olofsson

    # 10 of 100 crop-mapped samples are not crop; 6 of 300 non-crop samples are crop.
    r = olofsson([[294, 6], [10, 90]], [9000.0, 1000.0])
    crop, other = r["classes"]["crop"], r["classes"]["not_crop"]
    # p(crop) = 0.9 * 6/300 + 0.1 * 90/100 = 0.108
    assert crop["estimated_area"] == pytest.approx(1080.0)
    assert crop["estimated_area"] + other["estimated_area"] == pytest.approx(10000.0)
    assert crop["users_accuracy"] == pytest.approx(0.9)
    assert crop["producers_accuracy"] == pytest.approx(0.09 / 0.108)
    assert r["overall_accuracy"] == pytest.approx(0.9 * 294 / 300 + 0.09)
    se = 10000 * np.sqrt(0.81 * 0.02 * 0.98 / 299 + 0.01 * 0.9 * 0.1 / 99)
    assert crop["area_se"] == pytest.approx(se)


def test_sample_allocation_keeps_minimum_for_rare_class():
    from catchment.accuracy import sample_allocation

    alloc = sample_allocation([62_000_000, 1_000_000], (0.95, 0.75), 0.01, 100)
    assert alloc == [390, 100]


def test_holdout_metrics():
    from catchment.accuracy import holdout_metrics

    m = holdout_metrics([[80, 10], [20, 90]])  # rows predicted, cols label
    assert m["crop_precision"] == pytest.approx(90 / 110)
    assert m["crop_recall"] == pytest.approx(0.9)
    assert m["overall_accuracy"] == pytest.approx(0.85)


def test_cropland_config_and_asset_ids():
    from catchment.cropland import BANDS, asset_ids, load_config

    cfg = load_config("config/cropland_somalia.yaml")
    ids = asset_ids(cfg, 2024)
    assert ids["map"].endswith("/somalia_cropland/map_2024")
    assert BANDS[0] == "A00" and BANDS[-1] == "A63" and len(BANDS) == 64
    assert cfg["sources"]["country"]["sha256"] == load_config("config/somalia.yaml")["sources"]["country"]["sha256"]
