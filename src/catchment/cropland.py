"""Cropland map of Somalia from Google's Satellite Embedding (AlphaEarth Foundations).

Each 10 m pixel of ``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`` carries a
64-number summary of a whole year of Sentinel-1/2, Landsat and other
observations, so a simple classifier on those numbers separates cropland
from rangeland, bush and bare ground without building seasonal composites.

The work runs in Earth Engine as five steps.  Each submits batch tasks (no
5-minute limit) and writes a JSON record to ``outputs/cropland/<year>/``:

1. ``samples``   training points: where existing land-cover products agree,
                 spread across every region, plus any points you labelled
2. ``classify``  random forest on the 64 bands -> crop probability and a
                 cropland/not-cropland map (Earth Engine asset, optional GeoTIFF)
3. ``stats``     mapped cropland area per admin-1 region
4. ``reference`` a stratified random sample of the map to label by eye
5. ``assess``    accuracy and an unbiased cropland area with a 95 % interval
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

from . import __version__
from .accuracy import holdout_metrics, olofsson, sample_allocation
from .boundaries import fetch

log = logging.getLogger(__name__)

EMBEDDING = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
BANDS = [f"A{i:02d}" for i in range(64)]
STEPS = ("samples", "classify", "stats", "reference", "assess")

try:
    import ee
except ImportError:  # the rainfall maps do not need Earth Engine
    ee = None


def load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def asset_ids(cfg: dict, year: int) -> dict[str, str]:
    root = cfg["earthengine"]["asset_root"].rstrip("/")
    return {name: f"{root}/{name}_{year}" for name in ("samples", "map", "areas")}


def init(cfg: dict, project: str | None) -> None:
    if ee is None:
        raise SystemExit("Earth Engine is not installed: pip install 'catchment[cropland]'")
    project = project or os.environ.get("EE_PROJECT") or cfg["earthengine"].get("project")
    if not project or "YOUR-" in project:
        raise SystemExit("Set earthengine.project in the config, EE_PROJECT, or --project")
    ee.Initialize(project=project)
    root = cfg["earthengine"]["asset_root"].rstrip("/")
    try:
        ee.data.getAsset(root)
    except ee.EEException:
        log.info("Creating asset folder %s", root)
        ee.data.createAsset({"type": "FOLDER"}, root)


# --- inputs ---------------------------------------------------------------


def boundaries(cfg: dict, cache: Path) -> tuple[ee.FeatureCollection, ee.FeatureCollection, list[dict]]:
    """Somalia outline and admin-1 regions (pinned geoBoundaries files) as Earth Engine features."""
    prov, out = [], {}
    for layer in ("country", "regions"):
        spec = cfg["sources"][layer]
        dest = Path(cache) / "static" / Path(spec["url"]).name
        prov.append({"layer": layer, **fetch(spec["url"], dest, spec.get("sha256"))})
        gdf = gpd.read_file(dest).to_crs("EPSG:4326")
        gdf["name"] = gdf[spec["name_field"]] if spec.get("name_field") else layer
        out[layer] = ee.FeatureCollection(json.loads(gdf[["name", "geometry"]].to_json(drop_id=True)))
    return out["country"], out["regions"], prov


def embeddings(year: int, region: ee.Geometry) -> ee.Image:
    col = ee.ImageCollection(EMBEDDING).filterDate(f"{year}-01-01", f"{year + 1}-01-01").filterBounds(region)
    if col.size().getInfo() == 0:
        raise SystemExit(f"{EMBEDDING} has no images for {year}")
    return col.mosaic().select(BANDS)


def product_crop_mask(spec: dict, year: int, region: ee.Geometry) -> ee.Image:
    """1 where a land-cover product calls the pixel cropland, 0 where it does not."""
    col = ee.ImageCollection(spec["collection"]).filterBounds(region)
    if spec.get("same_year"):
        col = col.filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    col = col.select(spec["band"])
    img = col.mode() if spec.get("reduce") == "mode" else col.mosaic()
    crop = ee.Image(0)
    for value in spec["crop_values"]:
        crop = crop.Or(img.eq(value))
    return crop.updateMask(img.mask())


def consensus_labels(products: list[dict], year: int, region: ee.Geometry) -> ee.Image:
    """``crop`` = 1 where every product says cropland, 0 where none does; masked where they disagree."""
    masks = [product_crop_mask(p, year, region).rename(f"p{i}") for i, p in enumerate(products)]
    votes = ee.Image.cat(masks).reduce(ee.Reducer.sum())
    agree_crop, agree_other = votes.eq(len(products)), votes.eq(0)
    return agree_crop.rename("crop").toByte().updateMask(agree_crop.Or(agree_other))


def read_points(path: Path) -> ee.FeatureCollection:
    """Labelled points from CSV (lon, lat, crop) or any vector file with a ``crop`` column."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    else:
        gdf = gpd.read_file(path).to_crs("EPSG:4326")
    if "crop" not in gdf or not gdf["crop"].isin([0, 1]).all():
        raise ValueError(f"{path} needs a 'crop' column of 0 (not crop) / 1 (crop)")
    gdf["crop"] = gdf["crop"].astype(int)
    keep = ["crop", "geometry"] + [c for c in ("plotid",) if c in gdf]
    return ee.FeatureCollection(json.loads(gdf[keep].to_json(drop_id=True)))


# --- steps ----------------------------------------------------------------


def step_samples(cfg: dict, year: int, cache: Path) -> dict:
    country, regions, prov = boundaries(cfg, cache)
    aoi = country.geometry()
    t = cfg["training"]
    emb = embeddings(year, aoi)
    stack = emb.addBands(consensus_labels(cfg["labels"]["products"], year, aoi))

    def per_region(f):
        pts = stack.stratifiedSample(
            numPoints=0,
            classBand="crop",
            region=f.geometry(),
            scale=t["scale"],
            seed=t["seed"],
            classValues=[0, 1],
            classPoints=[t["points_per_region"]["not_crop"], t["points_per_region"]["crop"]],
            geometries=True,
            tileScale=t["tile_scale"],
        )
        return pts.map(lambda p: p.set({"region": f.get("name"), "source": "consensus"}))

    samples = regions.map(per_region).flatten()
    if cfg["labels"].get("points"):
        own = emb.sampleRegions(
            collection=read_points(cfg["labels"]["points"]), properties=["crop"], scale=t["scale"], geometries=True
        ).map(lambda p: p.set({"region": "", "source": "user"}))
        samples = samples.merge(own)
        prov.append({"layer": "user_points", "file": str(cfg["labels"]["points"])})
    samples = samples.randomColumn("split", t["seed"])

    ids = asset_ids(cfg, year)
    task = ee.batch.Export.table.toAsset(collection=samples, description=f"cropland_samples_{year}", assetId=ids["samples"])
    task.start()
    return {"tasks": [task.id], "outputs": {"samples": ids["samples"]}, "inputs": prov}


def train(cfg: dict, year: int) -> tuple[ee.Classifier, dict]:
    t, c = cfg["training"], cfg["classifier"]
    table = ee.FeatureCollection(asset_ids(cfg, year)["samples"])
    fit = table.filter(ee.Filter.lt("split", t["train_fraction"]))
    hold = table.filter(ee.Filter.gte("split", t["train_fraction"]))
    clf = (
        ee.Classifier.smileRandomForest(
            numberOfTrees=c["trees"],
            minLeafPopulation=c["min_leaf_population"],
            bagFraction=c["bag_fraction"],
            seed=t["seed"],
        )
        .setOutputMode("PROBABILITY")
        .train(features=fit, classProperty="crop", inputProperties=BANDS)
    )
    thr = c["threshold"]
    scored = hold.classify(clf, "prob").map(lambda f: f.set("pred", ee.Number(f.get("prob")).gte(thr)))
    # errorMatrix rows are the actual label; transpose to rows = predicted.
    cm = scored.errorMatrix("crop", "pred", [0, 1]).array().transpose().getInfo()
    counts = table.aggregate_histogram("crop").getInfo()
    return clf, {"holdout": holdout_metrics(cm), "holdout_matrix_pred_by_label": cm, "training_labels": counts}


def step_classify(cfg: dict, year: int, cache: Path) -> dict:
    country, _, prov = boundaries(cfg, cache)
    aoi = country.geometry()
    c, x = cfg["classifier"], cfg["export"]
    clf, metrics = train(cfg, year)
    log.info("Hold-out (consensus labels): %s", metrics["holdout"])

    prob = embeddings(year, aoi).classify(clf, "prob")
    crop = prob.gte(c["threshold"])
    mmu = int(c.get("min_patch_pixels", 1))
    if mmu > 1:  # drop specks smaller than the minimum patch
        crop = crop.And(crop.connectedPixelCount(mmu, True).gte(mmu))
    image = (
        ee.Image.cat(prob.multiply(100).round().toByte().rename("crop_prob"), crop.toByte().rename("cropland"))
        .clipToCollection(country)
        .set({"year": year, "threshold": c["threshold"], "embedding": EMBEDDING})
    )

    ids = asset_ids(cfg, year)
    common = dict(image=image, region=aoi.bounds(), scale=x["scale"], crs=x["crs"], maxPixels=1e13)
    tasks = [
        ee.batch.Export.image.toAsset(
            description=f"cropland_map_{year}",
            assetId=ids["map"],
            pyramidingPolicy={"crop_prob": "mean", "cropland": "mode"},
            **common,
        )
    ]
    if x.get("drive_folder"):
        tasks.append(
            ee.batch.Export.image.toDrive(
                description=f"cropland_geotiff_{year}",
                folder=x["drive_folder"],
                fileNamePrefix=f"somalia_cropland_{year}",
                formatOptions={"cloudOptimized": True},
                **common,
            )
        )
    for task in tasks:
        task.start()
    return {"tasks": [t.id for t in tasks], "outputs": {"map": ids["map"]}, "metrics": metrics, "inputs": prov}


def step_stats(cfg: dict, year: int, cache: Path) -> dict:
    _, regions, prov = boundaries(cfg, cache)
    ids = asset_ids(cfg, year)
    cropland = ee.Image(ids["map"]).select("cropland")
    ha = ee.Image.pixelArea().divide(1e4)
    areas = ha.multiply(cropland).rename("crop_ha").addBands(ha.updateMask(cropland.mask()).rename("mapped_ha"))
    table = areas.reduceRegions(
        collection=regions, reducer=ee.Reducer.sum(), scale=cfg["export"]["scale"], tileScale=cfg["training"]["tile_scale"]
    ).map(lambda f: f.setGeometry(None))
    tasks = [ee.batch.Export.table.toAsset(collection=table, description=f"cropland_areas_{year}", assetId=ids["areas"])]
    if cfg["export"].get("drive_folder"):
        tasks.append(
            ee.batch.Export.table.toDrive(
                collection=table,
                description=f"cropland_areas_csv_{year}",
                folder=cfg["export"]["drive_folder"],
                fileNamePrefix=f"somalia_cropland_areas_{year}",
                selectors=["name", "crop_ha", "mapped_ha"],
            )
        )
    for task in tasks:
        task.start()
    return {"tasks": [t.id for t in tasks], "outputs": {"areas": ids["areas"]}, "inputs": prov}


def mapped_areas(cfg: dict, year: int) -> list[float]:
    """Mapped [not crop, crop] hectares for Somalia, from the ``stats`` table."""
    table = ee.FeatureCollection(asset_ids(cfg, year)["areas"])
    crop, total = ee.List([table.aggregate_sum("crop_ha"), table.aggregate_sum("mapped_ha")]).getInfo()
    return [total - crop, crop]


def step_reference(cfg: dict, year: int, cache: Path) -> dict:
    country, _, prov = boundaries(cfg, cache)
    r = cfg["reference"]
    areas = mapped_areas(cfg, year)
    alloc = sample_allocation(areas, r["expected_users_accuracy"], r["target_se_overall"], r["min_per_class"])
    cropland = ee.Image(asset_ids(cfg, year)["map"]).select("cropland")
    points = cropland.rename("map_class").stratifiedSample(
        numPoints=0,
        classBand="map_class",
        region=country.geometry(),
        scale=cfg["export"]["scale"],
        seed=r["seed"],
        classValues=[0, 1],
        classPoints=alloc,
        geometries=True,
        tileScale=cfg["training"]["tile_scale"],
    )
    points = points.randomColumn("order", r["seed"]).sort("order")  # shuffle so interpreters cannot see the strata

    def number(pair):
        pair = ee.List(pair)
        f = ee.Feature(pair.get(0))
        xy = f.geometry().coordinates()
        return f.set({"plotid": pair.get(1), "lon": xy.get(0), "lat": xy.get(1)})

    shuffled = points.toList(points.size())
    points = ee.FeatureCollection(shuffled.zip(ee.List.sequence(1, shuffled.size())).map(number))
    folder = cfg["export"].get("drive_folder") or "catchment"
    tasks = [
        ee.batch.Export.table.toDrive(
            collection=points,
            description=f"cropland_reference_{year}",
            folder=folder,
            fileNamePrefix=f"somalia_cropland_reference_{year}",
            selectors=["plotid", "lon", "lat"],
        ),
        ee.batch.Export.table.toDrive(  # keep the map class apart from the interpreters' sheet
            collection=points,
            description=f"cropland_reference_key_{year}",
            folder=folder,
            fileNamePrefix=f"somalia_cropland_reference_key_{year}",
            selectors=["plotid", "map_class"],
        ),
    ]
    for task in tasks:
        task.start()
    return {
        "tasks": [t.id for t in tasks],
        "allocation": {"not_crop": alloc[0], "crop": alloc[1]},
        "mapped_ha": {"not_crop": areas[0], "crop": areas[1]},
        "inputs": prov,
    }


def step_assess(cfg: dict, year: int, cache: Path, reference: Path, out_dir: Path) -> dict:
    """Score labelled reference points (lon, lat, crop) against the map."""
    cropland = ee.Image(asset_ids(cfg, year)["map"]).select("cropland")
    pts = read_points(reference)
    sampled = cropland.sampleRegions(collection=pts, properties=["crop"], scale=cfg["export"]["scale"])
    cm = sampled.errorMatrix("crop", "cropland", [0, 1]).array().transpose().getInfo()  # rows = map
    n_in = pts.size().getInfo()
    n_used = sum(map(sum, cm))
    if n_used < n_in:
        log.warning("%d of %d reference points fall outside the map and were skipped", n_in - n_used, n_in)
    areas = mapped_areas(cfg, year)
    result = olofsson(cm, areas)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "area_estimate.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "mapped_ha", "estimated_ha", "ci95_low_ha", "ci95_high_ha",
                    "users_accuracy", "users_ci95", "producers_accuracy", "producers_ci95"])
        for name, c in result["classes"].items():
            w.writerow([name, round(c["mapped_area"]), round(c["estimated_area"]), round(c["area_ci95"][0]),
                        round(c["area_ci95"][1]), round(c["users_accuracy"], 3), round(c["users_accuracy_ci95"], 3),
                        round(c["producers_accuracy"], 3), round(c["producers_accuracy_ci95"], 3)])
    crop = result["classes"]["crop"]
    log.info(
        "Cropland %.0f ha (95%% CI %.0f-%.0f); mapped %.0f ha. Overall accuracy %.3f ± %.3f",
        crop["estimated_area"], *crop["area_ci95"], crop["mapped_area"],
        result["overall_accuracy"], result["overall_accuracy_ci95"],
    )
    return {"reference": str(reference), "points_used": n_used, "matrix_map_by_ref": cm, "estimate": result}


# --- runner ---------------------------------------------------------------


def wait_for(task_ids: list[str], poll: int = 30) -> None:
    pending = set(task_ids)
    while pending:
        for status in ee.data.getTaskStatus(list(pending)):
            if status["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
                pending.discard(status["id"])
                log.info("%s %s %s", status.get("description"), status["state"], status.get("error_message", ""))
                if status["state"] != "COMPLETED":
                    raise SystemExit(f"Earth Engine task {status['id']} {status['state']}")
        if pending:
            time.sleep(poll)


def run_step(
    step: str,
    config_path: Path,
    year: int | None,
    out_root: Path,
    cache: Path,
    project: str | None = None,
    reference: Path | None = None,
    wait: bool = False,
) -> Path:
    cfg = load_config(config_path)
    year = year or cfg["year"]
    init(cfg, project)
    out_dir = Path(out_root) / "cropland" / str(year)
    if step == "samples":
        record = step_samples(cfg, year, cache)
    elif step == "classify":
        record = step_classify(cfg, year, cache)
    elif step == "stats":
        record = step_stats(cfg, year, cache)
    elif step == "reference":
        record = step_reference(cfg, year, cache)
    elif step == "assess":
        if not reference:
            raise SystemExit("assess needs --reference: a CSV of plotid, lon, lat, crop (0/1)")
        record = step_assess(cfg, year, cache, reference, out_dir)
    else:
        raise ValueError(step)

    record = {
        "step": step,
        "year": year,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "catchment": __version__,
        "earthengine_api": version("earthengine-api"),
        "config": cfg,
        **record,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{step}.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    for tid in record.get("tasks", []):
        log.info("Started Earth Engine task %s (see https://code.earthengine.google.com/tasks)", tid)
    if wait and record.get("tasks"):
        wait_for(record["tasks"])
    return path
