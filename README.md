# catchment

Reproducible GFS rainfall forecast maps for Somalia and the upstream Juba and
Shabelle basins, plus a Sentinel-2 check of whether berkads, ponds and other
water points are filling (see [Surface water](#surface-water-at-berkads-and-water-points)).

One command takes a GFS cycle and produces seven daily maps,
a weekly total map, per-region statistics, a ready-to-post caption and a
manifest that records exactly how the maps were made.

```
catchment forecast --run 2026-10-04T00 --start 2026-10-05
```

```
outputs/2026100400Z/
├── daily_2026-10-05.png … daily_2026-10-11.png
├── total_2026-10-05_2026-10-11.png
├── week_overview.png    all seven days and the total on one page
├── week_animation.mp4   day-by-day animation (full size, for LinkedIn)
├── week_animation.gif   the same animation, 720 px, loops
├── region_stats.csv     mean / max rainfall per admin-1 region and period
├── caption.txt          post text with the wettest regions + hashtags
└── manifest.json        inputs, checksums, versions, the command to reproduce
```

## Setup

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install --require-hashes -r requirements.lock   # exact pinned versions
pip install --no-deps -e .
pytest -q
```

## Usage

```bash
catchment forecast                                   # latest 00Z cycle, next Mon-Sun
catchment forecast --run 2026-10-04T00               # a specific cycle
catchment forecast --run 2026-10-04T00 --start 2026-10-06
catchment forecast --config config/my_region.yaml    # another area or style
```

`--run latest` picks the most recent cycle (from `gfs.cycles` in the config)
whose forecast is fully published.

## How it works

1. **GFS precipitation.** Each GFS pgrb2 file holds the running total
   precipitation since the model started (`APCP 0-N hour acc`). The tool reads
   the `.idx` inventory and downloads only that GRIB message with an HTTP
   range request, which is under 1 MB per forecast hour. Daily rainfall is the
   difference between two running totals, so the seven daily maps always add
   up to the weekly map. Data comes from the NOAA Open Data archive on AWS,
   which keeps every past cycle, with NOMADS as a fallback for recent cycles.
2. **Days.** Days run midnight to midnight UTC by default. Set
   `period.utc_offset_hours: 3` to use East Africa Time days.
3. **Boundaries.** Somalia and its regions come from geoBoundaries (gbOpen);
   neighbouring countries and rivers come from Natural Earth v5.1.2. Each file
   is pinned by sha256 in the config, so if an upstream file changes the run
   stops instead of silently drawing different borders.
4. **Basins.** The Juba and Shabelle basins are traced upstream through
   HydroBASINS (level 7) from outlet points at Luuq and Belet Weyne. Only the
   part outside Somalia is drawn. You can use your own polygons instead with
   `basins.file`.
5. **Rendering.** The 0.25° grid is interpolated bilinearly onto a 0.05° grid
   and contoured using the classes in `config/somalia.yaml`. PNG timestamps are
   removed, so the same inputs produce byte-identical images.

## Map design

Each map is 1080 × 1350 px (4:5, sized for LinkedIn and other social feeds).

- **Headline:** the day (or week) is the title. A strip of seven columns shows
  the Somalia-average rainfall for each day, with the current day highlighted.
- **Colour scale:** a single sequential ramp from pale straw through green and
  teal to deep indigo and plum. Each class is darker than the one before, so
  more rain always reads as darker. The 0–2 mm class is left uncoloured.
- **Map:** rainfall is drawn over land only. Somalia is lighter than its
  neighbours, and the border, rivers and labels have halos so they stay
  readable over any colour.
- **Wettest regions:** a ranked chart of area-average rainfall by region sits
  in the empty sea area.
- **Fonts:** IBM Plex Sans is stored in the package (`src/catchment/fonts`,
  SIL Open Font License), so the maps look the same on every machine.

- **Day by day:** `week_overview.png` puts the seven days and the weekly
  total side by side, each with its Somalia average. `week_animation.mp4` and
  `.gif` step through Monday to Sunday (1.6 s per day) and hold the weekly
  total for 3.5 s before looping. The frames are the finished daily maps, so
  the animation always matches the stills. LinkedIn plays the MP4 inline but
  not the GIF. Turn either off under `animation` in the config.

Colours, the brand line in the footer and the chart position can be changed
under `style`, `brand` and `classes` in the config.

## Surface water at berkads and water points

`catchment water` checks every berkad, dam, pond (balli/war) or other water
point in your inventory for open water, using Sentinel-2 satellite images, and
shows whether they are filling or drying.

```
catchment water --points data/water_points.csv                    # 15 days to yesterday
catchment water --points data/water_points.csv --date 2026-10-06
```

```
outputs/water_20261006/
├── water_status.png      holding water / dry / no clear view, with % by region
├── water_change.png      filled, rising, steady, falling, dried up, still dry
├── water_points.csv      one row per point: status, date and scene used, water area
├── water_points.geojson  the same, for QGIS/ArcGIS
├── region_summary.csv    counts and % holding water per admin-1 region
├── caption.txt           post text
└── manifest.json         periods, every Sentinel-2 scene read, inputs, versions
```

**Points file.** A CSV with longitude and latitude columns (`lon`/`lat`,
`longitude`/`latitude` or `x`/`y` are found automatically), or any vector file
(GeoJSON, GeoPackage, shapefile). Optional `id`, `name` and `type` columns;
the column names are set under `water.points` in `config/water.yaml`. Mapped
pond outlines (polygons) are used as their own footprint. See
`data/water_points.example.csv`. Put the national inventory at
`data/water_points.csv` for the weekly workflow.

**How it works.**

1. **Periods.** The latest `window_days` (15) ending on `--date` are compared
   with the 15 days before.
2. **Images.** Sentinel-2 L2A scenes come from the Earth Search COG archive on
   AWS Open Data (`s3://sentinel-cogs`). Scenes are found by listing the
   bucket for each point's MGRS tile, so no search API or account is needed.
   Only the pixels around each point are downloaded.
3. **Footprint.** Each point gets a circle with a radius set by type under
   `water.radius_m` (berkad 25 m, dam/pond 60 m, default 30 m).
4. **Water test.** A 10 m pixel counts as water when MNDWI (green vs SWIR,
   B03/B11) and NDWI (green vs NIR, B03/B08) are both above 0. Cloud, cloud
   shadow and cirrus are masked with the scene classification layer (SCL).
   Over clear dry land this flags about 3 pixels in a million.
5. **Status.** The newest view in each period with at least 60 % of the
   footprint cloud-free decides **holding water** or **dry**. When no view is
   that clear, water seen through gaps in the cloud still counts. Otherwise
   the point has **no clear view**.
6. **Change.** Dry → water is *filled* and water → dry is *dried up*. When a
   point holds water in both periods, a change in water area of 30 % (and at
   least 3 pixels) makes it *rising* or *falling*.

**Limits.** Sentinel-2 pixels are 10 m (20 m for SWIR). Covered berkads, and
open ones smaller than about 10 × 10 m, cannot be seen. Algae or floating
vegetation can hide water. During the Gu and Deyr rains cloud can hide a
point for weeks; such points show as *no clear view*, not as dry. Check
important points on the ground.

**Scale.** A national inventory touches roughly 70 Sentinel-2 tiles. The
pipeline reads tiles in parallel (`workers`) and opens scenes newest first,
stopping as soon as every point in the tile has a clear view, so most tiles
need only a few scenes. `.github/workflows/water-points.yml` runs every
Monday and commits the maps to `maps/water_<date>/`. It skips quietly until
`data/water_points.csv` exists. If the inventory is sensitive, keep the
repository private, because the outputs list every point.

## Reproducing an old map

Every `manifest.json` has a `reproduce` line, the full config, the source URL
and sha256 of each input, and the package versions used. To rebuild a map,
check out the commit that made it, install `requirements.lock`, and run the
`reproduce` command. AWS keeps old GFS cycles, so maps from any past run can
be rebuilt.

## Automation

`.github/workflows/weekly-forecast.yml` runs every Sunday at 06:40 UTC on the
00Z cycle. It installs the locked environment, runs the tests, makes the maps
and commits them to `maps/<run>/` in this repository (for example
`maps/2026100400Z/`), so every week's maps are kept permanently on GitHub.
The same files are also attached to the run as a downloadable zip. You can
start it by hand from the Actions tab with a chosen `run` and `start`.

Scheduled workflows only run from the repository's default branch, so keep
this workflow on that branch.

## First run checklist

- After the first successful run with internet access, copy the HydroBASINS
  sha256 from `manifest.json` (`static_inputs`) into `basins.sha256` in the
  config to pin it.
- Check where the basin and region labels sit, and adjust `labels.*` in the
  config if needed.

## Disclaimer

The maps show raw computer-model guidance, which can change from one run to
the next. They are not warnings. Warnings and advisories for Somalia come from
the national meteorological and disaster-management authorities.
