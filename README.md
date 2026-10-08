# catchment

Reproducible GFS rainfall forecast maps for Somalia and the upstream Juba and
Shabelle basins, and a 10 m cropland map of Somalia (see
[Cropland map](#cropland-map-google-satellite-embedding)). One command takes a GFS cycle and produces seven daily maps,
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

## Cropland map (Google Satellite Embedding)

`catchment cropland` maps cropland across Somalia at 10 m in Google Earth
Engine, using the annual [Satellite Embedding](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)
dataset (AlphaEarth Foundations). Each pixel has 64 numbers that summarise a
whole year of optical, radar and other observations, so a random forest on
those numbers separates fields from rangeland, bush and bare ground. You do
not need to build seasonal composites or cloud masks.

```bash
pip install -e '.[cropland]'
earthengine authenticate
# set earthengine.project and asset_root in config/cropland_somalia.yaml (or EE_PROJECT)

catchment cropland samples   --wait   # training points  -> asset samples_<year>
catchment cropland classify  --wait   # probability + map -> asset map_<year> (+ GeoTIFF on Drive)
catchment cropland stats     --wait   # mapped hectares per region -> asset areas_<year> (+ CSV)
catchment cropland reference --wait   # random points to label by eye -> Drive CSV
catchment cropland assess --reference labelled.csv   # accuracy and estimated area
```

Each step sends batch tasks to Earth Engine and writes a record to
`outputs/cropland/<year>/<step>.json`. The record holds the full config, the
task ids, the boundary checksums and, for `classify`, the hold-out scores.

**Training labels.** Training points come from pixels where ESA WorldCover
2021 and the year's Dynamic World labels **agree**. A pixel is cropland when
both call it cropland and not cropland when neither does. The points are
spread over every admin-1 region (200 cropland and 400 other points per
region by default), so the riverine farms along the Juba and Shabelle do not
crowd out the rainfed sorghum areas of Bay and Bakool or the Gabiley plain.
Both products tend to miss small rainfed fields in Somalia, so add your own
points under `labels.points`: a CSV with `lon`, `lat` and `crop` (1 or 0).
They are merged with the consensus points. This is the best way to improve
the map.

**Map.** The `crop_prob` band holds the crop probability (0–100). The
`cropland` band is 1 where `crop_prob` is at least `classifier.threshold`,
with patches smaller than `min_patch_pixels` removed.

**How good is it, and how much cropland is there?** The hold-out score in
`classify.json` is measured on consensus pixels. Those are the easy cases, so
it overstates accuracy. Counting the map's pixels does not give a correct
area either, because every map has errors. For a defensible number:

1. `reference` draws a stratified random sample from the finished map. It
   gives cropland at least 100 points, because cropland is a small share of
   the country. The sample size comes from `reference.*`, following Olofsson
   et al. (2014). It writes two CSVs: `plotid, lon, lat` for the interpreters
   and a separate key with the map class, so the interpreters cannot see the
   map's answer.
2. Label each point (1 = crop, 0 = not crop) on high-resolution imagery, for
   example in Collect Earth Online or QGIS with Google or Bing basemaps. Add a
   `crop` column.
3. `assess --reference that.csv` gives user's and producer's accuracy and an
   estimated cropland area with a 95 % confidence interval. The results go to
   `outputs/cropland/<year>/area_estimate.csv` and `assess.json`.

Use the estimated area, not the pixel count, when you report how much
cropland Somalia has. To map another year, use `--year` (2017 onwards).

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
