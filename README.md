# catchment

Reproducible GFS rainfall forecast maps for Somalia and the upstream Juba and
Shabelle basins. One command takes a GFS cycle and produces seven daily maps,
a weekly total map, per-region statistics, a ready-to-post caption and a
manifest that records exactly how the maps were made.

```
catchment forecast --run 2026-10-04T00 --start 2026-10-05
```

```
outputs/2026100400Z/
├── daily_2026-10-05.png … daily_2026-10-11.png
├── total_2026-10-05_2026-10-11.png
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

Colours, the brand line in the footer and the chart position can be changed
under `style`, `brand` and `classes` in the config.

## Reproducing an old map

Every `manifest.json` has a `reproduce` line, the full config, the source URL
and sha256 of each input, and the package versions used. To rebuild a map,
check out the commit that made it, install `requirements.lock`, and run the
`reproduce` command. AWS keeps old GFS cycles, so maps from any past run can
be rebuilt.

## Automation

`.github/workflows/weekly-forecast.yml` runs every Sunday at 06:40 UTC on the
00Z cycle. It installs the locked environment, runs the tests, makes the maps
and uploads them as a build artifact. You can also start it by hand from the
Actions tab with a chosen `run` and `start`.

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
