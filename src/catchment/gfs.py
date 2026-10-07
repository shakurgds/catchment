"""Download cumulative GFS precipitation (APCP) fields.

GFS pgrb2 files carry two APCP records per forecast hour: the current bucket
(e.g. ``186-192 hour acc``) and the running total since initialisation
(``0-192 hour acc`` or ``0-8 day acc``).  We only fetch the running total, so
the rainfall between any two forecast hours is a simple difference and we
never need to reassemble buckets.

Only the bytes of that single GRIB message are downloaded (HTTP range request
driven by the ``.idx`` sidecar), which keeps a full week to a few megabytes.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

log = logging.getLogger(__name__)

AWS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"


@dataclass(frozen=True)
class IdxRecord:
    number: int
    start: int
    end: int | None  # exclusive; None means "to end of file"
    variable: str
    level: str
    description: str


@dataclass(frozen=True)
class Field:
    lats: np.ndarray  # 1-D, ascending
    lons: np.ndarray  # 1-D, ascending, -180..180
    values: np.ndarray  # 2-D (lat, lon), mm


def file_url(base: str, run: datetime, fh: int, resolution: str = "0p25") -> str:
    ymd, hh = run.strftime("%Y%m%d"), run.strftime("%H")
    return f"{base}/gfs.{ymd}/{hh}/atmos/gfs.t{hh}z.pgrb2.{resolution}.f{fh:03d}"


def parse_idx(text: str) -> list[IdxRecord]:
    """Parse a wgrib2-style inventory into records with byte ranges."""
    rows = [line.split(":") for line in text.strip().splitlines() if line.strip()]
    records = []
    for i, row in enumerate(rows):
        end = int(rows[i + 1][1]) if i + 1 < len(rows) else None
        records.append(
            IdxRecord(
                number=int(row[0]),
                start=int(row[1]),
                end=end,
                variable=row[3],
                level=row[4],
                description=row[5],
            )
        )
    return records


def _acc_hours(description: str) -> tuple[int, int] | None:
    """'0-27 hour acc fcst' -> (0, 27); '0-8 day acc fcst' -> (0, 192)."""
    m = re.fullmatch(r"(\d+)-(\d+) (hour|day) acc fcst", description.strip())
    if not m:
        return None
    scale = 24 if m.group(3) == "day" else 1
    return int(m.group(1)) * scale, int(m.group(2)) * scale


def find_cumulative_apcp(records: list[IdxRecord], fh: int) -> IdxRecord:
    """Return the APCP record accumulated from initialisation to ``fh``."""
    for rec in records:
        if rec.variable == "APCP" and rec.level == "surface" and _acc_hours(rec.description) == (0, fh):
            return rec
    found = [r.description for r in records if r.variable == "APCP"]
    raise LookupError(f"No 0-{fh}h APCP record in inventory (APCP records: {found})")


def valid_forecast_hour(fh: int) -> bool:
    """GFS 0.25° output is hourly to f120, 3-hourly to f240, 12-hourly to f384."""
    if fh <= 0 or fh > 384:
        return False
    if fh <= 120:
        return True
    if fh <= 240:
        return fh % 3 == 0
    return fh % 12 == 0


class Downloader:
    def __init__(self, cache_dir: Path, base_urls: list[str], retries: int = 4, timeout: int = 60):
        self.cache_dir = Path(cache_dir)
        self.base_urls = base_urls
        self.retries = retries
        self.timeout = timeout
        self.session = requests.Session()

    def _get(self, url: str, headers: dict | None = None) -> bytes:
        delay = 2
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.get(url, headers=headers or {}, timeout=self.timeout)
                if resp.status_code == 404:
                    raise FileNotFoundError(url)
                resp.raise_for_status()
                return resp.content
            except FileNotFoundError:
                raise
            except requests.RequestException as exc:
                if attempt == self.retries:
                    raise
                log.warning("GET %s failed (%s); retrying in %ss", url, exc, delay)
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    def exists(self, run: datetime, fh: int) -> bool:
        for base in self.base_urls:
            try:
                self._get(file_url(base, run, fh) + ".idx")
                return True
            except (FileNotFoundError, requests.RequestException):
                continue
        return False

    def fetch_cumulative_apcp(self, run: datetime, fh: int) -> tuple[Path, dict]:
        """Download (or reuse from cache) the 0-fh APCP GRIB message.

        Returns the local path and a provenance record.
        """
        out = self.cache_dir / "gfs" / run.strftime("%Y%m%d%H") / f"apcp_0-{fh:03d}h.grib2"
        meta_errors = []
        for base in self.base_urls:
            url = file_url(base, run, fh)
            try:
                if out.exists():
                    data = out.read_bytes()
                else:
                    rec = find_cumulative_apcp(parse_idx(self._get(url + ".idx").decode()), fh)
                    byte_range = f"bytes={rec.start}-{'' if rec.end is None else rec.end - 1}"
                    data = self._get(url, headers={"Range": byte_range})
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(data)
                    log.info("Downloaded %s (%d bytes) from %s", out.name, len(data), base)
                return out, {
                    "forecast_hour": fh,
                    "source_url": url,
                    "file": str(out),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            except (FileNotFoundError, LookupError, requests.RequestException) as exc:
                meta_errors.append(f"{base}: {exc}")
        raise RuntimeError(f"Could not fetch APCP 0-{fh}h for run {run:%Y-%m-%d %HZ}: {meta_errors}")


def read_field(path: Path, bbox: tuple[float, float, float, float]) -> Field:
    """Read a single-message GRIB2 file and subset it to (west, south, east, north)."""
    import pygrib

    west, south, east, north = bbox
    with pygrib.open(str(path)) as grbs:
        msg = grbs.message(1)
        values, lats, lons = msg.data(lat1=south, lat2=north, lon1=west % 360, lon2=east % 360)
    lat1d, lon1d = lats[:, 0], lons[0, :]
    lon1d = np.where(lon1d > 180, lon1d - 360, lon1d)
    values = np.ma.filled(values, np.nan).astype(float)
    if lat1d[0] > lat1d[-1]:
        lat1d, values = lat1d[::-1], values[::-1, :]
    return Field(lats=lat1d, lons=lon1d, values=values)


def parse_run(text: str) -> datetime:
    """Accept '2026-10-04T00', '2026100400' or '2026-10-04 00Z'."""
    cleaned = re.sub(r"[^0-9]", "", text)
    if len(cleaned) != 10:
        raise ValueError(f"Run must look like YYYY-MM-DDTHH, got {text!r}")
    run = datetime.strptime(cleaned, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    if run.hour % 6:
        raise ValueError("GFS cycles are 00, 06, 12 and 18 UTC")
    return run


def latest_run(downloader: Downloader, max_fh: int, now: datetime | None = None, cycles: tuple[int, ...] = (0, 6, 12, 18)) -> datetime:
    """Most recent cycle whose last needed forecast hour is already published."""
    now = now or datetime.now(timezone.utc)
    t = now.replace(minute=0, second=0, microsecond=0)
    t -= timedelta(hours=t.hour % 6)
    for _ in range(4 * 5):  # look back five days
        if t.hour in cycles and downloader.exists(t, max_fh):
            return t
        t -= timedelta(hours=6)
    raise RuntimeError("No complete GFS cycle found in the last five days")
