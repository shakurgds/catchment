"""Turn calendar days into GFS forecast-hour windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

from .gfs import valid_forecast_hour


@dataclass(frozen=True)
class Period:
    key: str  # used in file names, e.g. "2026-10-07" or "week_2026-10-05_2026-10-11"
    label: str  # yellow banner text
    start_fh: int
    end_fh: int
    kind: str  # "daily" or "weekly"


def default_start(run: datetime) -> date:
    """The first Monday strictly after the run date."""
    d = run.date() + timedelta(days=1)
    return d + timedelta(days=(7 - d.weekday()) % 7)


def _fh(run: datetime, day: date, utc_offset_hours: int) -> int:
    local_midnight = datetime.combine(day, time(0), tzinfo=timezone(timedelta(hours=utc_offset_hours)))
    hours = (local_midnight - run).total_seconds() / 3600
    if hours != int(hours):
        raise ValueError("UTC offset must be a whole number of hours")
    return int(hours)


def _day_label(d: date) -> str:
    return f"{d:%A} {d.day} {d:%B %Y}"


def build_periods(run: datetime, start: date, days: int, utc_offset_hours: int = 0) -> list[Period]:
    """One period per day plus a total over all days."""
    periods = []
    for i in range(days):
        d = start + timedelta(days=i)
        periods.append(Period(d.isoformat(), _day_label(d), _fh(run, d, utc_offset_hours), _fh(run, d + timedelta(days=1), utc_offset_hours), "daily"))
    end = start + timedelta(days=days - 1)
    periods.append(
        Period(
            f"total_{start.isoformat()}_{end.isoformat()}",
            f"{'Weekly' if days == 7 else f'{days}-day'} Total: {_day_label(start).rsplit(' ', 1)[0]} to {_day_label(end)}",
            periods[0].start_fh,
            periods[-1].end_fh,
            "weekly",
        )
    )
    for p in periods:
        for fh in (p.start_fh, p.end_fh):
            if fh != 0 and not valid_forecast_hour(fh):
                raise ValueError(
                    f"{p.key}: forecast hour {fh} is not in GFS output (hourly to 120 h, 3-hourly to 240 h, "
                    f"12-hourly to 384 h); pick a different start date, run or UTC offset"
                )
            if fh < 0:
                raise ValueError(f"{p.key} starts before the model run; choose a later start date")
    return periods


def range_text(start: date, days: int) -> str:
    end = start + timedelta(days=days - 1)
    if start.month == end.month:
        return f"{start.day}-{end.day} {end:%B %Y}"
    return f"{start.day} {start:%B} - {end.day} {end:%B %Y}"
