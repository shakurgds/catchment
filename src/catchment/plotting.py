"""Render one rainfall map in the Catchment house style.

Layout, top to bottom: a headline block (period as the title, a strip showing
where the period sits in the week), a stepped colour bar, the map, and a
footer with sources.  Rainfall is drawn over land only; the empty sea in the
south-east carries a small ranked chart of the wettest regions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager, patheffects  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402
from shapely.geometry import MultiPolygon, Polygon  # noqa: E402

from .boundaries import Layers  # noqa: E402
from .gfs import Field  # noqa: E402

FONT_DIR = Path(__file__).parent / "fonts"
for _ttf in sorted(FONT_DIR.glob("*.ttf")):
    font_manager.fontManager.addfont(str(_ttf))
FONT = "IBM Plex Sans" if any(FONT_DIR.glob("*.ttf")) else "DejaVu Sans"

# Default theme; every key can be overridden under `style:` in the config.
THEME = {
    "paper": "#f7f5f0",
    "ink": "#1c2430",
    "ink_muted": "#5b6573",
    "ink_faint": "#8d95a0",
    "accent": "#0f7c80",
    "sea": "#e2ebee",
    "coast_glow": "#cddde3",
    "land": "#ebe7df",
    "focus_land": "#fbfaf7",
    "region_line": "#c9c3b6",
    "border": "#1c2430",
    "river": "#1f5fa8",
    "basin_line": "#5b6573",
    "grid": "#c9d3d8",
}


@dataclass
class Context:
    """What the map needs to know about the other periods in the run."""

    timeline: list[tuple[str, float, bool]]  # (day initial, country-mean mm, is this map's day)
    top_regions: list[tuple[str, float]]  # (display name, mean mm), wettest first
    headline: str  # e.g. "Wednesday 7 October"
    subhead: str  # e.g. "24-hour rainfall · day 3 of 7"


def smooth(field: Field, factor: int) -> Field:
    if factor <= 1:
        return field
    values = np.nan_to_num(field.values)
    lats = np.linspace(field.lats[0], field.lats[-1], (len(field.lats) - 1) * factor + 1)
    lons = np.linspace(field.lons[0], field.lons[-1], (len(field.lons) - 1) * factor + 1)
    return Field(lats=lats, lons=lons, values=zoom(values, (len(lats) / len(field.lats), len(lons) / len(field.lons)), order=1))


def class_style(classes: dict):
    bounds = list(classes["bounds"]) + [1e6]
    colours = classes["colours"]
    if len(colours) != len(bounds) - 1:
        raise ValueError("classes.colours needs exactly one colour per class")
    cmap = ListedColormap([(0, 0, 0, 0) if c == "none" else c for c in colours])
    return bounds, cmap, BoundaryNorm(bounds, cmap.N)


def shapely_to_path(geom) -> MplPath:
    """Polygon/MultiPolygon (with holes) -> one compound matplotlib path."""
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    verts, codes = [], []
    for poly in polys:
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        for ring in [poly.exterior, *poly.interiors]:
            xy = np.asarray(ring.coords)
            verts.append(xy)
            codes.append([MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 2) + [MplPath.CLOSEPOLY])
    return MplPath(np.concatenate(verts), np.concatenate(codes))


def _num(v: float) -> str:
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}"


def _halo(colour: str, width: float = 2.2):
    return [patheffects.withStroke(linewidth=width, foreground=colour)]


def _colour_bar(fig, rect, bounds, colours, t):
    """Stepped bar: equal-width cells, tick labels on the class edges."""
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    n = len(colours)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    for i, c in enumerate(colours):
        face = t["focus_land"] if c == "none" else c
        ax.add_patch(Rectangle((i + 0.03, 0.42), 0.94, 0.38, facecolor=face, edgecolor=t["grid"] if c == "none" else "none", lw=0.6))
    for i, b in enumerate(bounds[:-1]):
        ax.text(i, 0.12, f"{b:g}", ha="center", va="center", fontsize=6.6, color=t["ink_muted"])
    ax.text(n, 0.12, "mm", ha="center", va="center", fontsize=6.6, color=t["ink_muted"])


def _timeline(fig, rect, timeline, t):
    """Seven mini columns, one per day; height = country-mean rainfall."""
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    n = len(timeline)
    peak = max([v for _, v, _ in timeline] + [1.0])
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-0.62, 1.12)
    ax.text(n - 0.4, -0.55, "Somalia average per day, mm", ha="right", va="center", fontsize=5.6, color=t["ink_faint"])
    for i, (day, value, current) in enumerate(timeline):
        h = 0.06 + 0.84 * value / peak
        colour = t["accent"] if current else t["grid"]
        ax.add_patch(FancyBboxPatch((i - 0.28, 0), 0.56, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor=colour, edgecolor="none", mutation_aspect=0.4))
        ax.text(i, -0.22, day, ha="center", va="center", fontsize=6.8,
                color=t["ink"] if current else t["ink_faint"], fontweight="semibold" if current else "regular")
        if current:
            ax.text(i, h + 0.12, _num(value), ha="center", va="bottom", fontsize=6.6, color=t["accent"], fontweight="semibold")


def _region_chart(ax, rows, box, t):
    """Ranked horizontal bars drawn in data coordinates inside the map."""
    x0, y0, x1, y1 = box
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.18",
                                facecolor=t["paper"], edgecolor="none", alpha=0.92, zorder=30))
    ax.text(x0 + 0.25, y1 - 0.35, "WETTEST REGIONS", fontsize=6.4, color=t["accent"], fontweight="semibold", zorder=31)
    ax.text(x0 + 0.25, y1 - 0.68, "area-average rainfall, mm", fontsize=6, color=t["ink_muted"], zorder=31)
    if not rows:
        return
    peak = max(v for _, v in rows) or 1.0
    bar_x = x0 + 0.25 + 1.6
    bar_w = x1 - bar_x - 0.55
    step = (y1 - y0 - 1.05) / len(rows)
    for i, (name, value) in enumerate(rows):
        y = y1 - 1.05 - (i + 0.5) * step
        ax.text(bar_x - 0.08, y, name, ha="right", va="center", fontsize=6.2, color=t["ink"], zorder=31)
        w = max(bar_w * value / peak, 0.04)
        ax.add_patch(FancyBboxPatch((bar_x, y - step * 0.28), w, step * 0.56, boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=t["accent"], edgecolor="none", zorder=31))
        ax.text(bar_x + w + 0.08, y, _num(value), va="center", fontsize=6.2, color=t["ink_muted"], zorder=31)


def _base_map(ax, field: Field, layers: Layers, cfg: dict, t: dict, mini: bool = False) -> None:
    """Land, rainfall (clipped to land), graticule, borders and rivers."""
    west, south, east, north = cfg["bbox"]
    classes = cfg["classes"]
    bounds, cmap, norm = class_style(classes)
    scale = 0.55 if mini else 1.0
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(t["sea"])
    for s in ax.spines.values():
        s.set_visible(False)

    land = layers.neighbours.geometry.union_all().union(layers.country.geometry.iloc[0])
    layers.country.boundary.plot(ax=ax, color=t["coast_glow"], lw=5 * scale, zorder=1)
    layers.neighbours.boundary.plot(ax=ax, color=t["coast_glow"], lw=5 * scale, zorder=1)
    layers.neighbours.plot(ax=ax, color=t["land"], edgecolor="none", zorder=2)
    layers.country.plot(ax=ax, color=t["focus_land"], edgecolor="none", zorder=2)

    f = smooth(field, int(classes.get("upsample", 1)))
    cs = ax.contourf(f.lons, f.lats, np.clip(f.values, 0, None), levels=bounds, cmap=cmap, norm=norm, zorder=3, antialiased=True)
    clip = PathPatch(shapely_to_path(land), transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    cs.set_clip_path(clip)

    for lon in np.arange(math.ceil(west / 2) * 2, east + 0.01, 2):
        ax.axvline(lon, color=t["grid"], lw=0.4, ls=(0, (1, 2)), zorder=1.5)
        if not mini:
            ax.text(lon, -0.012, f"{lon:g}°E", transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=5.6, color=t["ink_faint"])
    for lat in np.arange(math.ceil(south / 2) * 2, north + 0.01, 2):
        ax.axhline(lat, color=t["grid"], lw=0.4, ls=(0, (1, 2)), zorder=1.5)
        if not mini:
            ax.text(1.008, lat, f"{abs(lat):g}°{'N' if lat >= 0 else 'S'}", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=5.6, color=t["ink_faint"])
    ax.set_xticks([])
    ax.set_yticks([])

    layers.regions.boundary.plot(ax=ax, color=t["region_line"], lw=0.6 * scale, zorder=4)
    layers.neighbours.boundary.plot(ax=ax, color=t["ink_faint"], lw=0.5 * scale, zorder=4)
    if not layers.basins.empty:
        layers.basins.boundary.plot(ax=ax, color=t["basin_line"], lw=0.9 * scale, linestyle=(0, (4, 2.5)), zorder=5)
    layers.rivers.plot(ax=ax, color="white", lw=2.0 * scale, zorder=5)
    layers.rivers.plot(ax=ax, color=t["river"], lw=0.9 * scale, zorder=5.1)
    layers.country.boundary.plot(ax=ax, color="white", lw=2.6 * scale, zorder=6)
    layers.country.boundary.plot(ax=ax, color=t["border"], lw=1.0 * scale, zorder=6.1)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)


def _footer(fig, W: float, H: float, margin: float, cfg: dict, t: dict, run: datetime, layers: Layers) -> None:
    fig.add_artist(plt.Line2D([margin / W, 1 - margin / W], [0.4 / H, 0.4 / H], color=t["grid"], lw=0.6))
    fig.text(margin / W, 0.2 / H, f"NOAA/NCEP GFS 0.25°, {run:%HZ} {run.day} {run:%b %Y}  ·  geoBoundaries, Natural Earth"
             f"{', HydroBASINS' if not layers.basins.empty else ''}  ·  Model guidance only, not a warning",
             fontsize=5.8, color=t["ink_faint"], va="center")
    if cfg.get("brand"):
        fig.text(1 - margin / W, 0.2 / H, cfg["brand"], fontsize=7.2, color=t["ink"], fontweight="semibold",
                 ha="right", va="center")



def render_map(field: Field, layers: Layers, cfg: dict, ctx: Context, run: datetime, out: Path) -> None:
    t = {**THEME, **cfg.get("style", {})}
    west, south, east, north = cfg["bbox"]
    classes = cfg["classes"]
    labels = cfg.get("labels", {})
    bounds, cmap, norm = class_style(classes)
    plt.rcParams.update({"font.family": FONT, "text.color": t["ink"]})

    W, H = 7.2, 9.0  # 4:5, 1080 x 1350 px at 150 dpi
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(t["paper"])
    margin = 0.42

    # Headline block
    fig.text(margin / W, 1 - 0.36 / H, f"{cfg.get('kicker', 'RAINFALL OUTLOOK')}  ·  {cfg.get('area_name', '')}".strip(" ·"),
             fontsize=8, color=t["accent"], fontweight="semibold")
    fig.text(margin / W, 1 - 0.78 / H, ctx.headline, fontsize=21, fontweight="bold", color=t["ink"])
    fig.text(margin / W, 1 - 1.06 / H, ctx.subhead, fontsize=9, color=t["ink_muted"])
    _timeline(fig, [(W - margin - 2.1) / W, 1 - 1.3 / H, 2.1 / W, 0.94 / H], ctx.timeline, t)
    _colour_bar(fig, [margin / W, 1 - 1.62 / H, (W - 2 * margin) / W, 0.36 / H], bounds, classes["colours"], t)

    # Map
    top, bottom = 1.72, 0.62
    map_h = H - top - bottom
    map_w = min(W - 2 * margin, map_h * (east - west) / (north - south))
    ax = fig.add_axes([(W - map_w) / 2 / W, bottom / H, map_w / W, map_h / H])
    _base_map(ax, field, layers, cfg, t)

    paper_halo = _halo(t["focus_land"], 2.4)
    for c in labels.get("countries", []):
        ax.text(c["lon"], c["lat"], c["name"].upper(), color=t["ink_muted"], fontsize=8, ha="center",
                fontweight="medium", zorder=10, path_effects=_halo(t["land"], 2.4))
    for s in labels.get("seas", []):
        ax.text(s["lon"], s["lat"], s["name"], color="#6f8f9c", fontstyle="italic", fontsize=8.5, ha="center", zorder=10)
    if not layers.basins.empty:
        for b in cfg.get("basins", {}).get("outlets", []):
            if "label_lon" in b:
                ax.text(b["label_lon"], b["label_lat"], b["name"], color=t["ink"], fontsize=6.6, fontstyle="italic",
                        ha="center", zorder=10, path_effects=paper_halo)
    rename = labels.get("region_names", {})
    fixed = labels.get("region_positions", {})
    for _, r in layers.regions.iterrows():
        name = rename.get(r["name"], r["name"])
        x, y = fixed.get(name) or (r.geometry.representative_point().x, r.geometry.representative_point().y)
        ax.text(x, y, name, ha="center", va="center", color=t["ink"], fontsize=6.6, fontweight="semibold",
                zorder=10, path_effects=paper_halo)
    for town in labels.get("towns", []):
        ax.plot(town["lon"], town["lat"], "o", ms=3.0, mfc=t["focus_land"], mec=t["ink"], mew=0.8, zorder=11)
        ax.text(town["lon"] + 0.1, town["lat"] - 0.02, town["name"], fontsize=5.2, va="center", color=t["ink_muted"],
                zorder=11, path_effects=_halo(t["focus_land"], 1.8))
    for rv in labels.get("rivers", []):
        ax.text(rv["lon"], rv["lat"], rv["name"], fontsize=6.6, color=t["river"], fontstyle="italic", zorder=10,
                path_effects=paper_halo)

    chart_box = cfg.get("style", {}).get("region_chart_box", [46.7, -2.2, 51.3, 2.9])
    _region_chart(ax, ctx.top_regions, chart_box, t)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)

    _footer(fig, W, H, margin, cfg, t, run, layers)

    out.parent.mkdir(parents=True, exist_ok=True)
    # Strip timestamps so identical inputs give byte-identical PNGs.
    fig.savefig(out, dpi=cfg.get("output", {}).get("dpi", 150), metadata={"Software": None}, facecolor=t["paper"])
    plt.close(fig)


def render_overview(
    panels: list[tuple[str, str, Field, bool]],
    layers: Layers,
    cfg: dict,
    headline: str,
    subhead: str,
    run: datetime,
    out: Path,
) -> None:
    """All days on one page: a 4 x 2 grid of small maps sharing one colour bar.

    ``panels`` holds (title, note, field, is_total) in reading order.
    """
    t = {**THEME, **cfg.get("style", {})}
    west, south, east, north = cfg["bbox"]
    bounds, _, _ = class_style(cfg["classes"])
    plt.rcParams.update({"font.family": FONT, "text.color": t["ink"]})

    W, margin, gap = 7.2, 0.42, 0.14
    cols = 4
    rows = math.ceil(len(panels) / cols)
    pw = (W - 2 * margin - (cols - 1) * gap) / cols
    ph = pw * (north - south) / (east - west)
    title_h, header_h, footer_h = 0.3, 1.72, 0.55
    H = header_h + rows * (ph + title_h) + (rows - 1) * gap + footer_h
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(t["paper"])

    fig.text(margin / W, 1 - 0.36 / H, f"{cfg.get('kicker', 'RAINFALL OUTLOOK')}  ·  {cfg.get('area_name', '')}".strip(" ·"),
             fontsize=8, color=t["accent"], fontweight="semibold")
    fig.text(margin / W, 1 - 0.78 / H, headline, fontsize=21, fontweight="bold", color=t["ink"])
    fig.text(margin / W, 1 - 1.06 / H, subhead, fontsize=9, color=t["ink_muted"])
    _colour_bar(fig, [margin / W, 1 - 1.62 / H, (W - 2 * margin) / W, 0.36 / H], bounds, cfg["classes"]["colours"], t)

    for i, (title, note, field, is_total) in enumerate(panels):
        r, c = divmod(i, cols)
        x = margin + c * (pw + gap)
        y_top = H - header_h - r * (ph + title_h + gap)
        ax = fig.add_axes([x / W, (y_top - title_h - ph) / H, pw / W, ph / H])
        _base_map(ax, field, layers, cfg, t, mini=True)
        colour = t["accent"] if is_total else t["ink"]
        fig.text(x / W, (y_top - 0.15) / H, title, fontsize=9, fontweight="bold", color=colour, va="center")
        fig.text((x + pw) / W, (y_top - 0.15) / H, note, fontsize=6.4, color=t["ink_muted"], va="center", ha="right")
        if is_total:
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_edgecolor(t["accent"])
                s.set_linewidth(1.4)

    _footer(fig, W, H, margin, cfg, t, run, layers)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=cfg.get("output", {}).get("dpi", 150), metadata={"Software": None}, facecolor=t["paper"])
    plt.close(fig)
