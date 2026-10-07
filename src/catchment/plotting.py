"""Render one rainfall map in the house layout."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402

from .boundaries import Layers  # noqa: E402
from .gfs import Field  # noqa: E402

HEADER_BLUE = "#0d3f7f"
ACCENT = "#f28c28"
BANNER = "#ffd21f"
OCEAN = "#d6e9f5"
OTHER_LAND = "#dcdcdc"
TEXT_GREY = "#555555"


def smooth(field: Field, factor: int) -> Field:
    if factor <= 1:
        return field
    values = np.nan_to_num(field.values)
    lats = np.linspace(field.lats[0], field.lats[-1], (len(field.lats) - 1) * factor + 1)
    lons = np.linspace(field.lons[0], field.lons[-1], (len(field.lons) - 1) * factor + 1)
    return Field(lats=lats, lons=lons, values=zoom(values, (len(lats) / len(field.lats), len(lons) / len(field.lons)), order=1))


def _class_style(classes: dict):
    bounds = list(classes["bounds"]) + [1e6]
    colours = classes["colours"]
    if len(colours) != len(bounds) - 1:
        raise ValueError("classes.colours needs exactly one colour per class")
    cmap = ListedColormap([(0, 0, 0, 0) if c == "none" else c for c in colours])
    return bounds, cmap, BoundaryNorm(bounds, cmap.N)


def _class_labels(bounds: list[float]) -> list[str]:
    fmt = lambda v: f"{v:g}"  # noqa: E731
    labels = [f"{fmt(a)} - {fmt(b)}" for a, b in zip(bounds[:-2], bounds[1:-1])]
    return labels + [f"Above {fmt(bounds[-2])}"]


def _scale_bar(ax, lon0: float, lat0: float, km: list[int]):
    deg_per_km = 1 / (111.32 * math.cos(math.radians(lat0)))
    total = km[-1] * deg_per_km
    for i, (a, b) in enumerate(zip([0] + km[:-1], km)):
        ax.add_patch(Rectangle((lon0 + a * deg_per_km, lat0), (b - a) * deg_per_km, 0.07,
                               facecolor="black" if i % 2 == 0 else "white", edgecolor="black", lw=0.6, zorder=20))
    for v in [0] + km:
        ax.text(lon0 + v * deg_per_km, lat0 + 0.12, str(v), ha="center", va="bottom", fontsize=6, zorder=20)
    ax.text(lon0 + total + 0.12, lat0, "km", fontsize=6, va="bottom", zorder=20)


def _north_arrow(ax, x: float, y: float):
    ax.annotate("", xy=(x, y), xytext=(x, y - 0.55), xycoords="data",
                arrowprops=dict(facecolor="black", width=2.5, headwidth=8, headlength=9), zorder=20)
    ax.text(x, y + 0.12, "N", ha="center", va="bottom", fontsize=8, fontweight="bold", zorder=20)


def render_map(
    field: Field,
    layers: Layers,
    cfg: dict,
    banner: str,
    date_range: str,
    run: datetime,
    out: Path,
) -> None:
    west, south, east, north = cfg["bbox"]
    classes = cfg["classes"]
    labels = cfg.get("labels", {})
    bounds, cmap, norm = _class_style(classes)
    width = cfg.get("output", {}).get("width_in", 9.0)
    map_h = width * 0.86 * (north - south) / (east - west)
    height = map_h + 2.4

    fig = plt.figure(figsize=(width, height))
    fig.patch.set_facecolor("white")

    # Header band
    hdr = fig.add_axes([0, 1 - 0.95 / height, 1, 0.95 / height])
    hdr.set_axis_off()
    hdr.add_patch(Rectangle((0, 0.04), 1, 0.96, transform=hdr.transAxes, color=HEADER_BLUE))
    hdr.add_patch(Rectangle((0, 0), 1, 0.04, transform=hdr.transAxes, color=ACCENT))
    hdr.text(0.04, 0.68, cfg.get("title", "Rainfall Forecast"), color="white", fontsize=19, fontweight="bold",
             va="center", transform=hdr.transAxes)
    hdr.text(0.535, 0.66, date_range, color="#c9d8ee", fontsize=10, va="center", transform=hdr.transAxes)
    hdr.text(0.96, 0.68, cfg.get("area_name", ""), color="white", fontsize=11, fontweight="bold",
             ha="right", va="center", transform=hdr.transAxes)
    txt = hdr.text(0.055, 0.27, banner, fontsize=12, fontweight="bold", color="#1a1a1a", va="center",
                   transform=hdr.transAxes, zorder=3)
    fig.canvas.draw()
    bb = txt.get_window_extent().transformed(hdr.transAxes.inverted())
    hdr.add_patch(FancyBboxPatch((bb.x0 - 0.012, bb.y0 - 0.05), bb.width + 0.024, bb.height + 0.1,
                                 boxstyle="round,pad=0,rounding_size=0.02", transform=hdr.transAxes,
                                 facecolor=BANNER, edgecolor="#e0a800", lw=0.8, zorder=2))

    # Map
    ax = fig.add_axes([0.06, 1.45 / height, 0.88, map_h / height])
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal")
    ax.set_facecolor(OCEAN)
    layers.neighbours.plot(ax=ax, color=OTHER_LAND, edgecolor="none", zorder=1)
    layers.country.plot(ax=ax, color="white", edgecolor="none", zorder=1)

    f = smooth(field, int(classes.get("upsample", 1)))
    ax.contourf(f.lons, f.lats, np.clip(f.values, 0, None), levels=bounds, cmap=cmap, norm=norm, zorder=2,
                antialiased=True)
    veil = float(classes.get("ocean_veil", 0))
    if veil > 0:
        layers.ocean.plot(ax=ax, color=OCEAN, alpha=veil, edgecolor="none", zorder=3)

    layers.neighbours.boundary.plot(ax=ax, color="#9a9a9a", lw=0.5, zorder=4)
    layers.regions.boundary.plot(ax=ax, color="#8c8c8c", lw=0.45, zorder=4)
    if not layers.basins.empty:
        layers.basins.boundary.plot(ax=ax, color="#7d7d7d", lw=0.9, linestyle=(0, (3, 2)), zorder=5)
    layers.rivers.plot(ax=ax, color="#3a78c8", lw=0.9, zorder=5)
    layers.country.boundary.plot(ax=ax, color="black", lw=1.3, zorder=6)

    halo = dict(fontsize=7, zorder=10)
    for c in labels.get("countries", []):
        ax.text(c["lon"], c["lat"], c["name"], color="#7a2e1d", fontweight="bold", fontsize=9, ha="center", zorder=10)
    for s in labels.get("seas", []):
        ax.text(s["lon"], s["lat"], s["name"], color="#2a5d8f", fontstyle="italic", fontsize=8.5, ha="center", zorder=10)
    for b in cfg.get("basins", {}).get("outlets", []) if not layers.basins.empty else []:
        if "label_lon" in b:
            ax.text(b["label_lon"], b["label_lat"], b["name"].upper(), color="#444444", fontsize=6.5,
                    fontweight="bold", ha="center", zorder=10, fontfamily="monospace",
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])
    rename = labels.get("region_names", {})
    fixed = labels.get("region_positions", {})
    for _, r in layers.regions.iterrows():
        name = rename.get(r["name"], r["name"])
        if name in fixed:
            x, y = fixed[name]
        else:
            p = r.geometry.representative_point()
            x, y = p.x, p.y
        ax.text(x, y, name, ha="center", va="center", color="#333333", fontweight="bold", **halo)
    for t in labels.get("towns", []):
        ax.plot(t["lon"], t["lat"], "o", ms=2.6, color="#333333", zorder=10)
        ax.text(t["lon"] + 0.09, t["lat"], t["name"], fontsize=4.8, va="center", color=TEXT_GREY, zorder=10)
    for rv in labels.get("rivers", []):
        ax.text(rv["lon"], rv["lat"], rv["name"], fontsize=7, color="#2a62b0", fontstyle="italic", zorder=10)

    xt = np.arange(math.ceil(west / 2) * 2, east + 0.01, 2)
    yt = np.arange(math.ceil(south / 2) * 2, north + 0.01, 2)
    ax.set_xticks(xt, [f"{abs(v):g}°{'E' if v >= 0 else 'W'}" for v in xt])
    ax.set_yticks(yt, [f"{abs(v):g}°{'N' if v >= 0 else 'S'}" for v in yt])
    ax.tick_params(labelsize=6, colors=TEXT_GREY, length=2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_edgecolor("#999999")
    _north_arrow(ax, west + 0.55, north - 0.35)
    _scale_bar(ax, west + 0.4, south + 0.6, [100, 200, 300])

    # Legend
    leg = fig.add_axes([0.06, 0.08 / height, 0.88, 1.2 / height])
    leg.set_axis_off()
    leg.set_xlim(0, 1)
    leg.set_ylim(0, 1)
    leg.text(0.5, 0.86, "Rainfall (mm)", ha="center", fontsize=8.5, fontweight="bold")
    texts = _class_labels(bounds)
    ncol = math.ceil(len(texts) / 2)
    x0, colw = 0.16, 0.72 / ncol
    for i, (label, colour) in enumerate(zip(texts, classes["colours"])):
        col, row = divmod(i, 2)
        x, y = x0 + col * colw, 0.66 - row * 0.15
        leg.add_patch(Rectangle((x, y), 0.032, 0.1, facecolor="white" if colour == "none" else colour,
                                edgecolor="#888888", lw=0.4))
        leg.text(x + 0.04, y + 0.05, label, fontsize=6.5, va="center")
    handles = [
        Line2D([], [], color="black", lw=1.3, label="Somalia border"),
        Line2D([], [], color="#8c8c8c", lw=0.6, label="Region boundary"),
        Line2D([], [], color="#7d7d7d", lw=0.9, linestyle=(0, (3, 2)), label="Juba and Shabelle basins (upstream)"),
        Line2D([], [], color="#3a78c8", lw=0.9, label="Rivers"),
    ]
    if layers.basins.empty:
        handles.pop(2)
    leg.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.18), ncol=len(handles), frameon=False, fontsize=6.5)
    leg.text(0.5, -0.02, f"Source: NOAA/NCEP GFS 0.25°, {run:%HZ} run of {run.day} {run:%B %Y}. "
             "Model outlook, not an official warning.", ha="center", fontsize=6, color=TEXT_GREY)

    out.parent.mkdir(parents=True, exist_ok=True)
    # Strip timestamps so identical inputs give byte-identical PNGs.
    fig.savefig(out, dpi=cfg.get("output", {}).get("dpi", 150), metadata={"Software": None})
    plt.close(fig)
