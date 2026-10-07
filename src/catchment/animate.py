"""Turn the daily maps into a day-by-day animation (GIF and MP4).

Frames are the finished PNGs, so the animation always matches the stills.
Each day is held for a moment and then cut to the next (cross-fades ghost the
text); the weekly total is held longest before the loop restarts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def write_gif(pngs: list[Path], out: Path, width: int = 720, hold_s: float = 1.6, final_hold_s: float = 3.5) -> Path:
    """Looping GIF, scaled to ``width`` px to keep the file small."""
    frames = []
    for p in pngs:
        img = Image.open(p).convert("RGB")
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
        frames.append(img.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    durations = [int(hold_s * 1000)] * (len(frames) - 1) + [int(final_hold_s * 1000)]
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    return out


def write_mp4(pngs: list[Path], out: Path, fps: int = 25, hold_s: float = 1.6, final_hold_s: float = 3.5) -> Path:
    """Full-size H.264 MP4 (yuv420p), the format LinkedIn plays inline."""
    import imageio_ffmpeg

    first = Image.open(pngs[0])
    w, h = first.width - first.width % 2, first.height - first.height % 2
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(out), (w, h), fps=fps, codec="libx264", pix_fmt_out="yuv420p", quality=None, macro_block_size=2,
        output_params=["-crf", "20", "-preset", "medium", "-movflags", "+faststart",
                       "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact", "-threads", "1"],
    )
    writer.send(None)
    for i, p in enumerate(pngs):
        frame = np.ascontiguousarray(np.asarray(Image.open(p).convert("RGB"))[:h, :w])
        for _ in range(round((final_hold_s if i == len(pngs) - 1 else hold_s) * fps)):
            writer.send(frame)
    writer.close()
    return out
