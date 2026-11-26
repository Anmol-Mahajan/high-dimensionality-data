"""Record the demo GIFs from the running app with scripted, eased mouse paths.

Each take is a real capture of ``notebooks/explorer.py``. Usage::

    uv run python scripts/record_gifs.py
    uv run python scripts/record_gifs.py --only 02_lasso_interference
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import random
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from playwright.async_api import BrowserContext, Page, async_playwright

ROOT = Path(__file__).resolve().parent.parent
GIF_DIR = ROOT / "assets" / "gifs"
TAKE_DIR = ROOT / "assets" / "takes"
VIEWPORT = {"width": 1200, "height": 1200}
STROKE_STEPS = 40
LASSO_STEPS = 72
LATENCY, RSRP, SINR = 0, 1, 2
RESULTS_CURSOR = (840.0, 1120.0)

FIND_SCROLLER = """
const findScroller = () => {
  let box = window.__kpiView.root.parentElement;
  while (box && !(box.scrollHeight > box.clientHeight + 4 &&
         /auto|scroll/.test(getComputedStyle(box).overflowY))) {
    box = box.parentElement ?? box.getRootNode().host;
  }
  return box ?? document.scrollingElement;
};
"""

# Headless Chromium does not paint the OS cursor, so a visible one is drawn in the page.
CURSOR_SCRIPT = """
window.addEventListener('DOMContentLoaded', () => {
  const cursor = document.createElement('div');
  cursor.innerHTML = `<svg width="22" height="22" viewBox="0 0 22 22">
    <path d="M3 2 L3 18 L7.5 14 L10.5 20.5 L13 19.3 L10.1 13 L16 13 Z"
      fill="#111" stroke="#fff" stroke-width="1.4" stroke-linejoin="round"/></svg>`;
  Object.assign(cursor.style, {
    position: 'fixed', left: '0', top: '0', zIndex: 2147483647, pointerEvents: 'none',
    transform: 'translate(600px, 700px)',
  });
  document.documentElement.appendChild(cursor);
  window.addEventListener('mousemove', (event) => {
    cursor.style.transform = `translate(${event.clientX - 3}px, ${event.clientY - 2}px)`;
  }, true);
});
"""


def ease(t: float) -> float:
    """Smoothstep easing from 0 to 1."""
    return t * t * (3 - 2 * t)


def find_ffmpeg() -> str:
    """Return ffmpeg from PATH, falling back to the imageio-ffmpeg binary."""
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def start_app(port: int) -> subprocess.Popen[bytes]:
    """Start the marimo app and wait until it responds."""
    server = subprocess.Popen(
        ["uv", "run", "marimo", "run", "notebooks/explorer.py", "--headless", "--port", str(port)],
        cwd=ROOT,
        env={**os.environ, "KPI_EXPLORER_DEV": "1"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return server
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.1)
    os.killpg(os.getpgid(server.pid), signal.SIGTERM)
    raise RuntimeError("marimo did not start")


async def pace(started: float, seconds: float, step: int, steps: int) -> None:
    """Sleep until this step's wall-clock time, so a stroke lasts the requested duration."""
    delay = started + seconds * step / steps - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)


class Director:
    """Drives one take: eased cursor moves, brush strokes and toolbar clicks."""

    def __init__(self, page: Page, seed: int = 7) -> None:
        self.page = page
        self.position = (600.0, 700.0)
        self.random = random.Random(seed)

    async def hold(self, seconds: float) -> None:
        await self.page.wait_for_timeout(seconds * 1000)

    async def glide(self, target: tuple[float, float], seconds: float = 0.42) -> None:
        """Move the cursor to a point along an eased path."""
        start = self.position
        started = time.monotonic()
        for step in range(1, STROKE_STEPS + 1):
            t = ease(step / STROKE_STEPS)
            await self.page.mouse.move(
                start[0] + (target[0] - start[0]) * t, start[1] + (target[1] - start[1]) * t
            )
            await pace(started, seconds, step, STROKE_STEPS)
        self.position = target

    async def data_point(
        self, row: int, col: int, x_value: float, y_value: float
    ) -> tuple[float, float]:
        """Convert data coordinates in a cell to viewport pixels."""
        return (await self.data_points(row, col, [(x_value, y_value)]))[0]

    async def data_points(
        self, row: int, col: int, values: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Convert many data coordinates in a cell to viewport pixels in one call."""
        points = await self.page.evaluate(
            """([row, col, values]) => {
                const v = window.__kpiView;
                const rect = v.overlayCanvas.getBoundingClientRect();
                const [x0, x1] = v.domains[col];
                const [y0, y1] = v.domains[row];
                const step = v.grid.cell + 1;
                return values.map(([xv, yv]) => [
                    rect.left + 44 + col * step + ((xv - x0) / (x1 - x0)) * v.grid.cell,
                    rect.top + 2 + row * step + (1 - (yv - y0) / (y1 - y0)) * v.grid.cell,
                ]);
            }""",
            [row, col, values],
        )
        return [(float(x), float(y)) for x, y in points]

    async def drag_line(
        self, start: tuple[float, float], end: tuple[float, float], seconds: float = 0.9
    ) -> None:
        """Press at ``start`` and drag to ``end`` along an eased path."""
        await self.glide(start)
        await self.hold(0.12)
        await self.page.mouse.down()
        started = time.monotonic()
        for step in range(1, STROKE_STEPS + 1):
            t = ease(step / STROKE_STEPS)
            await self.page.mouse.move(
                start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t
            )
            await pace(started, seconds, step, STROKE_STEPS)
        await self.page.mouse.up()
        self.position = end

    async def lasso(
        self,
        row: int,
        col: int,
        center: tuple[float, float],
        radii: tuple[float, float],
        seconds: float = 1.7,
        steps: int = LASSO_STEPS,
    ) -> None:
        """Draw a slightly wobbly closed loop around a point, like a hand would."""
        phases = [self.random.uniform(0, math.tau) for _ in range(2)]
        outline = []
        for step in range(steps + 1):
            angle = -math.pi * 0.75 + math.tau * step / steps
            wobble = (
                1 + 0.035 * math.sin(3 * angle + phases[0]) + 0.02 * math.sin(5 * angle + phases[1])
            )
            outline.append(
                (
                    center[0] + radii[0] * wobble * math.cos(angle),
                    center[1] + radii[1] * wobble * math.sin(angle),
                )
            )
        points = await self.data_points(row, col, outline)
        await self.glide(points[0])
        await self.hold(0.12)
        await self.page.mouse.down()
        started = time.monotonic()
        for step, point in enumerate(points[1:], start=1):
            await self.page.mouse.move(*point)
            await pace(started, seconds, step, steps)
        await self.page.mouse.up()
        self.position = points[-1]

    async def click(self, label: str) -> None:
        """Glide to a toolbar button and click it.

        Raises:
            RuntimeError: If the button is scrolled out of view.
        """
        button = self.page.locator("button.kpi-btn", has_text=label).first
        box = await button.bounding_box()
        if box is None or box["y"] < 0 or box["y"] + box["height"] > VIEWPORT["height"]:
            raise RuntimeError(f"Toolbar button {label!r} is outside the viewport")
        target = (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        await self.glide(target, seconds=0.35)
        await self.hold(0.1)
        await self.page.mouse.down()
        await self.page.mouse.up()
        await self.hold(0.15)

    async def choose_color(self, name: str) -> None:
        """Glide to the colour-by dropdown and switch it."""
        select = self.page.locator("select.kpi-select")
        box = await select.bounding_box()
        await self.glide((box["x"] + box["width"] / 2, box["y"] + box["height"] / 2), seconds=0.5)
        await self.hold(0.2)
        await select.select_option(name)
        await self.hold(0.2)

    async def scroll_to(self, top: float, seconds: float = 0.7) -> None:
        """Smoothly scroll marimo's page container to a position."""
        await self.page.evaluate(
            "async ([top, seconds]) => {"
            + FIND_SCROLLER
            + """
                const scroller = findScroller();
                const from = scroller.scrollTop;
                const started = performance.now();
                await new Promise((done) => {
                    const tick = (now) => {
                        const t = Math.min(1, (now - started) / (seconds * 1000));
                        scroller.scrollTop = from + (top - from) * t * t * (3 - 2 * t);
                        if (t < 1) requestAnimationFrame(tick); else done();
                    };
                    requestAnimationFrame(tick);
                });
            }""",
            [top, seconds],
        )

    async def widget_scroll_top(self, offset: float = 12) -> float:
        """Return the scroll position that places the widget ``offset`` px below the top."""
        return await self.page.evaluate(
            "(offset) => {"
            + FIND_SCROLLER
            + """
                const top = window.__kpiView.root.getBoundingClientRect().top;
                return findScroller().scrollTop + top - offset;
            }""",
            offset,
        )

    async def results_scroll_top(self) -> float:
        """Return the scroll position that brings the last results table into view."""
        return await self.page.evaluate(
            "() => {"
            + FIND_SCROLLER
            + """
                const scroller = findScroller();
                const tables = [];
                const walk = (root) => root.querySelectorAll('*').forEach((node) => {
                    if (node.tagName === 'TABLE') tables.push(node);
                    if (node.shadowRoot) walk(node.shadowRoot);
                });
                walk(document);
                const bottom = Math.max(...tables.map((t) => t.getBoundingClientRect().bottom));
                const target = scroller.scrollTop + bottom + 24 - window.innerHeight;
                return Math.max(0, Math.min(target, scroller.scrollHeight - scroller.clientHeight));
            }"""
        )

    async def wait_for_results(self) -> None:
        """Wait until the reactive Python cells show the widget's current counts."""
        await self.page.wait_for_function(
            """() => {
                const v = window.__kpiView;
                const format = (n) => n.toLocaleString('en-US');
                const expected = v.selection
                    ? `(${format(v.selectedCount)} rows)`
                    : `${format(v.activeCount)} of ${format(v.data.rows)} rows remain`;
                const roots = [document.body];
                const collect = (root) => root.querySelectorAll('*').forEach((node) => {
                    if (node.shadowRoot) {
                        roots.push(node.shadowRoot);
                        collect(node.shadowRoot);
                    }
                });
                collect(document);
                return roots.some((root) => root.textContent.includes(expected));
            }""",
            polling=100,
            timeout=10_000,
        )

    async def show_results(self, hold: float, seconds: float = 0.8) -> None:
        """Move the cursor off the matrix, then scroll down to the updated results."""
        await self.wait_for_results()
        await self.glide(RESULTS_CURSOR, seconds=0.3)
        await self.scroll_to(await self.results_scroll_top(), seconds=seconds)
        await self.hold(hold)

    async def status(self) -> str:
        return await self.page.locator(".kpi-status").inner_text()


async def wait_for_widget(page: Page) -> None:
    await page.wait_for_function(
        "() => window.__kpiStats?.firstPaint !== undefined", timeout=60_000
    )
    await page.wait_for_timeout(250)


async def take_overview(director: Director) -> None:
    await director.hold(0.5)
    await director.scroll_to(await director.widget_scroll_top(), seconds=0.8)
    await director.hold(2.0)
    await director.click("Lower")
    await director.hold(0.8)
    await director.click("Full")
    await director.hold(0.4)
    await director.show_results(hold=1.3)


async def take_lasso(director: Director) -> None:
    await director.scroll_to(await director.widget_scroll_top(), seconds=0.01)
    await director.hold(0.5)
    await director.click("Lasso")
    await director.hold(0.15)
    await director.lasso(SINR, RSRP, (-86.0, 2.2), (15.0, 5.4), seconds=1.8)
    await director.hold(0.7)
    await director.show_results(hold=2.2)


async def take_filter_stack(director: Director) -> None:
    await director.scroll_to(await director.widget_scroll_top(), seconds=0.01)
    await director.hold(0.1)
    y = (await director.data_point(LATENCY, LATENCY, 0, 150))[1]
    start = (await director.data_point(LATENCY, LATENCY, 18, 0))[0]
    end = (await director.data_point(LATENCY, LATENCY, 190, 0))[0]
    await director.drag_line((start, y), (end, y), seconds=0.55)
    await director.hold(0.15)
    await director.click("Keep")
    await director.hold(0.2)
    await director.page.keyboard.press("l")
    await director.lasso(SINR, RSRP, (-112.0, 1.5), (13.0, 7.2), seconds=0.7, steps=40)
    await director.click("Exclude")
    await director.hold(0.15)
    await director.show_results(hold=1.25, seconds=0.55)
    await director.page.keyboard.press("r")
    await director.wait_for_results()
    await director.hold(0.45)


async def take_python_sync(director: Director) -> None:
    await director.scroll_to(await director.widget_scroll_top(), seconds=0.01)
    await director.hold(0.4)
    await director.click("Lasso")
    await director.lasso(SINR, RSRP, (-86.0, 2.2), (15.0, 5.4), seconds=1.4)
    await director.hold(0.4)
    await director.show_results(hold=1.5)
    await director.page.keyboard.press("x")
    await director.wait_for_results()
    await director.hold(1.8)


async def take_cluster_vs_truth(director: Director) -> None:
    await director.scroll_to(await director.widget_scroll_top(), seconds=0.01)
    await director.hold(0.3)
    await director.choose_color("cluster")
    await director.hold(0.5)
    await director.click("Lasso")
    await director.lasso(SINR, RSRP, (-86.0, 2.2), (15.0, 5.4), seconds=1.3)
    await director.hold(0.4)
    await director.show_results(hold=2.2)


@dataclass(frozen=True)
class Shot:
    """One GIF in the shot list."""

    name: str
    scene: Callable[[Director], Awaitable[None]]


SHOTS: tuple[Shot, ...] = (
    Shot("01_overview", take_overview),
    Shot("02_lasso_interference", take_lasso),
    Shot("03_filter_stack", take_filter_stack),
    Shot("04_python_sync", take_python_sync),
    Shot("05_cluster_vs_truth", take_cluster_vs_truth),
)


def convert(ffmpeg: str, source: Path, name: str, start: float, duration: float) -> None:
    """Export a take as a two-pass palette GIF and an H.264 MP4."""
    GIF_DIR.mkdir(parents=True, exist_ok=True)
    palette = TAKE_DIR / f"{name}_palette.png"
    clip = ["-ss", f"{start:.2f}", "-t", f"{duration:.2f}", "-i", str(source)]
    scale = "fps=15,scale=900:-1:flags=lanczos"
    quiet = [ffmpeg, "-y", "-loglevel", "error"]
    subprocess.run(
        [*quiet, *clip, "-vf", f"{scale},palettegen=stats_mode=diff", str(palette)], check=True
    )
    subprocess.run(
        [
            *quiet,
            *clip,
            "-i",
            str(palette),
            "-lavfi",
            f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
            str(GIF_DIR / f"{name}.gif"),
        ],
        check=True,
    )
    subprocess.run(
        [
            *quiet,
            *clip,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(GIF_DIR / f"{name}.mp4"),
        ],
        check=True,
    )


async def record_shot(context: BrowserContext, shot: Shot, port: int, ffmpeg: str) -> None:
    """Record one take in a fresh page and export it."""
    page = await context.new_page()
    opened = time.monotonic()
    await page.goto(f"http://127.0.0.1:{port}/", wait_until="domcontentloaded")
    await wait_for_widget(page)
    director = Director(page)
    started = time.monotonic() - opened
    await shot.scene(director)
    finished = time.monotonic() - opened
    status = await director.status()
    video = page.video
    await page.close()
    await context.close()

    source = TAKE_DIR / f"{shot.name}.webm"
    await video.save_as(source)
    await video.delete()
    convert(ffmpeg, source, shot.name, max(0.0, started - 0.15), finished - started + 0.15)
    size = (GIF_DIR / f"{shot.name}.gif").stat().st_size / 1e6
    print(f"{shot.name}: {finished - started:.1f} s, {size:.1f} MB GIF, final status: {status}")


async def record(names: set[str] | None, port: int) -> None:
    """Record the selected shots against a running app."""
    ffmpeg = find_ffmpeg()
    TAKE_DIR.mkdir(parents=True, exist_ok=True)
    server = start_app(port)
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            for shot in SHOTS:
                if names and shot.name not in names:
                    continue
                context = await browser.new_context(
                    viewport=VIEWPORT,
                    device_scale_factor=2,
                    record_video_dir=str(TAKE_DIR / "raw"),
                    record_video_size=VIEWPORT,
                )
                await context.add_init_script(CURSOR_SCRIPT)
                await record_shot(context, shot, port, ffmpeg)
            await browser.close()
    finally:
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
    shutil.rmtree(TAKE_DIR / "raw", ignore_errors=True)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", action="append", choices=[shot.name for shot in SHOTS])
    parser.add_argument("--port", type=int, default=2718)
    args = parser.parse_args()
    asyncio.run(record(set(args.only) if args.only else None, args.port))


if __name__ == "__main__":
    main()
