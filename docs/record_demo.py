"""Record docs/demo.gif: replay alarm -> investigation report -> human approval in the Streamlit UI.

Needs the local stack from the README (PostgreSQL, fabguard-init, the API, a worker with
FABGUARD_LLM_PROVIDER=offline, and `streamlit run app.py`), the audited recordings and the
seed-17 model, plus ffmpeg and Google Chrome. Then:

    uv run --with playwright python docs/record_demo.py
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

VIEWPORT = {"width": 1280, "height": 800}


def scroll(page: Page, pixels: int, steps: int = 6) -> None:
    page.mouse.move(800, 450)
    for _ in range(steps):
        page.mouse.wheel(0, pixels / steps)
        page.wait_for_timeout(150)


def record(page: Page, url: str, recording_id: str) -> None:
    page.goto(url)
    page.get_by_text("FabGuard AI").first.wait_for()
    page.wait_for_timeout(1500)

    # 1. Replay a faulty recording until the detector alarms.
    page.get_by_role("combobox").first.click()
    page.keyboard.type(recording_id, delay=40)
    page.keyboard.press("Enter")
    page.wait_for_timeout(800)
    page.get_by_role("button", name="Run recorded-data replay").click()
    page.get_by_text("Anomaly score").wait_for(timeout=60_000)
    page.wait_for_timeout(2500)
    scroll(page, 450)
    page.wait_for_timeout(2500)
    page.get_by_role("button", name="Submit replay for investigation").click()
    page.get_by_text("Submitted incident").wait_for(timeout=30_000)
    page.wait_for_timeout(2500)

    # 2. Load the worker's investigation report for that incident.
    page.locator('[data-testid="stSidebar"]').get_by_text("Incident", exact=True).click()
    refresh = page.get_by_role("button", name="Load / refresh incident")
    refresh.wait_for()
    page.wait_for_timeout(1000)
    for _ in range(30):
        refresh.click()
        page.wait_for_timeout(1500)
        if page.get_by_text("Report revision").count():
            break
    else:
        raise RuntimeError("the worker produced no report; is fabguard-worker running?")
    page.wait_for_timeout(2000)
    for _ in range(4):
        scroll(page, 350)
        page.wait_for_timeout(1500)

    # 3. A human reviewer approves the concrete ticket draft.
    approve = page.get_by_role("button", name="Approve concrete ticket draft")
    approve.scroll_into_view_if_needed()
    page.wait_for_timeout(1500)
    approve.click()
    page.get_by_text("Ticket transaction completed").wait_for(timeout=30_000)
    scroll(page, 400)
    page.wait_for_timeout(3500)


def to_gif(video: Path, gif: Path, fps: int, width: int) -> None:
    filters = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf",
         f"{filters},split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer",
         str(gif)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--recording", default="uored-ea1d6c921bbd8312")
    parser.add_argument("--output", type=Path, default=Path("docs/demo.gif"))
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--width", type=int, default=960)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory, sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome")
        context = browser.new_context(viewport=VIEWPORT, record_video_dir=directory,
                                      record_video_size=VIEWPORT)
        page = context.new_page()
        try:
            record(page, args.url, args.recording)
        finally:
            context.close()
            browser.close()
        to_gif(Path(page.video.path()), args.output, args.fps, args.width)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
