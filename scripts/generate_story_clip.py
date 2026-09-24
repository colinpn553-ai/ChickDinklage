#!/usr/bin/env python3
"""Generate an original short horror story, narrate it with TTS, lay it over
licensed horror-themed stock footage, and drop the result into
queue/pending/ for scripts/post_to_instagram.py to publish.

Runs unattended in GitHub Actions on its own schedule (see
.github/workflows/generate-story-clip.yml), ahead of the posting workflow.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import requests
from gtts import gTTS

REPO_ROOT = Path(__file__).resolve().parent.parent
PENDING_DIR = REPO_ROOT / "queue" / "pending"
POSTED_DIR = REPO_ROOT / "queue" / "posted"

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
FONT_PATH = os.environ.get(
    "DRAWTEXT_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
)
OUTRO_TEXT = "FOLLOW FOR MORE"
OUTRO_DURATION_SECONDS = 3.0

STORY_PROMPT = """Write an original short horror story for a narrated Instagram Reel.

Requirements:
- 130-180 words, first or second person, unsettling and atmospheric, building to a creepy final line
- Fully original: do not reference real people, real tragedies/disasters, real place names tied to \
real events, or any copyrighted characters or franchises
- Suitable for a general audience: unsettling, not graphic or gory

Respond with exactly two lines and nothing else:
Line 1: a short punchy title (no quotes, no "Title:" prefix)
Line 2: the full story text, as a single paragraph
"""


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def generate_story() -> tuple[str, str]:
    api_key = env("ANTHROPIC_API_KEY")
    resp = requests.post(
        ANTHROPIC_API_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        },
        json={
            "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5"),
            "max_tokens": 500,
            "messages": [{"role": "user", "content": STORY_PROMPT}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    content_blocks = resp.json()["content"]
    # Some models emit a "thinking" block ahead of the actual "text" block,
    # so pick out the text block(s) rather than assuming content[0] is it.
    text_blocks = [b["text"] for b in content_blocks if b.get("type") == "text"]
    if not text_blocks:
        raise RuntimeError(f"No text content block in response: {content_blocks!r}")
    text = "\n".join(text_blocks).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"Unexpected story generation response: {text!r}")
    title = lines[0]
    story = " ".join(lines[1:])
    return title, story


def synthesize_narration(story_text: str, out_path: Path) -> None:
    gTTS(text=story_text, lang="en", slow=False).save(str(out_path))


def get_audio_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def title_fontsize(title: str) -> int:
    length = len(title)
    if length <= 16:
        return 84
    if length <= 24:
        return 68
    if length <= 32:
        return 56
    return 44


def assemble_clip(title: str, narration_path: Path, out_path: Path, work_dir: Path) -> None:
    """Render an animated procedural background (two color layers slowly
    cross-fading, with grain and a vignette), overlay a fading title card at
    the start and a "follow for more" card at the end, and mux in the
    narration audio. Everything here is generated, not sourced from any
    external footage."""
    duration = get_audio_duration(narration_path)
    outro_start = max(duration - OUTRO_DURATION_SECONDS, 0.0)

    # drawtext's inline `text=` option collides with the filter graph's own
    # colon syntax, so the (LLM-generated, unpredictable) title is written
    # to a file and read via `textfile=` instead of escaped inline.
    title_file = work_dir / "title.txt"
    title_file.write_text(title, encoding="utf-8")

    filter_complex = (
        f"[0:v][1:v]blend=all_expr='A*(0.5+0.5*sin(2*PI*T/8))+"
        f"B*(0.5-0.5*sin(2*PI*T/8))'[bg];"
        f"[bg]noise=alls=20:allf=t+u,eq=contrast=1.15:brightness=-0.02,"
        f"vignette=PI/4,format=yuv420p[bg2];"
        f"[bg2]drawtext=fontfile={FONT_PATH}:textfile={title_file.as_posix()}:"
        f"fontsize={title_fontsize(title)}:fontcolor=white:borderw=3:"
        f"bordercolor=black:box=1:boxcolor=black@0.35:boxborderw=20:"
        f"x=(w-text_w)/2:y=(h-text_h)/2:"
        f"alpha='if(lt(t\\,1)\\,t\\,if(lt(t\\,3)\\,1\\,if(lt(t\\,4)\\,4-t\\,0)))'[v1];"
        f"[v1]drawtext=fontfile={FONT_PATH}:text='{OUTRO_TEXT}':fontsize=40:"
        f"fontcolor=white:borderw=3:bordercolor=black:box=1:"
        f"boxcolor=black@0.35:boxborderw=16:x=(w-text_w)/2:y=h-200:"
        f"alpha='if(lt(t\\,{outro_start})\\,0\\,"
        f"if(lt(t\\,{outro_start + 1})\\,t-{outro_start}\\,1))'[v2]"
    )

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=0x1a0010:s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:d={duration}:r=30",
            "-f", "lavfi", "-i", f"color=c=0x02030a:s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:d={duration}:r=30",
            "-i", str(narration_path),
            "-filter_complex", filter_complex,
            "-map", "[v2]", "-map", "2:a",
            "-c:v", "libx264", "-c:a", "aac",
            "-t", str(duration),
            str(out_path),
        ],
        check=True,
    )


def slugify(title: str) -> str:
    slug = "".join(c.lower() if c.isalnum() else "_" for c in title)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")[:40]


def next_index() -> int:
    existing = list(PENDING_DIR.glob("*.mp4")) + list(POSTED_DIR.glob("*.mp4"))
    return len(existing) + 1


def main() -> int:
    PENDING_DIR.mkdir(parents=True, exist_ok=True)

    title, story = generate_story()
    print(f"Generated story: {title}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        narration_path = tmp_path / "narration.mp3"

        synthesize_narration(story, narration_path)
        print("Narration synthesized")

        base_name = f"{next_index():03d}_{slugify(title)}"
        out_video = PENDING_DIR / f"{base_name}.mp4"
        assemble_clip(title, narration_path, out_video, tmp_path)
        print(f"Assembled {out_video.name}")

    caption = (
        f"{title} \U0001f47b #horror #scarystory #creepy #horrorstories "
        f"#nosleep #scary #horrortok #storytime"
    )
    (PENDING_DIR / f"{base_name}.json").write_text(
        json.dumps({"caption": caption}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Queued {base_name}.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
