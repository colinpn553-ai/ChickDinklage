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
import random
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

PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"

HORROR_FOOTAGE_KEYWORDS = [
    "dark forest fog",
    "abandoned house night",
    "empty hallway dark",
    "old cemetery fog",
    "creepy basement",
    "foggy woods night",
    "abandoned asylum",
    "dark attic",
    "old mirror dark room",
    "flickering light hallway",
]

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
    text = resp.json()["content"][0]["text"].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"Unexpected story generation response: {text!r}")
    title = lines[0]
    story = " ".join(lines[1:])
    return title, story


def synthesize_narration(story_text: str, out_path: Path) -> None:
    gTTS(text=story_text, lang="en", slow=False).save(str(out_path))


def fetch_stock_footage(out_path: Path) -> None:
    api_key = env("PEXELS_API_KEY")
    query = random.choice(HORROR_FOOTAGE_KEYWORDS)
    resp = requests.get(
        PEXELS_SEARCH_URL,
        headers={"Authorization": api_key},
        params={"query": query, "orientation": "portrait", "per_page": 15},
        timeout=30,
    )
    resp.raise_for_status()
    videos = resp.json().get("videos", [])
    if not videos:
        raise RuntimeError(f"No Pexels results for query: {query!r}")
    video = random.choice(videos)
    files = sorted(video["video_files"], key=lambda f: f.get("height", 0), reverse=True)
    video_url = files[0]["link"]
    video_resp = requests.get(video_url, timeout=120)
    video_resp.raise_for_status()
    out_path.write_bytes(video_resp.content)


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


def assemble_clip(footage_path: Path, narration_path: Path, out_path: Path) -> None:
    duration = get_audio_duration(narration_path)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", str(footage_path),
            "-i", str(narration_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
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
        footage_path = tmp_path / "footage.mp4"

        synthesize_narration(story, narration_path)
        print("Narration synthesized")

        fetch_stock_footage(footage_path)
        print("Stock footage downloaded")

        base_name = f"{next_index():03d}_{slugify(title)}"
        out_video = PENDING_DIR / f"{base_name}.mp4"
        assemble_clip(footage_path, narration_path, out_video)
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
