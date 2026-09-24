#!/usr/bin/env python3
"""Publish the next queued clip in queue/pending/ to Instagram as a Reel.

Intended to run inside the GitHub Actions workflow at
.github/workflows/post-reel.yml, which checks out the repo, runs this
script, then commits the queue changes (moving the posted clip into
queue/posted/ and appending to the log) back to the repo.
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests

GRAPH_API_VERSION = "v20.0"
GRAPH_BASE = f"https://graph.instagram.com/{GRAPH_API_VERSION}"

REPO_ROOT = Path(__file__).resolve().parent.parent
PENDING_DIR = REPO_ROOT / "queue" / "pending"
POSTED_DIR = REPO_ROOT / "queue" / "posted"
LOG_PATH = POSTED_DIR / "log.csv"

POLL_INTERVAL_SECONDS = 10
POLL_TIMEOUT_SECONDS = 600  # 10 minutes; give large reels time to process


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def next_clip() -> Optional[Path]:
    """Pick the next clip alphabetically. Prefix filenames (001_, 002_, ...)
    to control posting order."""
    clips = sorted(PENDING_DIR.glob("*.mp4"))
    return clips[0] if clips else None


def raw_url(repo_slug: str, branch: str, relative_path: Path) -> str:
    return f"https://raw.githubusercontent.com/{repo_slug}/{branch}/{relative_path.as_posix()}"


def load_caption(clip_path: Path) -> str:
    sidecar = clip_path.with_suffix(".json")
    if not sidecar.exists():
        return ""
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    return data.get("caption", "")


def create_media_container(ig_user_id: str, access_token: str, video_url: str, caption: str) -> str:
    resp = requests.post(
        f"{GRAPH_BASE}/{ig_user_id}/media",
        data={
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "access_token": access_token,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def wait_until_ready(container_id: str, access_token: str) -> None:
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        resp = requests.get(
            f"{GRAPH_BASE}/{container_id}",
            params={"fields": "status_code", "access_token": access_token},
            timeout=30,
        )
        resp.raise_for_status()
        status = resp.json().get("status_code")
        if status == "FINISHED":
            return
        if status == "ERROR":
            raise RuntimeError(f"Instagram failed to process container {container_id}")
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Container {container_id} did not finish processing in time")


def publish(ig_user_id: str, access_token: str, container_id: str) -> str:
    resp = requests.post(
        f"{GRAPH_BASE}/{ig_user_id}/media_publish",
        data={"creation_id": container_id, "access_token": access_token},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def log_result(clip_path: Path, media_id: str, caption: str) -> None:
    is_new = not LOG_PATH.exists()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp_utc", "filename", "media_id", "caption"])
        writer.writerow([datetime.datetime.utcnow().isoformat(), clip_path.name, media_id, caption])


def move_to_posted(clip_path: Path) -> None:
    POSTED_DIR.mkdir(parents=True, exist_ok=True)
    for path in (clip_path, clip_path.with_suffix(".json")):
        if path.exists():
            subprocess.run(
                ["git", "mv", str(path), str(POSTED_DIR / path.name)],
                check=True,
                cwd=REPO_ROOT,
            )


def main() -> int:
    clip = next_clip()
    if clip is None:
        print("No clips waiting in queue/pending/. Nothing to post.")
        return 0

    ig_user_id = env("IG_BUSINESS_ACCOUNT_ID")
    access_token = env("IG_ACCESS_TOKEN")
    repo_slug = env("GITHUB_REPOSITORY")  # owner/repo, set automatically by GitHub Actions
    branch = os.environ.get("GITHUB_REF_NAME", "main")

    caption = load_caption(clip)
    video_url = raw_url(repo_slug, branch, clip.relative_to(REPO_ROOT))

    print(f"Posting {clip.name} -> {video_url}")
    container_id = create_media_container(ig_user_id, access_token, video_url, caption)
    print(f"Created container {container_id}, waiting for processing...")
    wait_until_ready(container_id, access_token)
    media_id = publish(ig_user_id, access_token, container_id)
    print(f"Published as media {media_id}")

    log_result(clip, media_id, caption)
    move_to_posted(clip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
