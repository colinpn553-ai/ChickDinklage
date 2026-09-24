#!/usr/bin/env python3
"""Generate a short narrated history explainer with crude scene-by-scene
2D animation and burned-in captions, and drop it into queue/review_pending/
for a human to approve before it ever reaches the posting bot.

Unlike the (removed) fictional horror-story pipeline, this one covers real
people and real events, so it deliberately does NOT auto-publish: see
.github/workflows/generate-history-clip.yml and approve-history-clip.yml.

Pipeline:
1. Anthropic API writes a title + a sequence of short narration "beats",
   each tagged with one scene from a fixed vocabulary (SCENES below).
2. Piper TTS (local, offline, free) synthesizes the full narration once.
3. faster-whisper transcribes that same audio locally to recover
   word-level timestamps -- used both for burned-in captions and for
   timing which crude scene illustration is on screen at any moment.
4. Frames are drawn with Pillow: whichever scene is active at each video
   frame's timestamp gets rendered, so the visuals track what's being
   said instead of showing one static image/character throughout.
5. ffmpeg muxes frames + narration audio + burned SRT captions + a title
   card and outro card into the final vertical Reel.
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import requests
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
REVIEW_DIR = REPO_ROOT / "queue" / "review_pending"
PENDING_DIR = REPO_ROOT / "queue" / "pending"
POSTED_DIR = REPO_ROOT / "queue" / "posted"

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
FPS = 10
FONT_PATH = os.environ.get(
    "DRAWTEXT_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
)
OUTRO_TEXT = "FOLLOW FOR MORE"
OUTRO_DURATION_SECONDS = 3.0
CAPTION_WORDS_PER_CHUNK = 4

PIPER_VOICE_URL_BASE = os.environ.get(
    "PIPER_VOICE_URL_BASE",
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium",
)

HISTORY_TOPICS = [
    "How Pol Pot and the Khmer Rouge came to power in Cambodia",
    "The Ottoman-era origins of Bosnia's Muslim population",
    "The causes of the Rwandan genocide",
    "How the Berlin Wall came to be built and why it fell",
    "The partition of India and Pakistan in 1947",
    "How the Cuban Missile Crisis unfolded",
    "The rise and fall of the Soviet Union",
    "The causes of World War One's outbreak in 1914",
]

SCENE_VOCAB = [
    "jungle", "building", "crowd", "soldiers", "map",
    "meeting", "leader", "fire", "prison", "mosque",
    "church", "exodus",
]

HISTORY_PROMPT = """You are writing a short, factual history explainer for a narrated \
vertical video (like a documentary voiceover), on this topic:

{topic}

Requirements:
- Serious, neutral, documentary tone. This is education, not entertainment -- \
never comedic, sensationalized, or flippant, especially regarding violence or \
death.
- Factually careful: prefer well-established mainstream historical consensus, \
note complexity/nuance briefly rather than oversimplifying, and do not invent \
specific facts, quotes, or figures you are not confident about.
- Do not include graphic descriptions of violence. Reference tragic events \
with gravity and respect for victims, without dwelling on graphic detail.
- 130-190 words total, split into 5-8 short "beats" (1-2 sentences each) that \
move through the topic roughly chronologically.
- For each beat, pick exactly one "scene" tag from this fixed list that best \
matches what the beat is describing: {scenes}
  (jungle = rural/countryside settings; building = government/capital; \
crowd = protests/gatherings/populations; soldiers = military/conflict; \
map = borders/geography/territory changes; meeting = negotiations/planning; \
leader = a ruler or leading figure being discussed abstractly, never a named \
likeness; fire = destruction/war damage, kept non-graphic; prison = \
detention/oppression; mosque = Islamic religious context; church = \
Christian religious context; exodus = displacement/refugees fleeing.)

Respond with ONLY a JSON object, no other text, in this exact shape:
{{"title": "Short punchy title, no quotes", "beats": [{{"narration": "...", \
"scene": "one_of_the_scene_tags"}}, ...]}}
"""


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def generate_script(topic: str) -> tuple[str, list[dict]]:
    api_key = env("ANTHROPIC_API_KEY")
    prompt = HISTORY_PROMPT.format(topic=topic, scenes=", ".join(SCENE_VOCAB))
    resp = requests.post(
        ANTHROPIC_API_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        },
        json={
            "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5"),
            "max_tokens": 1200,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=90,
    )
    resp.raise_for_status()
    content_blocks = resp.json()["content"]
    text_blocks = [b["text"] for b in content_blocks if b.get("type") == "text"]
    if not text_blocks:
        raise RuntimeError(f"No text content block in response: {content_blocks!r}")
    text = "\n".join(text_blocks).strip()

    # Models sometimes wrap JSON in a ```json fence despite instructions.
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    data = json.loads(text)
    title = data["title"].strip()
    beats = data["beats"]
    for beat in beats:
        if beat.get("scene") not in SCENE_VOCAB:
            beat["scene"] = "map"  # safe fallback for an unrecognized tag
    return title, beats


def synthesize_narration(full_text: str, voice_dir: Path, out_path: Path) -> None:
    from piper import PiperVoice

    voice = PiperVoice.load(
        str(voice_dir / "voice.onnx"), config_path=str(voice_dir / "voice.onnx.json")
    )
    with wave.open(str(out_path), "wb") as wav_file:
        voice.synthesize_wav(full_text, wav_file)


def download_piper_voice(voice_dir: Path) -> None:
    voice_dir.mkdir(parents=True, exist_ok=True)
    for suffix, name in ((".onnx", "voice.onnx"), (".onnx.json", "voice.onnx.json")):
        resp = requests.get(f"{PIPER_VOICE_URL_BASE}{suffix}", timeout=120)
        resp.raise_for_status()
        (voice_dir / name).write_bytes(resp.content)


def transcribe_words(audio_path: Path) -> list[tuple[float, float, str]]:
    from faster_whisper import WhisperModel

    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(str(audio_path), word_timestamps=True)
    words: list[tuple[float, float, str]] = []
    for seg in segments:
        for w in seg.words:
            words.append((w.start, w.end, w.word.strip()))
    return words


def srt_timestamp(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def build_captions_srt(words: list[tuple[float, float, str]], out_path: Path) -> None:
    lines = []
    idx = 1
    for i in range(0, len(words), CAPTION_WORDS_PER_CHUNK):
        chunk = words[i:i + CAPTION_WORDS_PER_CHUNK]
        start, end = chunk[0][0], chunk[-1][1]
        text = " ".join(w[2] for w in chunk)
        lines.append(f"{idx}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{text}\n")
        idx += 1
    out_path.write_text("\n".join(lines), encoding="utf-8")


def assign_beat_times(
    beats: list[dict], words: list[tuple[float, float, str]]
) -> list[dict]:
    """Split the flat whisper word list across beats in proportion to each
    beat's own word count, so each beat gets a (start, end) time range even
    if whisper's tokenization doesn't exactly match ours."""
    word_counts = [max(len(b["narration"].split()), 1) for b in beats]
    total_words = sum(word_counts)
    if not words:
        # Degenerate fallback: spread evenly with no audio-derived timing.
        cursor = 0.0
        for beat, count in zip(beats, word_counts):
            dur = count / total_words if total_words else 0
            beat["start"], beat["end"] = cursor, cursor + dur
            cursor += dur
        return beats

    total_duration = words[-1][1]
    cursor_idx = 0
    n_words = len(words)
    for beat, count in zip(beats, word_counts):
        start_idx = cursor_idx
        end_idx = min(cursor_idx + round(count / total_words * n_words), n_words)
        end_idx = max(end_idx, start_idx + 1)
        start_t = words[start_idx][0] if start_idx < n_words else total_duration
        end_t = words[min(end_idx, n_words) - 1][1] if end_idx > 0 else total_duration
        beat["start"], beat["end"] = start_t, end_t
        cursor_idx = end_idx
    beats[-1]["end"] = total_duration
    return beats


# --- Crude scene illustrations -------------------------------------------
# Every function draws onto `draw` (a PIL ImageDraw for a VIDEO_WIDTH x
# VIDEO_HEIGHT canvas) using only `t` (seconds) for animation. Deliberately
# crude: thin outline shapes, no fill detail, no attempt at real likenesses.

LINE = (235, 225, 210)


def _stick_figure(draw, x, y, t, phase=0.0, scale=1.0, arms_up=False):
    r = 22 * scale
    draw.ellipse([x - r, y, x + r, y + 2 * r], outline=LINE, width=3)
    body_top = y + 2 * r
    body_bot = body_top + 70 * scale
    draw.line([(x, body_top), (x, body_bot)], fill=LINE, width=3)
    if arms_up:
        draw.line([(x, body_top + 15 * scale), (x - 30 * scale, body_top - 15 * scale)], fill=LINE, width=3)
        draw.line([(x, body_top + 15 * scale), (x + 30 * scale, body_top - 15 * scale)], fill=LINE, width=3)
    else:
        sway = math.sin(t * 2 + phase) * 8 * scale
        draw.line([(x, body_top + 15 * scale), (x - 25 * scale + sway, body_top + 40 * scale)], fill=LINE, width=3)
        draw.line([(x, body_top + 15 * scale), (x + 25 * scale - sway, body_top + 40 * scale)], fill=LINE, width=3)
    step = math.sin(t * 3 + phase) * 15 * scale
    draw.line([(x, body_bot), (x - 20 * scale + step, body_bot + 55 * scale)], fill=LINE, width=3)
    draw.line([(x, body_bot), (x + 20 * scale - step, body_bot + 55 * scale)], fill=LINE, width=3)


def draw_jungle(draw, t, W, H):
    for i, x in enumerate([W * 0.15, W * 0.37, W * 0.63, W * 0.85]):
        sway = math.sin(t * 1.5 + i) * 6
        base_y = H * 0.42
        draw.polygon([(x + sway, base_y), (x - 45, base_y + 180), (x + 45, base_y + 180)], outline=LINE, width=4)
        draw.line([(x + sway, base_y), (x + sway, base_y + 180)], fill=LINE, width=4)
    _stick_figure(draw, W * 0.42, H * 0.58, t, phase=0)
    _stick_figure(draw, W * 0.56, H * 0.58, t, phase=1.2)


def draw_building(draw, t, W, H):
    bx0, by0, bx1, by1 = W * 0.28, H * 0.32, W * 0.72, H * 0.62
    draw.rectangle([bx0, by0, bx1, by1], outline=LINE, width=5)
    n_cols = 4
    for col in range(n_cols):
        cx = bx0 + (bx1 - bx0) * (col + 0.5) / n_cols
        draw.line([(cx, by0 + 20), (cx, by1)], fill=LINE, width=4)
    draw.polygon([(bx0 - 20, by0), (bx1 + 20, by0), ((bx0 + bx1) / 2, by0 - 80)], outline=LINE, width=5)
    flag_sway = math.sin(t * 3) * 8
    pole_x = (bx0 + bx1) / 2
    draw.line([(pole_x, by0 - 80), (pole_x, by0 - 160)], fill=LINE, width=3)
    draw.polygon([(pole_x, by0 - 160), (pole_x + 50 + flag_sway, by0 - 145), (pole_x, by0 - 130)], outline=LINE, width=3)


def draw_crowd(draw, t, W, H):
    positions = [(0.25, 0), (0.4, 0.6), (0.55, 0), (0.7, 0.6), (0.35, 1.1), (0.6, 1.1)]
    for i, (fx, off) in enumerate(positions):
        arms_up = i % 2 == 0
        _stick_figure(draw, W * fx, H * 0.5 + off * 20, t, phase=i * 0.7, scale=0.85, arms_up=arms_up)


def draw_soldiers(draw, t, W, H):
    for i, fx in enumerate([0.32, 0.5, 0.68]):
        x, y = W * fx, H * 0.5
        r = 20
        draw.ellipse([x - r, y, x + r, y + 2 * r], outline=LINE, width=3)
        draw.line([(x - r, y + 4), (x + r, y + 4)], fill=LINE, width=3)  # helmet line
        body_top, body_bot = y + 2 * r, y + 2 * r + 70
        draw.line([(x, body_top), (x, body_bot)], fill=LINE, width=3)
        draw.line([(x, body_top + 20), (x + 50, body_top + 5)], fill=LINE, width=4)  # rifle
        step = math.sin(t * 2 + i) * 10
        draw.line([(x, body_bot), (x - 18 + step, body_bot + 55)], fill=LINE, width=3)
        draw.line([(x, body_bot), (x + 18 - step, body_bot + 55)], fill=LINE, width=3)


def draw_map(draw, t, W, H):
    cx, cy = W / 2, H * 0.5
    pts = []
    for i in range(10):
        ang = i / 10 * 2 * math.pi
        rad = 180 + 20 * math.sin(ang * 3 + t * 0.5)
        pts.append((cx + rad * math.cos(ang), cy + rad * 0.7 * math.sin(ang)))
    draw.polygon(pts, outline=LINE, width=4)
    dash_phase = int(t * 4) % 2
    draw.line([(cx, cy - 150), (cx, cy + 150)], fill=LINE, width=(4 if dash_phase else 2))
    draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=LINE)


def draw_meeting(draw, t, W, H):
    tx0, ty, tx1 = W * 0.25, H * 0.55, W * 0.75
    draw.line([(tx0, ty), (tx1, ty)], fill=LINE, width=5)
    for i, fx in enumerate([0.32, 0.45, 0.58, 0.7]):
        _stick_figure(draw, W * fx, ty - 90, t, phase=i * 0.9, scale=0.6)


def draw_leader(draw, t, W, H):
    cx, cy = W / 2, H * 0.42
    bob = math.sin(t * 1.5) * 4
    r = 90
    draw.ellipse([cx - r, cy - r + bob, cx + r, cy + r + bob], outline=LINE, width=5)
    draw.rectangle([cx - 140, cy + r + bob, cx + 140, cy + r + 260 + bob], outline=LINE, width=5)
    draw.rectangle([cx - 160, cy + r + 260 + bob, cx + 160, cy + r + 290 + bob], outline=LINE, width=4)


def draw_fire(draw, t, W, H):
    bx0, by0, bx1, by1 = W * 0.3, H * 0.4, W * 0.7, H * 0.62
    jag = [(bx0, by0)]
    for i in range(5):
        jag.append((bx0 + (bx1 - bx0) * i / 4, by0 - 20 * (i % 2)))
    jag.append((bx1, by0))
    draw.rectangle([bx0, by0, bx1, by1], outline=LINE, width=4)
    for i in range(3):
        fx = bx0 + (bx1 - bx0) * (i + 0.5) / 3
        flick = math.sin(t * 8 + i) * 10
        draw.polygon(
            [(fx, by0), (fx - 20, by0 - 60 + flick), (fx, by0 - 100 + flick), (fx + 20, by0 - 60 + flick)],
            outline=LINE, width=3,
        )


def draw_prison(draw, t, W, H):
    bx0, by0, bx1, by1 = W * 0.3, H * 0.4, W * 0.7, H * 0.65
    draw.rectangle([bx0, by0, bx1, by1], outline=LINE, width=4)
    for i in range(6):
        bx = bx0 + (bx1 - bx0) * i / 5
        draw.line([(bx, by0 - 20), (bx, by1 + 20)], fill=LINE, width=4)
    _stick_figure(draw, (bx0 + bx1) / 2, by0 + 20, t, scale=0.6)


def draw_mosque(draw, t, W, H):
    cx, base_y = W / 2, H * 0.55
    draw.rectangle([cx - 100, base_y, cx + 100, base_y + 150], outline=LINE, width=4)
    draw.arc([cx - 100, base_y - 100, cx + 100, base_y + 20], 180, 360, fill=LINE, width=4)
    draw.arc([cx - 15, base_y - 190, cx + 15, base_y - 160], 200, 520, fill=LINE, width=3)


def draw_church(draw, t, W, H):
    cx, base_y = W / 2, H * 0.55
    draw.rectangle([cx - 90, base_y, cx + 90, base_y + 150], outline=LINE, width=4)
    draw.polygon([(cx - 20, base_y - 120), (cx + 20, base_y - 120), (cx, base_y - 190)], outline=LINE, width=4)
    draw.line([(cx, base_y - 190), (cx, base_y - 230)], fill=LINE, width=3)
    draw.line([(cx - 15, base_y - 215), (cx + 15, base_y - 215)], fill=LINE, width=3)


def draw_exodus(draw, t, W, H):
    y = H * 0.55
    for i in range(5):
        x = ((t * 60 + i * 90) % (W + 150)) - 75
        _stick_figure(draw, x, y, t, phase=i * 0.5, scale=0.65)
        draw.ellipse([x - 10, y + 50, x + 15, y + 70], outline=LINE, width=2)  # bundle


SCENES = {
    "jungle": draw_jungle,
    "building": draw_building,
    "crowd": draw_crowd,
    "soldiers": draw_soldiers,
    "map": draw_map,
    "meeting": draw_meeting,
    "leader": draw_leader,
    "fire": draw_fire,
    "prison": draw_prison,
    "mosque": draw_mosque,
    "church": draw_church,
    "exodus": draw_exodus,
}


def render_frames(beats: list[dict], duration: float, frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    n_frames = int(duration * FPS) + 1
    BG = (35, 10, 10)
    for i in range(n_frames):
        t = i / FPS
        scene_fn = SCENES["map"]
        for beat in beats:
            if beat["start"] <= t < beat["end"] or (beat is beats[-1] and t >= beat["start"]):
                scene_fn = SCENES.get(beat["scene"], SCENES["map"])
                break
        img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), BG)
        draw = ImageDraw.Draw(img)
        scene_fn(draw, t, VIDEO_WIDTH, VIDEO_HEIGHT)
        img.save(frames_dir / f"frame_{i:05d}.png")


def title_fontsize(title: str) -> int:
    length = len(title)
    if length <= 16:
        return 84
    if length <= 24:
        return 68
    if length <= 32:
        return 56
    return 44


def assemble_video(
    title: str,
    frames_dir: Path,
    narration_path: Path,
    captions_path: Path,
    duration: float,
    out_path: Path,
    work_dir: Path,
) -> None:
    outro_start = max(duration - OUTRO_DURATION_SECONDS, 0.0)
    title_file = work_dir / "title.txt"
    title_file.write_text(title, encoding="utf-8")
    font_posix = Path(FONT_PATH).as_posix()

    vf = (
        f"subtitles={captions_path.as_posix()}:force_style='FontName=DejaVu Sans,"
        f"FontSize=16,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        f"BorderStyle=1,Outline=2,Alignment=2,MarginV=100',"
        f"drawtext=fontfile={font_posix}:textfile={title_file.as_posix()}:"
        f"fontsize={title_fontsize(title)}:fontcolor=white:borderw=3:"
        f"bordercolor=black:box=1:boxcolor=black@0.35:boxborderw=20:"
        f"x=(w-text_w)/2:y=120:"
        f"alpha='if(lt(t\\,1)\\,t\\,if(lt(t\\,3)\\,1\\,if(lt(t\\,4)\\,4-t\\,0)))',"
        f"drawtext=fontfile={font_posix}:text='{OUTRO_TEXT}':fontsize=40:"
        f"fontcolor=white:borderw=3:bordercolor=black:box=1:"
        f"boxcolor=black@0.35:boxborderw=16:x=(w-text_w)/2:y=h-200:"
        f"alpha='if(lt(t\\,{outro_start})\\,0\\,"
        f"if(lt(t\\,{outro_start + 1})\\,t-{outro_start}\\,1))'"
    )

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(FPS), "-i", str(frames_dir / "frame_%05d.png"),
            "-i", str(narration_path),
            "-vf", vf,
            "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
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
    existing = (
        list(REVIEW_DIR.glob("*.mp4"))
        + list(PENDING_DIR.glob("*.mp4"))
        + list(POSTED_DIR.glob("*.mp4"))
    )
    return len(existing) + 1


def main() -> int:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    topic = os.environ.get("HISTORY_TOPIC") or random.choice(HISTORY_TOPICS)
    print(f"Topic: {topic}")

    title, beats = generate_script(topic)
    print(f"Generated: {title} ({len(beats)} beats)")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        narration_path = tmp_path / "narration.wav"
        captions_path = tmp_path / "captions.srt"
        frames_dir = tmp_path / "frames"
        voice_dir = tmp_path / "voice"

        download_piper_voice(voice_dir)
        print("Piper voice downloaded")

        full_text = " ".join(b["narration"] for b in beats)
        synthesize_narration(full_text, voice_dir, narration_path)
        print("Narration synthesized")

        words = transcribe_words(narration_path)
        duration = words[-1][1] if words else 0.0
        print(f"Transcribed {len(words)} words, duration={duration:.2f}s")

        build_captions_srt(words, captions_path)
        beats = assign_beat_times(beats, words)

        render_frames(beats, duration, frames_dir)
        print(f"Rendered {len(list(frames_dir.glob('*.png')))} animation frames")

        base_name = f"{next_index():03d}_{slugify(title)}"
        out_video = REVIEW_DIR / f"{base_name}.mp4"
        assemble_video(title, frames_dir, narration_path, captions_path, duration, out_video, tmp_path)
        print(f"Assembled {out_video.name}")

    caption = f"{title} \U0001f4dc #history #historytok #didyouknow #education"
    (REVIEW_DIR / f"{base_name}.json").write_text(
        json.dumps({"caption": caption, "topic": topic}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Queued for review: {base_name}.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
