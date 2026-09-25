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
   said instead of showing one static image/character throughout. People
   are pre-made illustrated sprites from assets/characters/base/, cast by
   role (rural workers, officials, uniformed personnel, ...) and era, with
   the faces varying per clip.
5. ffmpeg muxes frames + narration audio + burned SRT captions + a title
   card and outro card into the final vertical Reel.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
CAPTION_WORDS_PER_CHUNK = 3

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

- Do not glorify or promote any regime, ideology, or leader. Describe what they \
did, how they gained and used power, and the consequences, plainly and \
accurately.
- Also pick one "era" for the illustrations' clothing: "early_1900s" if the \
topic is set mostly before about 1930, otherwise "modern".
- Also pick one "setting": "europe" if the events take place mainly in Europe \
in the period discussed (up to about 1945), so the illustrations draw only \
characters who fit that setting; otherwise "global".
- For a beat whose scene is "map", also add "places": a list of 1-4 ISO 3166-1 \
alpha-2 codes (e.g. "KH") for the PRESENT-DAY countries the beat is about, main \
one first. The map shows present-day borders, so use today's countries even if \
the period had different ones (e.g. for the former Yugoslavia, list the \
successor countries).
- For a beat whose scene is "building", "meeting" or "leader", you may add \
"flag": a lowercase ISO alpha-2 code, but ONLY if you are confident that \
country used essentially the same national flag as today's during the period \
discussed. Do NOT set it for regimes or eras with a different flag (e.g. \
Cambodia under the Khmer Rouge, Nazi Germany, the Soviet Union, or Bosnia \
before 1998). When unsure, omit "flag". Never add "flag" to other scenes.

Respond with ONLY a JSON object, no other text, in this exact shape:
{{"title": "Short punchy title, no quotes", "era": "modern_or_early_1900s", \
"setting": "europe_or_global", "beats": [{{"narration": "...", "scene": "one_of_the_scene_tags"}}, ...]}}
(map beats may also carry "places": ["XX"]; building/meeting/leader beats may \
also carry "flag": "xx" under the rules above.)
"""


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def generate_script(topic: str) -> tuple[str, str, str, list[dict]]:
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
            "max_tokens": 4096,
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
    era = data.get("era") if data.get("era") in ("modern", "early_1900s") else "modern"
    setting = data.get("setting") if data.get("setting") in ("europe", "global") else "global"
    beats = data["beats"]
    for beat in beats:
        if beat.get("scene") not in SCENE_VOCAB:
            beat["scene"] = "map"  # safe fallback for an unrecognized tag
        # Codes are validated against the real data files; anything the
        # model invented is silently dropped rather than drawn.
        raw_places = beat.get("places") if beat["scene"] == "map" else None
        beat["places"] = [c.upper() for c in (raw_places or []) if valid_place(c)][:4]
        flag = beat.get("flag") if beat["scene"] in ("building", "meeting", "leader") else None
        beat["flag"] = flag.lower() if valid_flag(flag) else None
    return title, era, setting, beats


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


_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _two_digit_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")


def _year_words(year: int) -> str:
    """1975 -> 'nineteen seventy-five', 1906 -> 'nineteen oh six',
    1900 -> 'nineteen hundred' -- how years are actually said aloud,
    rather than a TTS engine's default 'one thousand nine hundred...'."""
    if year == 2000:
        return "two thousand"  # the one exception to the "X hundred" pattern
    century, remainder = divmod(year, 100)
    century_word = _two_digit_words(century)
    if remainder == 0:
        return f"{century_word} hundred"
    if remainder < 10:
        return f"{century_word} oh {_ONES[remainder]}"
    return f"{century_word} {_two_digit_words(remainder)}"


_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def spell_out_years(text: str) -> str:
    return _YEAR_RE.sub(lambda m: _year_words(int(m.group(0))), text)


def align_text_to_timestamps(
    text: str, whisper_words: list[tuple[float, float, str]]
) -> list[tuple[float, float, str]]:
    """Pair OUR known-correct script text with Whisper's audio timestamps,
    proportionally by position. Whisper is only used for timing here --
    the displayed words always come from what we actually wrote, so
    captions can never carry a mis-transcription, regardless of how
    accurately Whisper heard the audio."""
    tokens = text.split()
    n_tokens = len(tokens)
    n_whisper = len(whisper_words)
    if n_tokens == 0:
        return []
    if n_whisper == 0:
        return [(0.0, 0.0, tok) for tok in tokens]
    aligned = []
    for i, tok in enumerate(tokens):
        idx = min(int(i * n_whisper / n_tokens), n_whisper - 1)
        start, end, _ = whisper_words[idx]
        aligned.append((start, end, tok))
    return aligned


def build_captions_srt(words: list[tuple[float, float, str]], out_path: Path) -> None:
    lines = []
    idx = 1
    prev_end = 0.0
    for i in range(0, len(words), CAPTION_WORDS_PER_CHUNK):
        chunk = words[i:i + CAPTION_WORDS_PER_CHUNK]
        # Whisper's per-word timestamps are occasionally imprecise right at
        # a sentence-boundary pause, which can otherwise produce a chunk
        # that starts before the previous one ends (two caption cards
        # briefly overlapping on screen). Clamp to strictly increasing.
        start = max(chunk[0][0], prev_end)
        end = max(chunk[-1][1], start + 0.1)
        text = " ".join(w[2] for w in chunk)
        lines.append(f"{idx}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{text}\n")
        idx += 1
        prev_end = end
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
# Every function draws a FULL composed scene (sky, sun/moon, hills, ground,
# then foreground content) onto `draw`, using only `t` (seconds) for
# animation. Deliberately crude line-art -- but a full environment per
# scene, not one icon on a flat color, so the frame reads as a place with
# things happening in it rather than a static graphic.

LINE = (235, 225, 210)
HORIZON_FRAC = 0.62


def _lerp_color(c0, c1, frac):
    return tuple(int(c0[i] + (c1[i] - c0[i]) * frac) for i in range(3))


def draw_backdrop(draw, W, H, sky_top, sky_horizon, ground_color,
                   hills_color=None, sun=False, sun_color=(255, 205, 130), bands=18):
    """Sky gradient + optional sun + optional hills + ground plane. Returns
    horizon_y so callers can place foreground content grounded on it."""
    horizon_y = H * HORIZON_FRAC
    band_h = horizon_y / bands
    for i in range(bands):
        frac = i / bands
        color = _lerp_color(sky_top, sky_horizon, frac)
        draw.rectangle([0, band_h * i, W, band_h * (i + 1) + 1], fill=color)

    if sun:
        sx, sy, r = W * 0.78, horizon_y * 0.32, 68
        for i, rr in enumerate([r * 1.7, r * 1.3, r]):
            shade = tuple(min(255, c + i * 18) for c in sun_color)
            draw.ellipse([sx - rr, sy - rr, sx + rr, sy + rr], fill=shade)

    if hills_color:
        pts = [(0, horizon_y)]
        for i in range(7):
            x = W * i / 6
            yoff = 45 * math.sin(i * 1.7 + 0.4)
            pts.append((x, horizon_y - 35 - yoff))
        pts.append((W, horizon_y))
        draw.polygon(pts, fill=hills_color)

    draw.rectangle([0, horizon_y, W, H], fill=ground_color)
    return horizon_y


def _shade(color, factor=0.72):
    return tuple(max(0, int(c * factor)) for c in color)


def _tint(color, factor=1.18):
    return tuple(min(255, int(c * factor)) for c in color)


TRUNK_COLOR = (92, 62, 40)
CANOPY_COLORS = [(70, 120, 60), (60, 105, 55), (85, 130, 65)]

SKIN_TONES = [(235, 200, 165), (205, 165, 125), (150, 105, 75), (100, 70, 50), (245, 215, 190)]
HAIR_COLORS = [(35, 25, 22), (65, 45, 30), (20, 18, 20), (95, 68, 42), (45, 42, 44)]
SHIRT_COLORS = [(95, 150, 140), (195, 95, 75), (80, 80, 95), (190, 150, 60), (125, 100, 150), (70, 115, 150)]
PANTS_COLOR = (48, 42, 52)
SHOE_COLOR = (28, 24, 26)
PROP_COLOR = (96, 76, 52)


def _tree(draw, x, base_y, t, phase=0.0, height=180, canopy=None):
    canopy = canopy or CANOPY_COLORS[int(phase) % len(CANOPY_COLORS)]
    sway = math.sin(t * 1.5 + phase) * 6
    top_x = x + sway
    trunk_w = height * 0.06
    draw.polygon(
        [(x - trunk_w, base_y), (x + trunk_w, base_y), (top_x + trunk_w * 0.5, base_y - height * 0.55)],
        fill=TRUNK_COLOR,
    )
    canopy_r = height * 0.4
    cy = base_y - height * 0.65
    draw.ellipse([top_x - canopy_r, cy - canopy_r, top_x + canopy_r, cy + canopy_r], fill=canopy)
    draw.ellipse([top_x - canopy_r * 0.55, cy - canopy_r * 0.6, top_x + canopy_r * 1.05, cy + canopy_r * 0.5],
                 fill=_shade(canopy, 0.8))


def _stick_figure(draw, x, foot_y, t, phase=0.0, scale=1.0, arms_up=False,
                   seed=0, hat=None, robe=False, prop=None):
    """x, foot_y = ground position (feet). Figure is built upward from there.

    seed: picks a consistent skin/hair/shirt color combo for this figure
          (vary it per figure in a scene for visual diversity).
    hat: None, "cap", or "helmet"
    robe: draw a tapered, filled robe/dress silhouette instead of separate
          legs/pants
    prop: None, "bag", "staff", "banner", "book", or "rifle" -- something
          simple held in/near the swinging hand, kept generic rather than
          any specific real-world costume, since the same scene types get
          reused across many unrelated topics.
    """
    skin = SKIN_TONES[seed % len(SKIN_TONES)]
    hair = HAIR_COLORS[(seed * 7 + 3) % len(HAIR_COLORS)]
    shirt = SHIRT_COLORS[(seed * 5 + 1) % len(SHIRT_COLORS)]

    total_h = 169 * scale
    y = foot_y - total_h  # head-top
    r = 24 * scale
    body_top = y + 2 * r - 4 * scale
    body_bot = body_top + 74 * scale

    sway = math.sin(t * 2 + phase) * 8 * scale
    if arms_up:
        l_hand = (x - 30 * scale, body_top - 15 * scale)
        r_hand = (x + 30 * scale, body_top - 15 * scale)
    else:
        l_hand = (x - 25 * scale + sway, body_top + 42 * scale)
        r_hand = (x + 25 * scale - sway, body_top + 42 * scale)
    arm_w = max(int(9 * scale), 3)
    draw.line([(x - 10 * scale, body_top + 6 * scale), l_hand], fill=skin, width=arm_w)
    draw.line([(x + 10 * scale, body_top + 6 * scale), r_hand], fill=skin, width=arm_w)

    if robe:
        hem = 36 * scale
        draw.polygon(
            [(x - 16 * scale, body_bot), (x + 16 * scale, body_bot),
             (x + hem, body_bot + 62 * scale), (x - hem, body_bot + 62 * scale)],
            fill=shirt,
        )
        draw.polygon(
            [(x, body_bot), (x + 16 * scale, body_bot),
             (x + hem, body_bot + 62 * scale), (x + hem * 0.15, body_bot + 62 * scale)],
            fill=_shade(shirt),
        )
        foot_by = body_bot + 62 * scale
        draw.ellipse([x - 16 * scale, foot_by - 4 * scale, x, foot_by + 8 * scale], fill=SHOE_COLOR)
        draw.ellipse([x, foot_by - 4 * scale, x + 16 * scale, foot_by + 8 * scale], fill=SHOE_COLOR)
    else:
        step = math.sin(t * 3 + phase) * 15 * scale
        l_foot = (x - 20 * scale + step, body_bot + 55 * scale)
        r_foot = (x + 20 * scale - step, body_bot + 55 * scale)
        leg_w = max(int(11 * scale), 4)
        draw.line([(x - 8 * scale, body_bot), l_foot], fill=PANTS_COLOR, width=leg_w)
        draw.line([(x + 8 * scale, body_bot), r_foot], fill=PANTS_COLOR, width=leg_w)
        for fx, fy in (l_foot, r_foot):
            draw.ellipse([fx - 9 * scale, fy - 4 * scale, fx + 9 * scale, fy + 7 * scale], fill=SHOE_COLOR)

    draw.rounded_rectangle([x - 22 * scale, body_top, x + 22 * scale, body_bot], radius=10 * scale, fill=shirt)
    draw.rectangle([x + 6 * scale, body_top + 4 * scale, x + 22 * scale, body_bot], fill=_shade(shirt))

    draw.ellipse([x - r, y, x + r, y + 2 * r], fill=skin)
    draw.ellipse([x + r * 0.15, y, x + r, y + 2 * r], fill=_shade(skin, 0.88))

    if hat == "cap":
        draw.pieslice([x - r - 4 * scale, y - 15 * scale, x + r + 4 * scale, y + 9 * scale], 180, 360, fill=hair)
    elif hat == "helmet":
        draw.pieslice([x - r - 3 * scale, y - 7 * scale, x + r + 3 * scale, y + 11 * scale], 180, 360,
                       fill=(120, 125, 112))
        draw.rectangle([x - r - 5 * scale, y + 3 * scale, x + r + 5 * scale, y + 8 * scale], fill=(95, 100, 88))
    else:
        draw.pieslice([x - r - 2 * scale, y - 9 * scale, x + r + 2 * scale, y + r * 0.75], 180, 360, fill=hair)
        for sx in (-0.45, 0.0, 0.45):
            spike_x = x + sx * r
            draw.polygon(
                [(spike_x - 7 * scale, y + 2 * scale), (spike_x, y - 15 * scale), (spike_x + 7 * scale, y + 2 * scale)],
                fill=hair,
            )

    if scale >= 0.7:
        ey = y + r * 1.15
        draw.ellipse([x - 9 * scale, ey - 4 * scale, x - 1 * scale, ey + 4 * scale], fill=(35, 28, 28))
        draw.ellipse([x + 1 * scale, ey - 4 * scale, x + 9 * scale, ey + 4 * scale], fill=(35, 28, 28))

    hand = r_hand
    if prop == "bag":
        bx, by = l_hand[0] - 4 * scale, l_hand[1] + 6 * scale
        draw.line([(x - 12 * scale, body_top + 8 * scale), (bx, by - 8 * scale)], fill=PROP_COLOR, width=3)
        draw.ellipse([bx - 15 * scale, by - 10 * scale, bx + 15 * scale, by + 12 * scale], fill=PROP_COLOR)
    elif prop == "staff":
        draw.line([hand, (hand[0] + 4 * scale, foot_y)], fill=PROP_COLOR, width=4)
    elif prop == "banner":
        top = (hand[0], hand[1] - 70 * scale)
        draw.line([hand, top], fill=PROP_COLOR, width=3)
        draw.rectangle([top[0], top[1], top[0] + 32 * scale, top[1] + 24 * scale], fill=SHIRT_COLORS[(seed + 2) % len(SHIRT_COLORS)])
    elif prop == "book":
        draw.rectangle([x - 15 * scale, body_top + 22 * scale, x + 15 * scale, body_top + 36 * scale], fill=(160, 70, 60))
    elif prop == "rifle":
        draw.line([(x, body_top + 20 * scale), (x + 50 * scale, body_top + 5 * scale)], fill=(50, 45, 40), width=5)


def _building(draw, cx, base_y, t, width=240, height=330, color=(150, 140, 130), flag=False):
    bx0, bx1 = cx - width / 2, cx + width / 2
    by0, by1 = base_y - height, base_y
    draw.rectangle([bx0, by0, bx1, by1], fill=color)
    draw.rectangle([cx, by0, bx1, by1], fill=_shade(color))

    n_cols = max(int(width / 60), 2)
    win_color = _tint(color, 1.3) if sum(color) < 550 else (255, 235, 170)
    for col in range(n_cols):
        cxi = bx0 + width * (col + 0.5) / n_cols
        for row_y in [by0 + height * 0.28, by0 + height * 0.58]:
            ww = width / n_cols * 0.4
            draw.rectangle([cxi - ww / 2, row_y, cxi + ww / 2, row_y + height * 0.16], fill=win_color)
    draw.rectangle([cx - width * 0.06, by1 - height * 0.22, cx + width * 0.06, by1], fill=_shade(color, 0.6))

    draw.polygon([(bx0 - 20, by0), (bx1 + 20, by0), (cx, by0 - width * 0.3)], fill=_shade(color, 0.55))
    if flag:
        real = SCENE_CTX["flag"] if valid_flag(SCENE_CTX["flag"]) else None
        pole_top = by0 - width * 0.3 - (170 if real else 80)
        draw.line([(cx, by0 - width * 0.3), (cx, pole_top)], fill=(210, 200, 190), width=4)
        if real:
            _paste_flag(real, cx, pole_top, 190, t)
        else:
            flag_sway = math.sin(t * 3) * 8
            draw.polygon([(cx, pole_top), (cx + 50 + flag_sway, pole_top + 15), (cx, pole_top + 30)],
                         fill=(190, 70, 60))


# --- Character cast (pre-made illustrated sprites) -----------------------
# assets/characters/base/*.png are transparent, 1000px-tall standing
# characters. Scenes pick people by ROLE so, e.g., soldiers scenes get the
# uniformed characters and rural scenes get farmers -- and the pick is
# seeded per clip (CAST["seed"]) so different clips show different faces.
# If the asset folder is missing, _person() falls back to the drawn
# stick figures so generation never breaks.

CAST_DIR = REPO_ROOT / "assets" / "characters" / "base"

ROLE_POOLS = {
    "civilian": ["elder_man_vest", "woman_red_cardigan", "young_man_hoodie",
                 "woman_yellow_blouse", "man_teal_polo"],
    "rural": ["farmer_overalls", "woman_apron", "teen_boy_bag", "elder_woman_shawl"],
    "official": ["man_navy_suit", "woman_blazer", "older_man_cream_suit", "clerk_vest_bowtie"],
    "military": ["soldier_helmet", "woman_soldier_beret", "officer_peaked_cap", "recruit_bush_hat"],
    "period": ["man_bowler_1910s", "woman_hat_dress_1910s", "boy_newsboy_cap", "man_tunic_robe"],
}
ROLE_POOLS["mixed"] = ROLE_POOLS["civilian"] + ROLE_POOLS["rural"] + ROLE_POOLS["official"]
ROLE_POOLS["urban"] = ROLE_POOLS["civilian"] + ROLE_POOLS["official"]

# Setting-aware casting. When a script is set in Europe, illustrations draw
# only from characters whose depicted appearance fits the period's
# population there, rather than an evenly mixed modern crowd -- the same
# reason early-1900s topics get period clothing. This is a plain
# historical-accuracy filter based on how each sprite looks; it is not
# used for "global" topics. (Small pools mean repeated faces; a bigger,
# setting-specific cast is the real fix.)
EUROPE_FIT = {
    "elder_man_vest", "woman_red_cardigan", "man_teal_polo", "farmer_overalls",
    "teen_boy_bag", "elder_woman_shawl", "man_navy_suit", "woman_blazer",
    "clerk_vest_bowtie", "soldier_helmet", "officer_peaked_cap",
    "man_bowler_1910s", "woman_hat_dress_1910s",
}
EUROPE_EARLY = {  # early-1900s Europe: period-plausible clothing AND appearance
    "civilian": ["man_bowler_1910s", "woman_hat_dress_1910s", "farmer_overalls", "elder_woman_shawl",
                 "clerk_vest_bowtie", "man_navy_suit", "elder_man_vest"],
    "official": ["man_navy_suit", "clerk_vest_bowtie", "man_bowler_1910s", "woman_hat_dress_1910s"],
    "rural": ["farmer_overalls", "elder_woman_shawl", "elder_man_vest"],
    "military": ["soldier_helmet", "officer_peaked_cap"],
}
EUROPE_EARLY["mixed"] = EUROPE_EARLY["urban"] = EUROPE_EARLY["civilian"]

CAST = {"seed": 0, "era": "modern", "setting": "global", "img": None, "sprites": None}
_SPRITE_CACHE: dict = {}


def _load_sprites() -> dict:
    if CAST["sprites"] is None:
        sprites = {}
        if CAST_DIR.is_dir():
            for f in sorted(CAST_DIR.glob("*.png")):
                sprites[f.stem] = Image.open(f).convert("RGBA")
        CAST["sprites"] = sprites
    return CAST["sprites"]


def _cast(role: str, n: int, salt: int = 0) -> list:
    """n character names for a role, deterministic per clip seed. Early-1900s
    clips swap civilian/rural/official roles for the period-dress pool."""
    pool = list(ROLE_POOLS[role])
    if CAST["era"] == "early_1900s" and role in ("civilian", "official", "rural", "mixed", "urban"):
        pool = ROLE_POOLS["period"] + (ROLE_POOLS["rural"][:2] if role in ("rural", "mixed") else [])
    if CAST["setting"] == "europe":
        if CAST["era"] == "early_1900s" and role in EUROPE_EARLY:
            pool = list(EUROPE_EARLY[role])
        else:
            pool = [n for n in pool if n in EUROPE_FIT] or pool
    rng = random.Random(CAST["seed"] * 1009 + salt)
    rng.shuffle(pool)
    return [pool[i % len(pool)] for i in range(n)]


def _sprite(name: str, height: int, flip: bool):
    key = (name, height, flip)
    if key not in _SPRITE_CACHE:
        base = _load_sprites()[name]
        im = base.resize((max(1, int(base.width * height / base.height)), height), Image.LANCZOS)
        _SPRITE_CACHE[key] = im.transpose(Image.FLIP_LEFT_RIGHT) if flip else im
    return _SPRITE_CACHE[key]


def _person(draw, name, x, foot_y, height, t, phase=0.0, flip=False, hop=0.0, seed=0):
    """Draw one character standing with its feet at (x, foot_y), `height` px
    tall, with a gentle idle bob (or a bigger walking hop when `hop` > 0)."""
    height = int(height)
    bob = math.sin(t * (7 if hop else 3.5) + phase) * (hop or 3)
    if name in _load_sprites() and CAST["img"] is not None:
        sp = _sprite(name, height, flip)
        CAST["img"].paste(sp, (int(x - sp.width / 2), int(foot_y - sp.height - abs(bob))), sp)
    else:
        _stick_figure(draw, x, foot_y - abs(bob), t, phase=phase, seed=seed, scale=height / 169)


# --- Scene palettes (sky top, sky horizon, ground, hills or None) --------

def draw_jungle(draw, t, W, H):
    horizon = draw_backdrop(draw, W, H, (110, 160, 210), (235, 200, 150), (70, 110, 55),
                             hills_color=(50, 85, 45), sun=True)
    for i, xf in enumerate([0.1, 0.26, 0.74, 0.9]):
        _tree(draw, W * xf, horizon + 60, t, phase=i, height=150 + 20 * (i % 2))
    for i, xf in enumerate([0.42, 0.5, 0.58]):
        _tree(draw, W * xf, horizon + 130, t, phase=i + 2, height=110)
    people = _cast("rural", 3, salt=1)
    for i, (xf, h) in enumerate([(0.3, 400), (0.7, 400), (0.5, 470)]):
        _person(draw, people[i], W * xf, horizon + 260, h, t, phase=i * 1.7, flip=(i == 1), seed=i)


def draw_building(draw, t, W, H):
    horizon = draw_backdrop(draw, W, H, (150, 190, 225), (220, 225, 220), (95, 95, 100), sun=True)
    _building(draw, W * 0.14, horizon + 40, t, width=260, height=400)
    _building(draw, W * 0.86, horizon + 40, t, width=280, height=440)
    _building(draw, W * 0.5, horizon + 60, t, width=460, height=620, flag=True)
    names = _cast("urban", 4, salt=2)  # one combined pick, so nobody appears twice
    lineup = [(0.2, names[0], False), (0.4, names[1], False),
              (0.62, names[2], True), (0.82, names[3], True)]
    for i, (xf, name, flip) in enumerate(lineup):
        _person(draw, name, W * xf, horizon + 250, 430, t, phase=i * 1.3, flip=flip, seed=i)


def draw_crowd(draw, t, W, H):
    horizon = draw_backdrop(draw, W, H, (140, 175, 215), (230, 210, 180), (110, 100, 90), sun=True)
    _building(draw, W * 0.5, horizon + 20, t, width=400, height=480)
    back = _cast("mixed", 7, salt=4)
    front = _cast("mixed", 5, salt=5)
    for i, xf in enumerate([0.1, 0.23, 0.36, 0.5, 0.63, 0.76, 0.9]):
        _person(draw, back[i], W * xf, horizon + 170, 330, t, phase=i * 0.9, flip=(i % 2 == 0), seed=i)
    for i, xf in enumerate([0.18, 0.34, 0.5, 0.66, 0.82]):
        _person(draw, front[i], W * xf, horizon + 270, 450, t, phase=i * 1.1 + 2, flip=(i % 2 == 1), seed=i + 3)


def draw_soldiers(draw, t, W, H):
    horizon = draw_backdrop(draw, W, H, (140, 130, 130), (200, 170, 140), (90, 80, 65), hills_color=(60, 55, 55))
    _building(draw, W * 0.13, horizon + 20, t, width=240, height=300, color=(180, 170, 165))
    back = _cast("military", 3, salt=6)
    front = _cast("military", 4, salt=7)
    for i, xf in enumerate([0.3, 0.5, 0.7]):
        _person(draw, back[i], W * xf, horizon + 170, 330, t, phase=i * 1.4, flip=(i % 2 == 1), seed=i)
    for i, xf in enumerate([0.16, 0.39, 0.62, 0.85]):
        _person(draw, front[i], W * xf, horizon + 270, 450, t, phase=i * 0.8, flip=(i % 2 == 0), seed=i + 4)


# --- Real maps (Natural Earth, public domain) and real flags (flag-icons, MIT)
# assets/geo/countries.json.gz holds present-day country outlines;
# assets/flags/<iso2>.png holds present-day national flags. The script
# names places/flags by ISO code, so borders and flags come from data,
# never from a model drawing them.

GEO_PATH = REPO_ROOT / "assets" / "geo" / "countries.json.gz"
FLAG_DIR = REPO_ROOT / "assets" / "flags"

# Per-beat context, set by render_frames before each scene is drawn.
SCENE_CTX = {"places": [], "flag": None, "progress": 0.0}
_GEO = {"countries": None}
_FLAG_CACHE: dict = {}


def _geo() -> dict:
    """iso2 -> {name, polys:[(bbox, area, ndarray)]} (outer rings only)."""
    if _GEO["countries"] is None:
        countries: dict = {}
        if GEO_PATH.is_file():
            with gzip.open(GEO_PATH, "rt", encoding="utf-8") as fh:
                for c in json.load(fh):
                    polys = []
                    for poly in c["p"]:
                        ring = np.array(poly[0], dtype=float)
                        if len(ring) < 3:
                            continue
                        x, y = ring[:, 0], ring[:, 1]
                        area = abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2
                        polys.append(((x.min(), y.min(), x.max(), y.max()), area, ring))
                    countries[c["i"]] = {"name": c["n"], "polys": polys}
        _GEO["countries"] = countries
    return _GEO["countries"]


def valid_place(code) -> bool:
    return isinstance(code, str) and code.upper() in _geo()


def valid_flag(code) -> bool:
    return isinstance(code, str) and code.isalpha() and (FLAG_DIR / f"{code.lower()}.png").is_file()


def _main_bbox(code: str):
    """Bounding box of a country's main landmass(es), ignoring far-flung
    islands/exclaves (e.g. Alaska, overseas territories) so the view frames
    the country people actually mean."""
    polys = _geo()[code]["polys"]
    biggest = max(p[1] for p in polys)
    keep = [p for p in polys if p[1] >= 0.35 * biggest]
    return (min(p[0][0] for p in keep), min(p[0][1] for p in keep),
            max(p[0][2] for p in keep), max(p[0][3] for p in keep))


def _flag_image(code: str, width: int):
    key = (code.lower(), width)
    if key not in _FLAG_CACHE:
        im = Image.open(FLAG_DIR / f"{code.lower()}.png").convert("RGBA")
        _FLAG_CACHE[key] = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    return _FLAG_CACHE[key]


def _paste_flag(code: str, x: float, y: float, width: int, t: float, wave: float = 6.0):
    """Paste a flag with its hoist edge at (x, y), rippling in strips."""
    img = CAST["img"]
    if img is None or not valid_flag(code):
        return False
    fl = _flag_image(code, width)
    strip = 6
    for sx in range(0, fl.width, strip):
        dy = math.sin(t * 4 + sx / 22.0) * wave * (sx / fl.width)
        piece = fl.crop((sx, 0, min(sx + strip, fl.width), fl.height))
        img.paste(piece, (int(x + sx), int(y + dy)), piece)
    return True


def _smoothstep(v: float) -> float:
    v = min(max(v, 0.0), 1.0)
    return v * v * (3 - 2 * v)


def draw_map(draw, t, W, H):
    OCEAN, LAND, BORDER = (58, 104, 150), (222, 210, 176), (150, 134, 104)
    HILITE, HILITE_EDGE = (200, 78, 60), (255, 236, 200)
    draw.rectangle([0, 0, W, H], fill=OCEAN)
    countries = _geo()
    places = [p for p in SCENE_CTX["places"] if p in countries]

    if places:
        boxes = [_main_bbox(p) for p in places]
        lon_min, lat_min = min(b[0] for b in boxes), min(b[1] for b in boxes)
        lon_max, lat_max = max(b[2] for b in boxes), max(b[3] for b in boxes)
        lon0, lat0 = (lon_min + lon_max) / 2, (lat_min + lat_max) / 2
        span_lon, span_lat = max(lon_max - lon_min, 6.0), max(lat_max - lat_min, 6.0)
        pad = 2.6 - 1.2 * _smoothstep(SCENE_CTX["progress"])  # slow zoom-in over the beat
    else:
        lon0, lat0, span_lon, span_lat, pad = 10.0, 15.0, 360.0, 150.0, 1.0
    cos0 = max(math.cos(math.radians(lat0)), 0.2)
    scale = min(W / (span_lon * pad * cos0), H * 0.9 / (span_lat * pad))
    cx, cy = W / 2, H * 0.45

    half_lon = (W / 2) / (scale * cos0) + 2
    half_lat = (H * 0.6) / scale + 2
    shifts = [0.0] + ([360.0] if lon0 > 90 else []) + ([-360.0] if lon0 < -90 else [])

    def project(ring, shift):
        px = cx + (ring[:, 0] + shift - lon0) * cos0 * scale
        py = cy + (lat0 - ring[:, 1]) * scale
        return list(zip(px.tolist(), py.tolist()))

    # faint graticule for a map-like feel
    step = 5 if span_lon * pad < 60 else 15
    g0 = int((lon0 - half_lon) // step) * step
    for gl in range(g0, int(lon0 + half_lon) + step, step):
        x = cx + (gl - lon0) * cos0 * scale
        draw.line([(x, 0), (x, H)], fill=(70, 118, 164), width=1)
    l0 = int((lat0 - half_lat) // step) * step
    for gl in range(l0, int(lat0 + half_lat) + step, step):
        y = cy + (lat0 - gl) * scale
        draw.line([(0, y), (W, y)], fill=(70, 118, 164), width=1)

    drawn = []
    for code, c in countries.items():
        for bbox, area, ring in c["polys"]:
            for sh in shifts:
                if (bbox[2] + sh < lon0 - half_lon or bbox[0] + sh > lon0 + half_lon
                        or bbox[3] < lat0 - half_lat or bbox[1] > lat0 + half_lat):
                    continue
                drawn.append((area, code, ring, sh))
    drawn.sort(key=lambda d: -d[0])  # big first so enclaves stay visible on top
    for area, code, ring, sh in drawn:
        pts = project(ring, sh)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if max(xs) - min(xs) < 2 and max(ys) - min(ys) < 2:
            continue
        hi = code in places
        draw.polygon(pts, fill=HILITE if hi else LAND, outline=HILITE_EDGE if hi else BORDER,
                     width=4 if hi else 2)

    try:
        font = ImageFont.truetype(FONT_PATH, 44)
        small = ImageFont.truetype(FONT_PATH, 30)
    except OSError:
        font = small = ImageFont.load_default()

    label_pts = []
    for n, code in enumerate(places):
        b = _main_bbox(code)
        lx = cx + ((b[0] + b[2]) / 2 - lon0) * cos0 * scale
        ly = cy + (lat0 - (b[1] + b[3]) / 2) * scale
        label_pts.append((lx, ly))
        r = 16 + 8 * (0.5 + 0.5 * math.sin(t * 4))
        draw.ellipse([lx - r, ly - r, lx + r, ly + r], outline=HILITE_EDGE, width=4)
        draw.ellipse([lx - 7, ly - 7, lx + 7, ly + 7], fill=HILITE_EDGE)
        # With several places, alternate labels above/below their markers
        # and shrink them a little so neighbours' names don't pile up.
        f = font if len(places) == 1 else small
        dy = -46 if n % 2 == 0 else 78
        draw.text((lx, ly + dy), countries[code]["name"], font=f, fill=(255, 255, 255),
                  anchor="ms", stroke_width=5, stroke_fill=(30, 30, 30))

    # Neighbours: label the larger ones that are on screen, so the view has context.
    for code, c in countries.items():
        if code in places:
            continue
        big = max(c["polys"], key=lambda p: p[1])
        bx = (big[0][0] + big[0][2]) / 2
        by = (big[0][1] + big[0][3]) / 2
        px = cx + (bx - lon0) * cos0 * scale
        py = cy + (lat0 - by) * scale
        wpx = (big[0][2] - big[0][0]) * cos0 * scale
        crowded = any((px - qx) ** 2 + (py - qy) ** 2 < 230 ** 2 for qx, qy in label_pts)
        if (0.08 * W < px < 0.92 * W and 0.08 * H < py < 0.62 * H and wpx > 190 and not crowded):
            draw.text((px, py), c["name"], font=small, fill=(60, 50, 35), anchor="mm",
                      stroke_width=3, stroke_fill=(232, 222, 190))

    draw.text((W / 2, H * 0.72), "Present-day borders", font=small, fill=(255, 255, 255),
              anchor="mm", stroke_width=4, stroke_fill=(30, 30, 30))

    flag = SCENE_CTX["flag"]
    if flag and valid_flag(flag):
        fl = _flag_image(flag, 340)
        x0, y0 = int(W / 2 - fl.width / 2), int(H * 0.08 + 250)
        draw.rectangle([x0 - 8, y0 - 8, x0 + fl.width + 8, y0 + fl.height + 8], fill=(250, 250, 245))
        CAST["img"].paste(fl, (x0, y0), fl)


def draw_meeting(draw, t, W, H):
    # Indoor scene: warm wall instead of sky, a window for depth.
    draw.rectangle([0, 0, W, H], fill=(70, 45, 35))
    floor_y = H * HORIZON_FRAC + 60
    draw.rectangle([0, floor_y, W, H], fill=(55, 35, 28))
    wx0, wy0, wx1, wy1 = W * 0.68, H * 0.18, W * 0.92, H * 0.4
    draw.rectangle([wx0, wy0, wx1, wy1], outline=LINE, width=3)
    draw.line([((wx0 + wx1) / 2, wy0), ((wx0 + wx1) / 2, wy1)], fill=LINE, width=2)
    if valid_flag(SCENE_CTX["flag"]):
        pole_x = W * 0.09
        draw.line([(pole_x, H * 0.24), (pole_x, floor_y + 40)], fill=(200, 190, 175), width=6)
        _paste_flag(SCENE_CTX["flag"], pole_x, H * 0.24, 300, t, wave=8)
    tx0, tx1 = W * 0.17, W * 0.83
    foot_y = floor_y + 70
    people = _cast("official", 4, salt=8)
    h = 480
    for i, xf in enumerate([0.27, 0.42, 0.58, 0.73]):
        _person(draw, people[i], W * xf, foot_y, h, t, phase=i * 0.9, flip=(i >= 2), seed=i)
    # The table is drawn over their lower bodies so they read as seated.
    top = foot_y - h * 0.42
    draw.rectangle([tx0, top, tx1, top + 26], fill=(150, 105, 65))
    draw.rectangle([tx0 + 12, top + 26, tx1 - 12, foot_y + 30], fill=(105, 70, 42))
    for px in (0.36, 0.5, 0.64):
        draw.rectangle([W * px - 40, top - 8, W * px + 40, top + 4], fill=(235, 230, 215))


def draw_leader(draw, t, W, H):
    horizon = draw_backdrop(draw, W, H, (90, 60, 70), (200, 110, 70), (55, 40, 40), sun=True,
                             sun_color=(230, 120, 90))
    audience = _cast("mixed", 6, salt=9)
    for i, xf in enumerate([0.1, 0.24, 0.37, 0.63, 0.76, 0.9]):
        _person(draw, audience[i], W * xf, horizon + 190, 300, t, phase=i * 1.3, flip=(xf > 0.5), seed=i + 1)
    cx = W / 2
    if valid_flag(SCENE_CTX["flag"]):
        pole_x = cx - 330
        draw.line([(pole_x, horizon - 330), (pole_x, horizon + 240)], fill=(200, 190, 175), width=7)
        _paste_flag(SCENE_CTX["flag"], pole_x, horizon - 330, 360, t, wave=9)
    ped_top = horizon + 230
    draw.rectangle([cx - 140, ped_top - 20, cx + 140, ped_top + 70], fill=(120, 110, 100))
    draw.rectangle([cx, ped_top - 20, cx + 140, ped_top + 70], fill=_shade((120, 110, 100)))
    # Generic authority figure -- deliberately not a specific likeness.
    leader = _cast("official", 1, salt=10)[0]
    _person(draw, leader, cx, ped_top - 20, 560, t, phase=0, seed=2)


def draw_fire(draw, t, W, H):
    horizon = draw_backdrop(draw, W, H, (60, 30, 30), (120, 50, 35), (40, 30, 28))
    bx0, by0, bx1 = W * 0.3, horizon - 190, W * 0.7
    by1 = horizon + 140
    draw.rectangle([bx0, by0, bx1, by1], fill=(70, 55, 50))
    draw.rectangle([(bx0 + bx1) / 2, by0, bx1, by1], fill=_shade((70, 55, 50)))
    draw.line([(bx0, by0), (bx0 + 30, by0 - 40)], fill=(50, 40, 38), width=4)
    draw.line([(bx1, by0), (bx1 - 40, by0 - 30)], fill=(50, 40, 38), width=4)
    for i in range(3):
        fx = bx0 + (bx1 - bx0) * (i + 0.5) / 3
        flick = math.sin(t * 8 + i) * 10
        draw.polygon(
            [(fx, by0), (fx - 20, by0 - 60 + flick), (fx, by0 - 100 + flick), (fx + 20, by0 - 60 + flick)],
            fill=(235, 140, 60),
        )
        draw.polygon(
            [(fx, by0), (fx - 10, by0 - 45 + flick), (fx, by0 - 75 + flick), (fx + 10, by0 - 45 + flick)],
            fill=(255, 210, 110),
        )
        smoke_y = by0 - 100 - ((t * 30 + i * 40) % 200)
        smoke_x = fx + math.sin(t * 1.2 + i) * 15
        smoke_alpha_gray = 120 + int(40 * (1 - ((t * 30 + i * 40) % 200) / 200))
        draw.ellipse([smoke_x - 12, smoke_y - 12, smoke_x + 12, smoke_y + 12],
                     outline=(smoke_alpha_gray,) * 3, width=2)
    onlookers = _cast("civilian", 2, salt=11)
    _person(draw, onlookers[0], W * 0.16, horizon + 270, 400, t, phase=0, seed=1)
    _person(draw, onlookers[1], W * 0.84, horizon + 270, 400, t, phase=2, flip=True, seed=3)


def draw_prison(draw, t, W, H):
    horizon = draw_backdrop(draw, W, H, (90, 95, 100), (150, 150, 145), (70, 68, 65))
    bx0, by0, bx1, by1 = W * 0.26, horizon - 210, W * 0.74, horizon + 230
    draw.rectangle([bx0, by0, bx1, by1], fill=(110, 108, 100))
    draw.rectangle([(bx0 + bx1) / 2, by0, bx1, by1], fill=_shade((110, 108, 100)))
    draw.rectangle([bx0 + 30, by0 + 40, bx1 - 30, by1 - 20], fill=(45, 42, 40))  # dark cell interior
    prisoner = _cast("civilian", 1, salt=12)[0]
    _person(draw, prisoner, (bx0 + bx1) / 2, by1 - 20, 340, t, phase=0, seed=1)
    for i in range(8):
        bx = bx0 + 30 + (bx1 - bx0 - 60) * i / 7
        draw.line([(bx, by0 + 40), (bx, by1 - 20)], fill=(30, 28, 26), width=8)
    draw.rectangle([bx0 - 110, by0 + 60, bx0 - 30, by1], fill=(90, 88, 82))  # watchtower


def draw_mosque(draw, t, W, H):
    horizon = draw_backdrop(draw, W, H, (110, 165, 215), (250, 210, 150), (190, 165, 110), sun=True)
    s = 1.9  # scale the building up so people stand at a believable size beside it
    cx, base_y = W / 2, horizon + 200
    wall = (225, 205, 165)
    gold = (190, 150, 80)
    top = base_y - 150 * s
    draw.rectangle([cx - 100 * s, top, cx + 100 * s, base_y], fill=wall)
    draw.rectangle([cx, top, cx + 100 * s, base_y], fill=_shade(wall))
    draw.pieslice([cx - 90 * s, top - 90 * s, cx + 90 * s, top + 90 * s], 180, 360, fill=gold)  # dome sits on the wall
    draw.line([(cx, top - 90 * s), (cx, top - 125 * s)], fill=(140, 110, 60), width=4)
    draw.pieslice([cx - 12 * s, top - 145 * s, cx + 12 * s, top - 121 * s], 200, 520, fill=gold)
    for mx in [cx - 150 * s, cx + 150 * s]:
        draw.rectangle([mx - 10 * s, base_y - 220 * s, mx + 10 * s, base_y], fill=wall)
        draw.polygon([(mx - 13 * s, base_y - 220 * s), (mx + 13 * s, base_y - 220 * s), (mx, base_y - 255 * s)], fill=gold)
    for wy in [base_y - 110 * s, base_y - 60 * s]:
        draw.rectangle([cx - 20 * s, wy, cx + 20 * s, wy + 35 * s], fill=(150, 115, 70))
    people = _cast("civilian", 3, salt=13)
    for i, (xf, flip) in enumerate([(0.17, False), (0.83, True), (0.68, True)]):
        _person(draw, people[i], W * xf, horizon + 290, 380, t, phase=i * 1.5, flip=flip, seed=i)


def draw_church(draw, t, W, H):
    horizon = draw_backdrop(draw, W, H, (130, 170, 210), (210, 210, 200), (70, 95, 60), hills_color=(45, 70, 40))
    s = 1.9
    cx, base_y = W / 2, horizon + 200
    wall = (200, 190, 175)
    draw.rectangle([cx - 90 * s, base_y - 150 * s, cx + 90 * s, base_y], fill=wall)
    draw.rectangle([cx, base_y - 150 * s, cx + 90 * s, base_y], fill=_shade(wall))
    draw.rectangle([cx - 28 * s, base_y - 270 * s, cx + 28 * s, base_y - 150 * s], fill=wall)  # bell tower
    draw.rectangle([cx, base_y - 270 * s, cx + 28 * s, base_y - 150 * s], fill=_shade(wall))
    draw.rectangle([cx - 8 * s, base_y - 250 * s, cx + 8 * s, base_y - 215 * s], fill=(70, 60, 55))
    draw.polygon([(cx - 34 * s, base_y - 270 * s), (cx + 34 * s, base_y - 270 * s), (cx, base_y - 350 * s)],
                 fill=(110, 60, 55))
    draw.line([(cx, base_y - 350 * s), (cx, base_y - 385 * s)], fill=(70, 60, 55), width=4)
    draw.line([(cx - 15 * s, base_y - 370 * s), (cx + 15 * s, base_y - 370 * s)], fill=(70, 60, 55), width=4)
    for wy in [base_y - 110 * s, base_y - 60 * s]:
        draw.rectangle([cx - 18 * s, wy, cx + 18 * s, wy + 32 * s], fill=(120, 150, 190))
    for i in range(5):
        fx = W * 0.08 + i * 40
        draw.line([(fx, base_y), (fx, base_y - 40)], fill=(90, 90, 85), width=4)  # simple fence
    people = _cast("civilian", 3, salt=14)
    for i, (xf, flip) in enumerate([(0.17, False), (0.83, True), (0.68, True)]):
        _person(draw, people[i], W * xf, horizon + 290, 380, t, phase=i * 1.5, flip=flip, seed=i)


def draw_exodus(draw, t, W, H):
    horizon = draw_backdrop(draw, W, H, (110, 100, 100), (170, 140, 110), (95, 80, 65), hills_color=(65, 55, 50))
    draw.polygon([(W * 0.3, horizon - 20), (W * 0.4, horizon - 90), (W * 0.55, horizon - 20)],
                 outline=(80, 70, 65), width=3)  # distant damaged silhouette, far and small
    walkers = _cast("mixed", 7, salt=15)
    n = 7
    span = W + 300
    for i in range(n):
        x = ((t * 55 + i * (span / n)) % span) - 150
        # A steady walking hop; smaller/further ones sit higher up the road.
        far = i % 2 == 0
        _person(draw, walkers[i], x, horizon + (220 if far else 270), 300 if far else 380, t,
                phase=i * 0.7, hop=9, seed=i + 2)


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

# Slow "Ken Burns" zoom/pan: scenes are drawn oversized and cropped to the
# final frame, drifting over each beat's duration, so nothing sits static
# on screen even for a still-image-style composition.
ZOOM_MARGIN = 0.16


def render_frames(beats: list[dict], duration: float, frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    n_frames = int(duration * FPS) + 1
    big_W = int(VIDEO_WIDTH * (1 + ZOOM_MARGIN))
    big_H = int(VIDEO_HEIGHT * (1 + ZOOM_MARGIN))
    max_offset_x = big_W - VIDEO_WIDTH
    max_offset_y = big_H - VIDEO_HEIGHT

    for i in range(n_frames):
        t = i / FPS
        # Hold the most recent beat that has started. Whisper's word timing
        # leaves small gaps between beats, and a frame inside a gap must
        # keep showing the previous scene rather than flashing a fallback.
        beat_idx = 0
        for bi, beat in enumerate(beats):
            if beat["start"] <= t:
                beat_idx = bi
        active_beat = beats[beat_idx] if beats else {"start": 0, "end": duration, "scene": "map"}
        scene_fn = SCENES.get(active_beat["scene"], SCENES["map"])

        img = Image.new("RGB", (big_W, big_H), (35, 10, 10))
        draw = ImageDraw.Draw(img)
        CAST["img"] = img  # lets scenes paste character sprites onto this frame
        b_start, b_end = active_beat["start"], active_beat["end"]
        b_dur = max(b_end - b_start, 0.01)
        local_frac = min(max((t - b_start) / b_dur, 0.0), 1.0)
        SCENE_CTX["places"] = active_beat.get("places") or []
        SCENE_CTX["flag"] = active_beat.get("flag")
        SCENE_CTX["progress"] = local_frac
        scene_fn(draw, t, big_W, big_H)
        # Alternate zoom-in / zoom-out and pan corner by beat for variety.
        progress = local_frac if beat_idx % 2 == 0 else (1 - local_frac)
        corner_x = 1.0 if beat_idx % 3 in (0, 1) else 0.0
        corner_y = 1.0 if beat_idx % 2 == 0 else 0.0
        # Gentle drift only -- centered by default, nudged toward a corner
        # by at most ~30% of the available margin, so framing elements
        # near the edges of a scene (borders, distant buildings) never
        # get cropped out.
        offset_x = max_offset_x * (0.5 + (corner_x - 0.5) * 0.6 * progress)
        offset_y = max_offset_y * (0.72 + (corner_y - 0.5) * 0.4 * progress)  # biased down: content sits higher, clear of the captions
        crop_box = (offset_x, offset_y, offset_x + VIDEO_WIDTH, offset_y + VIDEO_HEIGHT)
        img = img.crop(crop_box)

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
        f"BorderStyle=1,Outline=2,Alignment=2,MarginV=45',"
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

    title, era, setting, beats = generate_script(topic)
    forced = os.environ.get("HISTORY_SETTING", "").strip().lower()
    CAST["setting"] = forced if forced in ("europe", "global") else setting
    CAST["era"] = era
    CAST["seed"] = int(hashlib.md5(title.encode("utf-8")).hexdigest()[:8], 16)
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
        speech_text = spell_out_years(full_text)
        synthesize_narration(speech_text, voice_dir, narration_path)
        print("Narration synthesized")

        whisper_words = transcribe_words(narration_path)
        duration = whisper_words[-1][1] if whisper_words else 0.0
        print(f"Transcribed {len(whisper_words)} words, duration={duration:.2f}s")

        # Whisper's transcription is used only for timing; the displayed
        # caption text is our own script (with years already spelled out),
        # so captions can't carry a mis-transcription.
        caption_words = align_text_to_timestamps(speech_text, whisper_words)
        build_captions_srt(caption_words, captions_path)
        beats = assign_beat_times(beats, whisper_words)

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
