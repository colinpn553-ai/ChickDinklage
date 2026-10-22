#!/usr/bin/env python3
"""Generate an original, copyright-free "intense beat" (phonk / drill / dark
trap style) as an mp3, using only numpy + ffmpeg -- no samples, no third-party
music. Each style/seed/key gives a different track.

usage: make_beat.py OUT.mp3 --style phonk|drill|trap|cinematic [--seed N] [--seconds 150]
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

SR = 44100
STYLES = {
    #            bpm, root midi, scale,                     snare steps (16ths per bar), cowbell?
    "phonk":     dict(bpm=132, root=43, scale=[0, 3, 5, 7, 10], snare=[4, 12], cow=True, hat_roll=0.25),
    "drill":     dict(bpm=142, root=41, scale=[0, 1, 3, 5, 7, 8], snare=[8], cow=False, hat_roll=0.5),
    "trap":      dict(bpm=140, root=38, scale=[0, 2, 3, 7, 8], snare=[8], cow=False, hat_roll=0.35),
    "noir":      dict(bpm=74, root=38, scale=[0, 2, 3, 5, 7, 8, 10], snare=[8], cow=False, hat_roll=0.0),
    "cinematic": dict(bpm=108, root=36, scale=[0, 2, 3, 5, 7, 8, 11], snare=[4, 12], cow=False, hat_roll=0.1),
}


def midi_hz(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def env_exp(n, decay):
    return np.exp(-np.arange(n) / (decay * SR))


def kick(rng):
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.tanh(2.2 * np.sin(ph) * env_exp(n, 0.16)) * 0.95


def snare(rng):
    n = int(0.28 * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    noise = noise - np.convolve(noise, np.ones(8) / 8, mode="same")  # crude high-pass
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30)
    return np.tanh(1.8 * (noise * env_exp(n, 0.07) * 0.9 + body * 0.6)) * 0.8


def hat(rng, open_=False):
    n = int((0.22 if open_ else 0.05) * SR)
    noise = rng.standard_normal(n)
    noise = noise - np.convolve(noise, np.ones(4) / 4, mode="same")
    return noise * env_exp(n, 0.09 if open_ else 0.018) * 0.35


def cowbell():
    n = int(0.30 * SR)
    t = np.arange(n) / SR
    sig = np.sign(np.sin(2 * np.pi * 562 * t)) + np.sign(np.sin(2 * np.pi * 845 * t))
    return np.tanh(sig * 0.5) * np.exp(-t * 16) * 0.4


def bass808(freq, dur, slide_to=None):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = np.full(n, freq) if slide_to is None else freq + (slide_to - freq) * np.clip(t / (dur * 0.45), 0, 1)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.tanh(2.6 * np.sin(ph)) * np.exp(-t * (1.6 / max(dur, 0.2))) * 0.9


def lead_note(freq, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    sig = sum(np.sign(np.sin(2 * np.pi * freq * (1 + d) * t)) for d in (-0.004, 0.0, 0.004)) / 3
    k = max(3, int(SR / (freq * 6)))  # one-pole-ish lowpass for a darker tone
    sig = np.convolve(sig, np.ones(k) / k, mode="same")
    return sig * np.exp(-t * 5.0) * 0.25


def pad(freqs, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    sig = sum(((2 * ((f * t) % 1) - 1) + (2 * ((f * 1.005 * t) % 1) - 1)) for f in freqs) / (2 * len(freqs))
    k = 40
    sig = np.convolve(sig, np.ones(k) / k, mode="same")
    fade = np.minimum(1, np.minimum(t / 0.3, (dur - t) / 0.5))
    return sig * fade * 0.12


def reverb(x, seconds=2.4, wet=0.45, seed=0):
    """Cavernous room: convolve with decaying noise (FFT)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / (seconds * SR / 5))
    ir[: int(0.02 * SR)] *= np.linspace(0, 1, int(0.02 * SR))
    size = 1 << (len(x) + n).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[: len(x)]
    y = y / (np.max(np.abs(y)) + 1e-9) * np.max(np.abs(x))
    return x * (1 - wet) + y * wet


def fuzz_lead(freq, dur):
    """Sustained, distorted, slightly out-of-tune lead (organ/guitar-ish)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.2 * t)
    sig = sum(np.sin(2 * np.pi * freq * m * d * vib * t) / m for m in (1, 2, 3) for d in (0.997, 1.003))
    sig = np.tanh(3.0 * sig) * 0.5
    fade = np.minimum(1, t / 0.04) * np.exp(-t * 0.9)
    return sig * fade * 0.45


def put(buf, start, snd):
    if start >= len(buf):
        return
    end = min(len(buf), start + len(snd))
    buf[start:end] += snd[: end - start]


def make(style: str, seed: int, seconds: float) -> np.ndarray:
    cfg = STYLES[style]
    rng = np.random.default_rng(seed)
    root = cfg["root"] + int(rng.integers(-2, 3))
    scale = cfg["scale"]
    step = 60.0 / cfg["bpm"] / 4  # one 16th note, seconds
    bars = int(seconds / (step * 16)) + 1
    total = int((bars * 16 * step + 2) * SR)
    drums = np.zeros(total)
    snr = np.zeros(total)
    bass = np.zeros(total)
    mel = np.zeros(total)
    pads = np.zeros(total)
    kick_hit, snare_hit = kick(rng), snare(rng)
    cow = cowbell()
    motif = [int(rng.choice(scale)) for _ in range(8)]
    riff = [int(rng.choice(scale[:4])) for _ in range(4)]
    kick_pat = [0, 3, 6, 10] if style != "drill" else [0, 7, 10]
    kick_pat = sorted(set(kick_pat + [int(rng.choice([11, 14]))]))
    chords = [[0, 3, 7], [-2, 2, 5], [-4, 0, 3], [-5, -2, 2]] if style == "cinematic" else [[0, 3, 7], [0, 3, 7], [-2, 2, 5], [-4, 0, 3]]
    for b in range(bars):
        t0 = b * 16 * step
        intro = b < 2
        # drums (skip kick/snare in the 2-bar intro for a "drop" feel)
        if not intro or style == "cinematic":
            for s in kick_pat:
                put(drums, int((t0 + s * step) * SR), kick_hit * (0.9 + 0.1 * rng.random()))
            for s in cfg["snare"]:
                put(drums, int((t0 + s * step) * SR), snare_hit)
        for s in range(16):
            if intro and style != "cinematic":
                continue
            v = 1.0 if s % 2 == 0 else 0.7
            if style == "noir":
                if s % 2:
                    continue  # sparse hats
                v *= 0.55
            if rng.random() < cfg["hat_roll"] and s % 4 == 3:  # roll: 3 fast hits
                for r in range(3):
                    put(drums, int((t0 + s * step + r * step / 3) * SR), hat(rng) * v)
            else:
                put(drums, int((t0 + s * step) * SR), hat(rng, open_=(s == 14)) * v)
            if cfg["cow"] and s in (0, 3, 6, 10, 13) and not intro:
                put(mel, int((t0 + s * step) * SR), cow * (0.9 if s in (0, 6) else 0.6))
        # 808 follows the kick, root-and-fifth movement, occasional slide
        if not intro or style == "cinematic":
            for i, s in enumerate(kick_pat):
                note = root - 12 + (0 if i % 3 != 2 else scale[int(rng.integers(1, len(scale)))])
                nxt = (kick_pat[i + 1] if i + 1 < len(kick_pat) else 16) - s
                slide = midi_hz(note + int(rng.choice([-2, -1, 0, 0, 2]))) if rng.random() < 0.35 else None
                put(bass, int((t0 + s * step) * SR), bass808(midi_hz(note), max(nxt * step, 0.25), slide))
        if style == "noir":
            if not intro:
                put(snr, int((t0 + 8 * step) * SR), snare_hit * 1.1)
            for k, s in enumerate((0, 6, 10, 13)):  # slow, sustained minor motif
                deg = motif[(b * 4 + k) % len(motif)]
                put(mel, int((t0 + s * step) * SR), fuzz_lead(midi_hz(root + 12 + deg), step * (5 if s != 13 else 3)))
            ch = chords[b % len(chords)]
            put(pads, int(t0 * SR), pad([midi_hz(root + 12 + c) for c in ch] + [midi_hz(root + c) for c in ch[:1]], 16 * step + 0.4) * 1.8)
            continue
        # lead arp
        for s in range(0, 16, 2):
            deg = motif[(s // 2 + b) % len(motif)] + (12 if (b // 4) % 2 and s % 8 == 0 else 0)
            put(mel, int((t0 + s * step) * SR), lead_note(midi_hz(root + 12 + deg), step * 3))
        # pad chord per bar
        ch = chords[b % len(chords)]
        put(pads, int(t0 * SR), pad([midi_hz(root + 12 + c) for c in ch], 16 * step + 0.2))
    n = int(seconds * SR)
    drums, bass, mel, pads, snr = drums[:n], bass[:n], mel[:n], pads[:n], snr[:n]
    if style == "noir":
        space = reverb(mel * 0.9 + pads + snr * 0.8, seconds=2.6, wet=0.55, seed=seed)
        drums = drums + snr * 0.8  # dry snare on top of its reverb tail
        mel, pads = space * 0.7, pads * 0.0
    # duck the pad and melody on kicks (poor man's sidechain)
    kick_env = np.zeros(n)
    for b in range(bars):
        for s in kick_pat:
            i = int((b * 16 * step + s * step) * SR)
            if i < n:
                seg = min(int(0.25 * SR), n - i)
                kick_env[i:i + seg] = np.maximum(kick_env[i:i + seg], np.linspace(1, 0, seg))
    duck = 1 - 0.6 * kick_env
    mix = drums * 0.9 + bass * 0.85 + (mel + pads) * duck
    if style == "noir":
        mix = drums * 0.95 + bass * 0.9 + mel * (0.55 + 0.45 * duck)
    mix = np.tanh(1.3 * mix)
    mix = mix / (np.max(np.abs(mix)) + 1e-9) * 0.92
    return mix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--style", choices=list(STYLES), default="phonk")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=150.0)
    a = ap.parse_args()
    audio = make(a.style, a.seed, a.seconds)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "beat.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((audio * 32767).astype("<i2").tobytes())
        a.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-ac", "2", "-b:a", "160k", str(a.out)], check=True)
    print(f"{a.out} ({a.style}, seed {a.seed}, {a.seconds:.0f}s)")


if __name__ == "__main__":
    main()
