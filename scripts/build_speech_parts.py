#!/usr/bin/env python3
"""Writes assets/scripts/mussolini_1941_part{N}.json for the remaining parts of
the Feb. 23, 1941 speech (English translation as published at the time).
Each paragraph is one beat; photos are public-domain, dated, and labelled.
Audio for part N lives in assets/audio/mussolini_1941_part{N}.wav."""
import json
from pathlib import Path

P = "mussolini_1941/"
PH = {  # key -> (file, on-screen label)
 "h41": ("hitler_mussolini_1941.jpg", "Hitler and Mussolini, Aug. 1941"),
 "h34": ("hitler_mussolini_1934.jpg", "Hitler and Mussolini, 1934"),
 "war40": ("piazza_venezia_1940.jpg", "Rome crowd, 1940 (other occasion)"),
 "pod32": ("podium_1932.jpg", "Mussolini, 1932 (other occasion)"),
 "crowd26": ("crowd_1926.jpg", "Rally crowd, 1926 (other occasion)"),
 "off28": ("officials_1928.jpg", "Mussolini with officials, 1928"),
 "cham41": ("chamber_speech_1941-06-10.jpg", "Mussolini in the Chamber, June 1941"),
 "chov41": ("chamber_overview_1941-06-10.jpg", "The Chamber, June 1941"),
 "franco": ("bordighera_franco_1941-02-12.jpg", "With Franco, Bordighera, Feb. 1941"),
 "cavour": ("cavour_after_taranto_1940.jpg", "Cavour after Taranto raid, 1940"),
 "bardia": ("bardia_prisoners_1941-01.jpg", "Prisoners taken at Bardia, 1941"),
 "beda": ("beda_fomm_tanks_1941-02.jpg", "Captured tanks, Beda Fomm, 1941"),
 "genoa": ("genoa_cathedral_damage_1941-02.jpg", "Genoa cathedral damage, Feb. 1941"),
 "churchill": ("churchill_liverpool_1941-04.jpg", "Churchill, Liverpool, April 1941"),
 "sidi": ("sidi_barrani_prisoners_1940-12.jpg", "Prisoners, Sidi Barrani, Dec. 1940"),
 "tanks": ("tanks_libya_1941.jpg", "Italian tanks, Libya, c. 1941"),
}
EXTRA = {  # per-part caption note
 13: " WARNING: the first paragraph repeats an antisemitic conspiracy claim (that the United States is 'dominated by Jews'). It is Mussolini's own propaganda, shown as historical record and not endorsed.",
}
exec(open("scripts/speech_text.py", encoding="utf-8").read())  # defines PARTS = {n: [(text, photo_key), ...]}
for n, beats in PARTS.items():
    sd = {
        "title": f"Mussolini, Feb. 23, 1941: Part {n}",
        "topic": f"Mussolini's February 23, 1941 speech to the Blackshirts of Rome, Part {n} (English translation, radio-style, with dated archival photographs)",
        "era": "early_1900s", "setting": "europe",
        "narration_audio": f"assets/audio/mussolini_1941_part{n}.wav",
        "audio_fx": "radio", "music": "rotate", "music_volume": 0.1,
        "caption_extra": (f"Part {n} of Mussolini's February 23, 1941 speech to the Blackshirts of Rome, as translated into English at the time, "
                          "continuing exactly where the previous part ended, nothing skipped. His claims are historical record, not endorsed. "
                          "Read by a synthetic voice with a vintage-radio effect, not Mussolini's. No freely licensed photos of this exact speech "
                          "were found; these public-domain photos are from other occasions or show the events he discusses, and are dated on screen."
                          + EXTRA.get(n, "")),
        "beats": [{"narration": t, "photo": P + PH[k][0],
                   "label": ("Antisemitic claim, not endorsed" if (n == 13 and i == 0) else PH[k][1])}
                  for i, (t, k) in enumerate(beats)],
    }
    Path(f"assets/scripts/mussolini_1941_part{n}.json").write_text(json.dumps(sd, ensure_ascii=False, indent=2), encoding="utf-8")
    print(n, sum(len(t.split()) for t, _ in beats), "words")
