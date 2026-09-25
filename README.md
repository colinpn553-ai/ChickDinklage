# ig-reels-bot

Scheduled Reels publisher for an Instagram Business/Creator account, using
Meta's official Instagram Graph API. Runs as a GitHub Actions workflow on a
cron schedule; you queue up clips by committing them to the repo, and each
scheduled run posts the oldest one still waiting.

**Scope note:** this only publishes clips you already have the rights to
post — your own footage, clips you have explicit permission to use, or
(for the history pipeline below) entirely original AI-written scripts,
locally-synthesized narration, and hand-coded procedural animation. It
does not download or scrape video from creators who haven't licensed it
for reuse.

## Two ways clips get into the queue

1. **Manual**: drop a clip into `queue/pending/` yourself (see "Queueing a
   clip" below).
2. **Automated, with mandatory human review**: the `generate-history-clip`
   workflow writes a short factual history explainer, narrates it, and
   animates it — see "History clip pipeline" below. Because this one
   covers real people and real events, it never posts automatically: output
   lands in `queue/review_pending/`, and nothing moves to `queue/pending/`
   (where the posting bot looks) until you explicitly approve it.

Either way, the posting side works the same:

## How it works

1. Drop a clip into `queue/pending/` (see "Queueing a clip" below), along
   with an optional `.json` caption file.
2. On its schedule, the `post-reel` workflow checks out the repo, runs
   `scripts/post_to_instagram.py`, which:
   - picks the oldest file in `queue/pending/`
   - builds a public URL for it via `raw.githubusercontent.com`
   - calls the Instagram Graph API to create a Reels media container from
     that URL, waits for Instagram to finish processing it, then publishes it
   - moves the clip into `queue/posted/` and appends a row to
     `queue/posted/log.csv`
   - commits and pushes that queue change back to the repo

## Important tradeoff: the repo must be public

Instagram's API fetches the video by URL and has no way to authenticate to
a private GitHub repo, so `raw.githubusercontent.com` links only work if
this repository is public. That means every clip you queue is publicly
visible in the repo (and its git history) before and after it's posted —
functionally no more private than the Instagram post itself, but worth
knowing going in.

If you don't want a public repo, swap `raw_url()` in
`scripts/post_to_instagram.py` for a call that uploads the clip to your own
storage (S3, Cloudflare R2, etc.) and returns a signed/public URL instead.
The rest of the script is unaffected.

## One-time setup (you do this manually in the Meta UI)

This part can't be scripted — Meta requires you to create the app and
generate the token yourself, logged in as the account owner. This uses
Meta's newer **Instagram API with Instagram Login** flow, which doesn't
require a linked Facebook Page.

1. **Convert your Instagram account** to a Professional (Business or
   Creator) account: Settings → Account type and tools.
2. **Create a Meta developer app** at https://developers.facebook.com/apps
   → "Create App" → type "Business".
3. Under **Use cases**, add **"Manage messaging & content on Instagram"**
   and click Customize.
4. On the "API setup with Instagram login" page:
   - Under **App roles → Roles**, add your Instagram account as an
     **Instagram Tester**, then accept the invite from inside the
     Instagram app (Settings → Apps and websites → Tester invites).
   - Under **"1. Add required messaging permissions"**, also add
     `instagram_content_publish` via "Go to permissions and features" (the
     default list only includes basic/comments/messages permissions).
   - Under **"2. Generate access tokens"**, click **Generate token** next
     to your Instagram account. Approve the permission prompt. The token
     shown there is usable directly (already long-lived, ~60 days) — no
     manual token exchange needed.
   - Note the **Instagram User ID** shown next to your account name (a
     large numeric ID) — that's your `IG_BUSINESS_ACCOUNT_ID`. It is
     *not* the same as the Business Portfolio asset ID shown elsewhere in
     Meta Business Suite; use the ID from this page.
5. In this GitHub repo, go to **Settings → Secrets and variables → Actions**
   and add:
   - `IG_ACCESS_TOKEN` — the token from step 4
   - `IG_BUSINESS_ACCOUNT_ID` — the Instagram User ID from step 4

Tokens expire (~60 days). Regenerate via the same "Generate token" button
and update the `IG_ACCESS_TOKEN` secret before it expires, or the
scheduled run will fail with an auth error.

Note: this flow calls `graph.instagram.com`, not `graph.facebook.com` —
that's already reflected in `scripts/post_to_instagram.py`.

## History clip pipeline

One API key, self-service:

1. **Anthropic API key**: https://console.anthropic.com → API Keys →
   Create Key (paid-as-you-go, separate from any Claude subscription)
2. Add it as a repo secret (**Settings → Secrets and variables →
   Actions**): `ANTHROPIC_API_KEY`

The `generate-history-clip` workflow then runs daily at 12:00 UTC (or
trigger manually from the Actions tab, optionally passing a specific
`topic` input instead of a random pick from `HISTORY_TOPICS` in
`scripts/generate_history_clip.py`). For each run it:

1. Asks Claude for a short, neutral, documentary-style script (130-190
   words) split into beats, each tagged with one scene from a fixed
   vocabulary (jungle, building, crowd, soldiers, map, meeting, leader,
   fire, prison, mosque, church, exodus) — see the prompt in
   `HISTORY_PROMPT` for the full tone/accuracy instructions.
2. Synthesizes the narration locally with **Piper** (free, offline neural
   TTS — no account, no per-use cost). The voice model downloads fresh
   each run from Hugging Face (~63MB); change `PIPER_VOICE_URL_BASE` to
   use a different Piper voice.
3. Transcribes that same audio locally with **faster-whisper** to recover
   word-level timestamps — used for burned-in captions and for timing
   which scene illustration is on screen, so the visuals track what's
   actually being said rather than looping one fixed animation.
4. Renders animated scenes per beat with Pillow: procedural sky, hills,
   buildings and trees, populated with illustrated character sprites from
   `assets/characters/base/` (21 transparent PNGs made in Canva, cast by
   role and by era — the script's `era` field picks early-1900s clothing
   for older topics; faces vary per clip via a title-derived seed). The
   sprites are single standing poses, so they bob and hop rather than
   walk. If the assets folder is missing, the older drawn stick figures
   are used instead. See the `draw_*` functions and `SCENES` dict. Then
   `ffmpeg` muxes frames + narration + burned SRT captions + a title card
   and outro card into the final 1080x1920 Reel.
   **Maps and flags** come from data, not from a model drawing them. Map
   beats carry ISO country codes and are drawn from Natural Earth
   outlines (`assets/geo/`, public domain) with a slow zoom, highlights,
   labels, and a "Present-day borders" tag; building/meeting/leader beats
   may show a national flag from `assets/flags/` (271 PNGs rendered from
   the MIT-licensed `flag-icons` project, license kept alongside). Codes
   the model invents are dropped at parse time. Because borders and flags
   are present-day, the prompt tells the model to leave `flag` unset when
   the period's state used a different flag (e.g. Khmer Rouge-era
   Cambodia), and everything still goes through the review step.
   **Asset sets** replace the generic cast and drawn scenery when a
   setting has its own pack. `assets/sets/<name>/` holds `cast.json`
   (which sprites fill which role, and which backdrop each scene uses),
   `characters/` (transparent sprites cut from generated character sheets
   with `scripts/extract_sprites.py`) and `backdrops/` (painted scenes
   fitted so their ground line sits on the horizon the sprites stand on,
   via `scripts/prepare_backdrop.py`). `SET_FOR` in the generator maps
   (setting, era) to a set; `europe_1920s` is used for early-1900s
   European topics, and `HISTORY_SET` overrides the choice. Sets were
   generated with Adobe Firefly. To add one, generate backdrops and
   character sheets on a plain white background, run the two scripts, and
   write a `cast.json`. Scenes without a backdrop, and the map scene,
   still use the drawn versions.
5. Writes the result to `queue/review_pending/`, **not**
   `queue/pending/`.

### Approving a clip

Review the file in `queue/review_pending/` (download it, watch/listen to
it). If it's good to post, go to the Actions tab → **"Approve History
Clip"** → Run workflow → enter the base filename (no extension, e.g.
`001_the_rise_of_a_revolution`). That moves it into `queue/pending/`,
where the normal posting workflow will pick it up. If it's not good,
just delete the two files from `queue/review_pending/` instead.

### Why this one isn't automatic

The fictional-content version of this pipeline was removed; this one
covers real history, including topics involving real atrocities (e.g.
the Khmer Rouge, the Bosnian genocide). An LLM can get facts wrong or
strike the wrong tone, and there's no acceptable failure mode where that
goes out publicly, unreviewed, under a real account. The review step is
a few seconds of your time per clip in exchange for a human actually
looking at claims about real events before they're published.

## Queueing a clip

```
queue/pending/001_my_clip.mp4
queue/pending/001_my_clip.json   (optional)
```

`001_my_clip.json`:
```json
{ "caption": "Caption text and #hashtags go here" }
```

Prefix filenames (`001_`, `002_`, ...) to control posting order — the
script always posts the alphabetically-first `.mp4` in `queue/pending/`.

Commit and push both files; the next scheduled run (or a manual run via
the Actions tab → "Post Instagram Reel" → "Run workflow") will post it.

## Rate limits

Instagram's Content Publishing API caps you at 25 posts per rolling 24-hour
window per IG account. Don't schedule the cron more often than that.

## Local testing

You can run the script locally, but `raw_url()` depends on
`GITHUB_REPOSITORY` / `GITHUB_REF_NAME`, which GitHub Actions sets
automatically. To test locally, export those yourself and make sure the
clip is already pushed to a public branch so the raw URL resolves:

```bash
export IG_ACCESS_TOKEN=...
export IG_BUSINESS_ACCOUNT_ID=...
export GITHUB_REPOSITORY=yourname/ig-reels-bot
export GITHUB_REF_NAME=main
pip install -r scripts/requirements.txt
python scripts/post_to_instagram.py
```
