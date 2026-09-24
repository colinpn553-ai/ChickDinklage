# ig-reels-bot

Scheduled Reels publisher for an Instagram Business/Creator account, using
Meta's official Instagram Graph API. Runs as a GitHub Actions workflow on a
cron schedule; you queue up clips by committing them to the repo, and each
scheduled run posts the oldest one still waiting.

**Scope note:** this only publishes clips you already have the rights to
post — your own footage, clips you have explicit permission to use, or
(for the automated story pipeline below) entirely original AI-generated
text, narration, and procedurally-generated visuals. It does not download
or scrape video from creators who haven't licensed it for reuse.

## Two ways clips get into the queue

1. **Manual**: drop a clip into `queue/pending/` yourself (see "Queueing a
   clip" below).
2. **Automated**: the `generate-story-clip` workflow
   (`.github/workflows/generate-story-clip.yml`) runs on its own daily
   schedule, generates an original short horror story via the Anthropic
   API, narrates it with free text-to-speech, composites it over a
   procedurally-animated dark background (no stock footage — generated
   entirely with `ffmpeg` color/noise/blend filters) with a fading title
   card at the open and a "follow for more" card at the close, and drops
   the assembled Reel into `queue/pending/` automatically — see "Automated
   story pipeline setup" below.

Either way, the posting side works the same:

## How it works

1. A clip lands in `queue/pending/` (manually or via the generator above),
   along with an optional `.json` caption file.
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

## Automated story pipeline setup

One more API key, self-service (you create it yourself, logged in as you):

1. **Anthropic API key** (for story generation):
   - Go to https://console.anthropic.com → API Keys → Create Key
   - This is a paid-as-you-go API (very cheap per story; a few hundred
     tokens each), separate from any Claude subscription
2. Add it to this repo's secrets (**Settings → Secrets and variables →
   Actions**):
   - `ANTHROPIC_API_KEY`

Once that's set, the `generate-story-clip` workflow runs daily at 12:00
UTC (edit the cron in `.github/workflows/generate-story-clip.yml` to
change cadence), or trigger it manually from the Actions tab the same way
as the posting workflow.

Notes on this pipeline:
- Narration uses `gTTS`, a free library built on Google Translate's
  text-to-speech endpoint. It's unofficial (not a documented public API),
  so it's usually reliable but can occasionally fail or get rate-limited —
  if a run fails here, retrying usually works.
- The story prompt explicitly instructs the model to avoid real people,
  real tragedies, and copyrighted characters/franchises, to keep generated
  stories original and avoid depicting real events.
- The visual background is entirely procedural — two dark color layers
  slowly cross-fading, with film-grain noise and a vignette, all generated
  by `ffmpeg` filters (`blend`, `noise`, `eq`, `vignette`). No stock
  footage or external video assets are used.
- The story title fades in as a text card for the first ~4 seconds, and a
  "FOLLOW FOR MORE" card fades in for the last ~3 seconds, both drawn with
  `ffmpeg`'s `drawtext` filter using the font at `DRAWTEXT_FONT` (defaults
  to DejaVu Sans Bold, installed via the workflow's `apt-get` step).
- Output video is 1080x1920 (vertical, matching Reels).
- Filenames are auto-numbered based on how many clips already exist in
  `queue/pending/` + `queue/posted/`, so they interleave safely with any
  clips you queue manually.

## Queueing a clip manually

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
