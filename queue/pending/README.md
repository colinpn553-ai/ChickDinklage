# ig-reels-bot

Scheduled Reels publisher for an Instagram Business/Creator account, using
Meta's official Instagram Graph API. Runs as a GitHub Actions workflow on a
cron schedule; you queue up clips by committing them to the repo, and each
scheduled run posts the oldest one still waiting.

**Scope note:** this only publishes clips you already have the rights to
post (your own footage, or clips you have explicit permission to use). It
does not download or scrape video from anywhere.

## How it works

1. You drop an `.mp4` file (and an optional `.json` caption file) into
   `queue/pending/` and push it to GitHub.
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
