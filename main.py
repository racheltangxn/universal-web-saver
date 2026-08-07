"""
Universal Web Saver — webhook backend for the "Save to Notion" Android Share Sheet.

Pipeline:
  1. Android (HTTP Shortcuts) POSTs a URL to /webhook with an X-API-Key header.
  2. We validate the key, then run the pipeline synchronously (offloaded to a
     threadpool so the event loop isn't blocked) and return the real result.
  3. The pipeline routes the URL by domain (Instagram/TikTok/LinkedIn via
     Apify, YouTube via transcript API, everything else via Jina Reader /
     trafilatura), sends the extracted content to Gemini for structured tagging,
     and pushes the result into a Notion database page.

Synchronous by design: on Cloud Run (scale-to-zero), CPU isn't guaranteed once
a response has been sent, so a "return 200 immediately, finish in the
background" pattern can get silently killed mid-pipeline. Waiting for the real
result (~10-30s) trades instant-ack UX for actually knowing whether the save
worked.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import hmac
import logging
import os
import tempfile
import time
from datetime import date, datetime
from typing import Any, Optional
from urllib.parse import urlparse

import requests
import trafilatura
from apify_client import ApifyClient
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from google import genai
from google.genai import types as genai_types
from notion_client import Client as NotionClient

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("universal-web-saver")

SECRET_WEBHOOK_KEY = os.environ["SECRET_WEBHOOK_KEY"]
APIFY_API_TOKEN = os.environ["APIFY_API_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
NOTION_API_KEY = os.environ["NOTION_API_KEY"]
NOTION_DATABASE_ID = os.environ["NOTION_DATABASE_ID"]

# Apify actor IDs — override in .env if you use different actors.
APIFY_INSTAGRAM_ACTOR = os.environ.get("APIFY_INSTAGRAM_ACTOR", "apify/instagram-scraper")
APIFY_TIKTOK_ACTOR = os.environ.get("APIFY_TIKTOK_ACTOR", "clockworks/tiktok-scraper")
APIFY_LINKEDIN_ACTOR = os.environ.get("APIFY_LINKEDIN_ACTOR", "apify/linkedin-post-scraper")
# youtube-transcript-api's direct calls get blocked from cloud-provider IPs
# (Cloud Run included), so YouTube is routed through Apify like the other
# platforms instead. starvibe/youtube-video-transcript picked for its 99.99%
# success rate across ~490k runs at evaluation time (2026-07-28).
APIFY_YOUTUBE_ACTOR = os.environ.get("APIFY_YOUTUBE_ACTOR", "starvibe/youtube-video-transcript")

JINA_READER_BASE = "https://r.jina.ai/"
# "gemini-1.5-flash" was the original spec's model; that line (and even 2.5-flash)
# is now retired for new API keys. "gemini-flash-latest" is Google's rolling
# alias to the current-gen Flash model, so this default won't rot again.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
NOTION_BLOCK_CHAR_LIMIT = 2000

apify_client = ApifyClient(APIFY_API_TOKEN)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
notion = NotionClient(auth=NOTION_API_KEY)

app = FastAPI(title="Universal Web Saver")


# --------------------------------------------------------------------------- #
# Step 1: The Universal Router
# --------------------------------------------------------------------------- #

def classify_url(url: str) -> str:
    """Return one of: instagram, tiktok, linkedin, youtube, general."""
    host = urlparse(url).netloc.lower()

    if "instagram.com" in host:
        return "instagram"
    if "tiktok.com" in host:
        return "tiktok"
    if "linkedin.com" in host:
        return "linkedin"
    if "youtube.com" in host or "youtu.be" in host:
        return "youtube"
    return "general"


def extract_youtube(url: str) -> dict[str, Any]:
    """Fetch transcript + title/channel for a YouTube video via Apify (direct
    youtube-transcript-api calls get blocked from cloud-provider IPs), and
    also hand the URL to Gemini directly as a video (video_uri) — Gemini
    fetches YouTube URLs natively, so it can watch/listen to the actual video
    (on-screen text, visuals) instead of relying only on the caption track."""
    log.info("YouTube: running Apify actor %s for %s", APIFY_YOUTUBE_ACTOR, url)
    run = apify_client.actor(APIFY_YOUTUBE_ACTOR).call(
        run_input={"youtube_url": url, "include_transcript_text": True}
    )
    if run is None:
        raise ValueError(f"Apify actor {APIFY_YOUTUBE_ACTOR} run failed to start for {url}")

    items = list(apify_client.dataset(run.default_dataset_id).iterate_items())
    if not items:
        raise ValueError(f"Apify actor {APIFY_YOUTUBE_ACTOR} returned no items for {url}")

    item = items[0]
    if item.get("status") != "success":
        log.warning("YouTube: actor reported non-success status for %s (%s)", url, item.get("message"))

    return {
        "text": item.get("transcript_text") or "",
        "title": item.get("title") or url,
        "author_name": item.get("channel_name"),
        "images": [],
        "video_uri": url,
    }


def extract_social(url: str, platform: str) -> dict[str, Any]:
    """Run the relevant Apify actor for Instagram / TikTok / LinkedIn and
    normalize the result into text + handle + media."""
    actor_map = {
        "instagram": APIFY_INSTAGRAM_ACTOR,
        "tiktok": APIFY_TIKTOK_ACTOR,
        "linkedin": APIFY_LINKEDIN_ACTOR,
    }
    actor_id = actor_map[platform]
    log.info("%s: running Apify actor %s for %s", platform, actor_id, url)

    # Input field names are actor-specific, not interchangeable — verified
    # against each actor's real input schema (2026-08-07). clockworks/tiktok-scraper
    # requires `postURLs` (capital URL) specifically; the previous shared
    # `postUrls` key was silently rejected by TikTok's schema validation on
    # every single run, 100% failure. `shouldDownloadVideos`/`shouldDownloadSlideshowImages`
    # make the actor persist a non-expiring media URL (field `mediaUrls`)
    # instead of only a signed CDN link.
    if platform == "tiktok":
        run_input: dict[str, Any] = {
            "postURLs": [url],
            "shouldDownloadVideos": True,
            "shouldDownloadCovers": True,
            "shouldDownloadSlideshowImages": True,
        }
    elif platform == "instagram":
        run_input = {"directUrls": [url]}
    else:
        # LinkedIn actor's exact schema is unverified — keep the previous
        # defensive multi-key guess rather than change untested behavior.
        run_input = {"directUrls": [url], "urls": [url], "postUrls": [url]}

    run = apify_client.actor(actor_id).call(run_input=run_input)
    if run is None:
        raise ValueError(f"Apify actor {actor_id} run failed to start for {url}")

    # Fail loudly on a non-SUCCEEDED run (e.g. actor-level input rejection)
    # instead of silently falling through to an empty/garbage item — this is
    # exactly how every TikTok save failed invisibly before this fix.
    run_status = getattr(run, "status", None)
    if run_status and run_status != "SUCCEEDED":
        status_message = getattr(run, "status_message", None)
        raise ValueError(
            f"Apify actor {actor_id} run did not succeed (status={run_status}): {status_message}"
        )

    # apify-client 3.x returns a typed Run model (snake_case attrs), not a dict.
    items = list(apify_client.dataset(run.default_dataset_id).iterate_items())

    if not items:
        raise ValueError(f"Apify actor {actor_id} returned no items for {url}")

    item = items[0]

    # Field names vary by actor/platform — grab the common variants defensively.
    caption = item.get("caption") or item.get("text") or item.get("description") or ""
    handle = (
        item.get("ownerUsername")
        or item.get("authorMeta", {}).get("name")
        or item.get("author", {}).get("username")
        or item.get("authorName")
    )
    # Video posts (Reels/TikTok/LinkedIn video) get a `videoUrl` on most actors;
    # clockworks/tiktok-scraper instead puts its (non-expiring, since we asked
    # it to download) media URL in `mediaUrls`. Keep this separate from still
    # images so it's uploaded to Gemini as an actual video (audio + motion +
    # on-screen text) instead of being treated as a photo.
    video_url = item.get("videoUrl") or next(iter(item.get("mediaUrls") or []), None)

    images = []
    if item.get("displayUrl"):
        images.append(item["displayUrl"])
    if item.get("images"):
        images.extend(item["images"])
    if not video_url and item.get("coverUrl"):
        # Cover thumbnail is only useful as a fallback when there's no actual
        # video to analyze (e.g. actor failed to return a video URL).
        images.append(item["coverUrl"])

    if not caption and not handle and not images and not video_url:
        # A "successful" run can still return a content-free item. Treat that
        # as a failure instead of silently creating an empty "Platform post by
        # unknown" Notion page, which is what happened for every TikTok save
        # before this fix.
        raise ValueError(f"{platform} actor returned an empty/unusable item for {url}")

    title = caption[:120] if caption else f"{platform.title()} post by {handle or 'unknown'}"

    return {
        "text": caption,
        "title": title,
        "author_name": handle,
        "images": [img for img in images if img],
        "video_url": video_url,
    }


def extract_general(url: str) -> dict[str, Any]:
    """Extract main article content as clean text via Jina Reader, falling
    back to trafilatura if Jina is unavailable."""
    log.info("General web: extracting %s via Jina Reader", url)
    try:
        resp = requests.get(
            f"{JINA_READER_BASE}{url}",
            headers={"Accept": "text/plain"},
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.text
    except requests.RequestException as e:
        log.warning("Jina Reader failed (%s), falling back to trafilatura", e)
        downloaded = trafilatura.fetch_url(url)
        content = trafilatura.extract(downloaded) if downloaded else None
        if not content:
            raise ValueError(f"Could not extract content from {url}")

    # Best-effort title: first markdown heading or non-empty line.
    title = url
    for line in content.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            title = stripped[:200]
            break

    return {"text": content, "title": title, "author_name": None, "images": []}


def route_and_extract(url: str) -> tuple[str, dict[str, Any]]:
    """Returns (platform, extracted_data)."""
    platform = classify_url(url)
    log.info("Routed %s -> %s", url, platform)

    if platform == "youtube":
        return platform, extract_youtube(url)
    if platform in ("instagram", "tiktok", "linkedin"):
        return platform, extract_social(url, platform)
    return platform, extract_general(url)


# --------------------------------------------------------------------------- #
# Step 2: Gemini — the tagging engine
# --------------------------------------------------------------------------- #

# Must match the "Category" select options in the Notion database exactly.
CATEGORY_OPTIONS = [
    "Article", "Recipe", "YouTube Video", "Social Media", "Tool",
    "News", "Business", "Career", "Creative", "Wellness", "Other",
]

TAGGING_PROMPT = """You are a universal tagging engine for a personal "MyMind"-style
knowledge base. You will be given the extracted text, and possibly images or a video,
from a saved web link. Analyze it and return STRICT JSON matching this schema:

{{
  "transcription": "The full verbatim content. For a video: the spoken-word audio transcript AND any on-screen/overlay/caption text, each clearly labeled (e.g. 'Spoken:' / 'On-screen text:'). For images: OCR'd text. For articles: the full body text.",
  "ai_summary": "A 2-sentence summary of the link's core value or main thesis",
  "category": "Exactly one of: {category_options}",
  "tags": ["Tag1", "Tag2", "Tag3"],
  "vibe": "Short description of tone/aesthetic (e.g., 'Educational', 'Minimalist', 'Opinionated')",
  "author_name": "Name of the creator, author, or website publisher"
}}

Source URL: {url}
Known author/handle (may be null, use your own judgement if content suggests otherwise): {author_name}

Extracted content:
---
{content}
---
"""


def normalize_category(raw_category: str) -> str:
    """Coerce Gemini's category guess into one of the Notion select's fixed
    options (case-insensitive match), defaulting to 'Other'."""
    lookup = {opt.lower(): opt for opt in CATEGORY_OPTIONS}
    return lookup.get((raw_category or "").strip().lower(), "Other")


GEMINI_FILE_POLL_INTERVAL_S = 2
GEMINI_FILE_POLL_TIMEOUT_S = 90
# Carousels can run to 20+ slides; a single combined Gemini request can
# silently skip slides even when all images are attached, so each slide is
# OCR'd in its own call instead and the results aggregated before tagging.
GEMINI_MAX_CAROUSEL_SLIDES = 20


def build_video_part(extracted: dict[str, Any]) -> tuple[Optional[Any], Optional[str]]:
    """Return (Part, gemini_file_name_to_clean_up_or_None) for a video, so Gemini
    can watch/listen to it (audio transcript + on-screen text) instead of just
    getting a caption/cover-thumbnail proxy for it.

    - YouTube: Gemini fetches youtube.com/watch?v=... URLs natively, no download needed.
    - IG/TikTok/LinkedIn: the actor's `videoUrl` is a plain CDN link Gemini can't
      fetch itself, so we download it and upload it through the Gemini File API.
    """
    youtube_uri = extracted.get("video_uri")
    if youtube_uri:
        return genai_types.Part.from_uri(file_uri=youtube_uri, mime_type="video/*"), None

    video_url = extracted.get("video_url")
    if not video_url:
        return None, None

    tmp_path = None
    try:
        log.info("Gemini: downloading video for upload: %s", video_url)
        resp = requests.get(video_url, timeout=60, stream=True)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
            tmp_path = f.name

        file_obj = gemini_client.files.upload(file=tmp_path)
        waited = 0
        while file_obj.state.name == "PROCESSING" and waited < GEMINI_FILE_POLL_TIMEOUT_S:
            time.sleep(GEMINI_FILE_POLL_INTERVAL_S)
            waited += GEMINI_FILE_POLL_INTERVAL_S
            file_obj = gemini_client.files.get(name=file_obj.name)

        if file_obj.state.name != "ACTIVE":
            log.warning("Gemini: video upload never became ACTIVE (state=%s) for %s", file_obj.state, video_url)
            return None, None

        return genai_types.Part.from_uri(file_uri=file_obj.uri, mime_type=file_obj.mime_type), file_obj.name
    except (requests.RequestException, OSError) as e:
        log.warning("Gemini: could not download/upload video %s (%s)", video_url, e)
        return None, None
    finally:
        if tmp_path:
            os.unlink(tmp_path)


def ocr_image_slide(image_url: str, index: int) -> Optional[str]:
    """Download one carousel slide and OCR just that image in its own Gemini
    call, labeled by position. Returns None if the slide can't be downloaded,
    isn't an image, or Gemini returns nothing — callers should skip it rather
    than fail the whole carousel over one bad slide."""
    try:
        img_resp = requests.get(image_url, timeout=15)
        img_resp.raise_for_status()
        mime_type = img_resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
        if not mime_type.startswith("image/"):
            log.warning("Gemini: slide %d has non-image content-type %s, skipping", index, mime_type)
            return None
    except requests.RequestException as e:
        log.warning("Gemini: could not download slide %d (%s): %s", index, image_url, e)
        return None

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                "OCR every piece of visible text in this image, verbatim, in reading order. "
                "If there is no text, briefly describe what's shown instead.",
                genai_types.Part.from_bytes(data=img_resp.content, mime_type=mime_type),
            ],
        )
        text = (response.text or "").strip()
        return f"Slide {index}: {text}" if text else None
    except Exception as e:
        log.warning("Gemini: OCR failed for slide %d (%s): %s", index, image_url, e)
        return None


def call_gemini(url: str, extracted: dict[str, Any]) -> dict[str, Any]:
    """Send extracted text (and optional images/video) to Gemini and get back
    structured JSON tags."""
    video_part, gemini_file_to_cleanup = build_video_part(extracted)
    images = (extracted.get("images") or [])[:GEMINI_MAX_CAROUSEL_SLIDES]

    content = extracted.get("text") or ""
    slide_texts: list[str] = []
    if not video_part and len(images) > 1:
        # Multi-slide carousel: OCR each slide separately and aggregate into
        # the text content, rather than attaching every image to one combined
        # request (see GEMINI_MAX_CAROUSEL_SLIDES comment above).
        log.info("Gemini: OCR'ing %d carousel slide(s) individually", len(images))
        for i, image_url in enumerate(images, start=1):
            slide_text = ocr_image_slide(image_url, i)
            if slide_text:
                slide_texts.append(slide_text)
        if slide_texts:
            content += "\n\n" + "\n\n".join(slide_texts)

    prompt = TAGGING_PROMPT.format(
        url=url,
        author_name=extracted.get("author_name"),
        category_options=", ".join(CATEGORY_OPTIONS),
        content=content[:50000],  # keep prompt bounded
    )

    parts: list[Any] = [prompt]

    if video_part:
        parts.append(video_part)
    elif len(images) == 1:
        # Single image (plain photo post, or a cover-thumbnail fallback) —
        # cheap enough to send inline with the main tagging call.
        try:
            img_resp = requests.get(images[0], timeout=15)
            img_resp.raise_for_status()
            mime_type = img_resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
            if mime_type.startswith("image/"):
                parts.append(
                    genai_types.Part.from_bytes(data=img_resp.content, mime_type=mime_type)
                )
        except requests.RequestException as e:
            log.warning("Gemini: could not download media %s (%s)", images[0], e)

    log.info(
        "Gemini: sending %d part(s) for analysis (%d carousel slide(s) pre-OCR'd)",
        len(parts),
        len(slide_texts),
    )
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=parts,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
    finally:
        if gemini_file_to_cleanup:
            try:
                gemini_client.files.delete(name=gemini_file_to_cleanup)
            except Exception as e:
                log.warning("Gemini: could not clean up uploaded file %s (%s)", gemini_file_to_cleanup, e)

    import json

    result = json.loads(response.text)

    # Guarantee required keys exist even if the model omits one.
    result.setdefault("transcription", extracted.get("text", ""))
    result.setdefault("ai_summary", "")
    result.setdefault("category", "Other")
    result.setdefault("tags", [])
    result.setdefault("vibe", "")
    result.setdefault("author_name", extracted.get("author_name"))

    # The Notion "Category" property is a fixed-option select — coerce
    # whatever Gemini returned onto one of the allowed options.
    result["category"] = normalize_category(result["category"])

    return result


# --------------------------------------------------------------------------- #
# Step 3: Notion push
# --------------------------------------------------------------------------- #

def chunk_text(text: str, size: int = NOTION_BLOCK_CHAR_LIMIT) -> list[str]:
    """Split text into <= `size`-character chunks for Notion's block limit."""
    if not text:
        return []
    return [text[i : i + size] for i in range(0, len(text), size)]


def build_transcription_blocks(transcription: str) -> list[dict[str, Any]]:
    blocks = []
    for chunk in chunk_text(transcription):
        blocks.append(
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": chunk}}]},
            }
        )
    return blocks


# Maps our internal platform classification to the Notion "Source Type" select options.
SOURCE_TYPE_MAP = {
    "instagram": "Instagram",
    "tiktok": "TikTok",
    "linkedin": "LinkedIn",
    "youtube": "YouTube",
    "general": "Web Article",
}


_data_source_id: Optional[str] = None


def get_data_source_id() -> str:
    """notion-client 3.x split database queries into a separate 'data source'
    concept (Notion API 2025-09-03+): databases.query() no longer exists,
    queries go through data_sources.query() against the data source id, not
    the database id. Resolved once and cached."""
    global _data_source_id
    if _data_source_id is None:
        db = notion.databases.retrieve(database_id=NOTION_DATABASE_ID)
        data_sources = db.get("data_sources") or []
        if not data_sources:
            raise ValueError(f"Notion database {NOTION_DATABASE_ID} has no data sources")
        _data_source_id = data_sources[0]["id"]
    return _data_source_id


def find_existing_page(url: str) -> Optional[str]:
    """Return the page ID already saved for this URL, if any. Guards against
    duplicate pages when a client (e.g. a timed-out HTTP Shortcuts request)
    retries the same webhook call."""
    results = notion.data_sources.query(
        data_source_id=get_data_source_id(),
        filter={"property": "External ID", "rich_text": {"equals": url[:2000]}},
        page_size=1,
    )
    pages = results.get("results") or []
    return pages[0]["id"] if pages else None


def push_to_notion(url: str, platform: str, title: str, ai_data: dict[str, Any]) -> str:
    """Create a Notion page with mapped properties + transcription page content.
    Returns the new (or existing, if already saved) page ID."""
    existing_id = find_existing_page(url)
    if existing_id:
        log.info("Notion: %s already saved as page %s, skipping duplicate create", url, existing_id)
        return existing_id

    properties = {
        "Title": {"title": [{"text": {"content": title[:2000] or url}}]},
        "URL": {"url": url},
        "Category": {"select": {"name": ai_data["category"]}},
        "Tags": {"multi_select": [{"name": tag} for tag in (ai_data.get("tags") or [])]},
        "AI Summary": {
            "rich_text": [{"type": "text", "text": {"content": (ai_data.get("ai_summary") or "")[:2000]}}]
        },
        "Vibe/Aesthetic": {
            "rich_text": [{"type": "text", "text": {"content": (ai_data.get("vibe") or "")[:2000]}}]
        },
        "Account Name": {
            "rich_text": [{"type": "text", "text": {"content": (ai_data.get("author_name") or "")[:2000]}}]
        },
        "Date Posted": {"date": {"start": date.today().isoformat()}},
        # Shared "Universal Content Inbox" database also ingests Audio Hub
        # entries — these two fields are what distinguish/filter our rows.
        "Capture Tool": {"select": {"name": "Universal Web Saver"}},
        "Source Type": {"select": {"name": SOURCE_TYPE_MAP.get(platform, "Other")}},
        "Status": {"select": {"name": "New"}},
        "External ID": {"rich_text": [{"type": "text", "text": {"content": url[:2000]}}]},
    }

    log.info("Notion: creating page '%s'", title[:80])
    page = notion.pages.create(
        parent={"database_id": NOTION_DATABASE_ID},
        properties=properties,
    )

    transcription = ai_data.get("transcription") or ""
    blocks = build_transcription_blocks(transcription)

    # Notion's append endpoint accepts at most 100 blocks per call.
    for i in range(0, len(blocks), 100):
        notion.blocks.children.append(block_id=page["id"], children=blocks[i : i + 100])

    log.info("Notion: page created (%s), %d transcription block(s) appended", page["id"], len(blocks))
    return page["id"]


# --------------------------------------------------------------------------- #
# The async pipeline
# --------------------------------------------------------------------------- #

def process_url(url: str) -> dict[str, Any]:
    """Full extraction -> Gemini -> Notion pipeline. Each stage's exception is
    caught and logged, then turned into a real {"ok": False, "error": ...}
    result instead of being swallowed, so the caller can report what actually
    happened."""
    log.info("=== Processing %s ===", url)
    try:
        platform, extracted = route_and_extract(url)
    except Exception as e:
        log.exception("Extraction failed for %s", url)
        return {"ok": False, "error": f"Extraction failed: {e}"}

    try:
        ai_data = call_gemini(url, extracted)
    except Exception as e:
        log.exception("Gemini tagging failed for %s", url)
        return {"ok": False, "error": f"Gemini tagging failed: {e}"}

    try:
        page_id = push_to_notion(url, platform, extracted.get("title") or url, ai_data)
    except Exception as e:
        log.exception("Notion push failed for %s", url)
        return {"ok": False, "error": f"Notion push failed: {e}"}

    log.info("=== Done: %s ===", url)
    return {"ok": True, "title": extracted.get("title") or url, "notion_page_id": page_id}


# --------------------------------------------------------------------------- #
# FastAPI routes
# --------------------------------------------------------------------------- #

@app.post("/webhook")
async def webhook(
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
):
    if not x_api_key or not hmac.compare_digest(x_api_key, SECRET_WEBHOOK_KEY):
        log.warning("Rejected request with invalid/missing X-API-Key")
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Body must be valid JSON")

    url = payload.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url' field in payload")

    log.info("Accepted webhook for %s", url)
    # process_url does blocking I/O (requests/apify_client/notion_client are all
    # sync); offload to a threadpool so the event loop can still serve /health etc.
    result = await run_in_threadpool(process_url, url)
    if not result["ok"]:
        raise HTTPException(status_code=502, detail=result["error"])
    return {"status": "saved", "title": result["title"], "notion_page_id": result["notion_page_id"]}


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}
