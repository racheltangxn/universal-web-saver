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
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from datetime import date, datetime, timedelta
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
CONTACTS_DATA_SOURCE_ID = os.environ.get(
    "CONTACTS_DATA_SOURCE_ID", "1a53f27f-73ec-806f-b581-000b125f159f"
)
FESTIVALS_DATA_SOURCE_ID = os.environ.get(
    "FESTIVALS_DATA_SOURCE_ID", "3e73e3c8-520d-441a-bb6a-113ed6def096"
)
EDITIONS_DATA_SOURCE_ID = os.environ.get(
    "EDITIONS_DATA_SOURCE_ID", "5946e4d5-8a32-439f-b213-cc0e705d11ac"
)

# Apify actor IDs — override in .env if you use different actors.
APIFY_INSTAGRAM_ACTOR = os.environ.get("APIFY_INSTAGRAM_ACTOR", "apify/instagram-scraper")
APIFY_TIKTOK_ACTOR = os.environ.get("APIFY_TIKTOK_ACTOR", "clockworks/tiktok-scraper")
APIFY_LINKEDIN_ACTOR = os.environ.get("APIFY_LINKEDIN_ACTOR", "supreme_coder/linkedin-post")
# youtube-transcript-api's direct calls get blocked from cloud-provider IPs
# (Cloud Run included), so YouTube is routed through Apify like the other
# platforms instead. starvibe/youtube-video-transcript picked for its 99.99%
# success rate across ~490k runs at evaluation time (2026-07-28).
APIFY_YOUTUBE_ACTOR = os.environ.get("APIFY_YOUTUBE_ACTOR", "starvibe/youtube-video-transcript")
APIFY_RUN_TIMEOUT_S = int(os.environ.get("APIFY_RUN_TIMEOUT_S", "60"))
APIFY_WAIT_TIMEOUT_S = int(os.environ.get("APIFY_WAIT_TIMEOUT_S", "65"))
APIFY_HTTP_TIMEOUT_S = int(os.environ.get("APIFY_HTTP_TIMEOUT_S", "70"))
MONITOR_MAX_SOURCES = int(os.environ.get("MONITOR_MAX_SOURCES", "25"))
MONITOR_MAX_ITEMS_PER_SOURCE = int(os.environ.get("MONITOR_MAX_ITEMS_PER_SOURCE", "5"))
# Keep live extraction bounded when an AI provider is under transient load.
# Dry-run discovery still reports the full per-source candidate set.
MONITOR_MAX_LIVE_ITEMS_PER_SOURCE = int(os.environ.get("MONITOR_MAX_LIVE_ITEMS_PER_SOURCE", "2"))
MONITOR_RUN_LOCK = Lock()
MONITOR_HTTP_TIMEOUT_S = int(os.environ.get("MONITOR_HTTP_TIMEOUT_S", "45"))
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
FIRECRAWL_API_URL = os.environ.get("FIRECRAWL_API_URL", "https://api.firecrawl.dev")

JINA_READER_BASE = "https://r.jina.ai/"
# "gemini-1.5-flash" was the original spec's model; that line (and even 2.5-flash)
# is now retired for new API keys. "gemini-flash-latest" is Google's rolling
# alias to the current-gen Flash model, so this default won't rot again.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
# Notion's limit is measured differently for some Unicode text; keep a
# conservative margin so page creation cannot leave an orphan on append.
NOTION_BLOCK_CHAR_LIMIT = 1800

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


def run_apify_actor(actor_id: str, run_input: dict[str, Any]) -> Any:
    """Run an actor with bounded actor, wait, and API request timeouts."""
    return apify_client.actor(actor_id).call(
        run_input=run_input,
        run_timeout=timedelta(seconds=APIFY_RUN_TIMEOUT_S),
        wait_duration=timedelta(seconds=APIFY_WAIT_TIMEOUT_S),
        timeout=timedelta(seconds=APIFY_HTTP_TIMEOUT_S),
    )


def extract_youtube(url: str) -> dict[str, Any]:
    """Fetch transcript + title/channel for a YouTube video via Apify (direct
    youtube-transcript-api calls get blocked from cloud-provider IPs), and
    also hand the URL to Gemini directly as a video (video_uri) — Gemini
    fetches YouTube URLs natively, so it can watch/listen to the actual video
    (on-screen text, visuals) instead of relying only on the caption track."""
    log.info("YouTube: running Apify actor %s for %s", APIFY_YOUTUBE_ACTOR, url)
    run = run_apify_actor(
        APIFY_YOUTUBE_ACTOR,
        {"youtube_url": url, "include_transcript_text": True},
    )
    if run is None:
        raise ValueError(f"Apify actor {APIFY_YOUTUBE_ACTOR} run failed to start for {url}")

    items = list(
        apify_client.dataset(run.default_dataset_id).iterate_items(
            limit=1, timeout=timedelta(seconds=APIFY_HTTP_TIMEOUT_S)
        )
    )
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


def extract_linkedin_video_url(item: dict[str, Any]) -> Optional[str]:
    """supreme_coder/linkedin-post nests the real downloadable video URL under
    linkedinVideo.videoPlayMetadata.progressiveStreams[N].streamingLocations[0].url
    — confirmed against a real live LinkedIn video post (2026-08-07). It's a
    signed but directly-downloadable LinkedIn CDN link, same as Instagram/TikTok's
    own CDN links — no auth wrapper needed."""
    streams = (item.get("linkedinVideo") or {}).get("videoPlayMetadata", {}).get(
        "progressiveStreams"
    ) or []
    if streams:
        locations = streams[0].get("streamingLocations") or []
        if locations:
            return locations[0].get("url")
    return None


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
        # LinkedIn. Verified against real live output (2026-08-07): the
        # previously configured actor `apify/linkedin-post-scraper` doesn't
        # exist at all (confirmed via Apify's own API — every LinkedIn save
        # was failing outright, not just extracting badly). Replaced with
        # supreme_coder/linkedin-post, whose real schema just needs `urls`.
        run_input = {"urls": [url], "limitPerSource": 1}

    run = run_apify_actor(actor_id, run_input)
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
    items = list(
        apify_client.dataset(run.default_dataset_id).iterate_items(
            limit=1, timeout=timedelta(seconds=APIFY_HTTP_TIMEOUT_S)
        )
    )

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
    video_url = (
        item.get("videoUrl")
        or next(iter(item.get("mediaUrls") or []), None)
        or extract_linkedin_video_url(item)
    )

    images = []
    if item.get("displayUrl"):
        images.append(item["displayUrl"])
    if item.get("images"):
        images.extend(item["images"])
    if item.get("isSlideshow"):
        # TikTok "photo mode" posts have no video at all — mediaUrls is empty
        # and videoMeta is all zeros. The actual carousel lives in
        # slideshowImageLinks instead; downloadLink is Apify's own hosted,
        # non-expiring copy (preferred over the signed/expiring tiktokLink).
        # Confirmed via a real slideshow post (2026-08-07): mediaUrls == [].
        for slide in item.get("slideshowImageLinks") or []:
            link = slide.get("downloadLink") or slide.get("tiktokLink")
            if link:
                images.append(link)
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


def with_apify_auth(url: str) -> str:
    """Apify's own key-value-store URLs (returned when we set
    shouldDownloadVideos/shouldDownloadSlideshowImages/shouldDownloadCovers)
    are private by default and 403 on a plain unauthenticated GET — confirmed
    against a real TikTok video URL (2026-08-07). Append the API token only
    for Apify-hosted URLs; third-party CDN links (Instagram, TikTok's own
    signed URLs) don't accept or need it."""
    if urlparse(url).netloc == "api.apify.com":
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}token={APIFY_API_TOKEN}"
    return url


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

# A separate routing taxonomy keeps the existing broad Category field stable.
# These values are also the recommended options for the Notion select property.
ROUTING_CATEGORY_OPTIONS = [
    "Events",
    "Follow-Up / To-Do",
    "AI & Tech",
    "People / Contacts",
    "Travel / Festivals",
    "Knowledge",
    "Other",
]

TAGGING_PROMPT = """You are a universal tagging engine for a personal "MyMind"-style
knowledge base. You will be given the extracted text, and possibly images or a video,
from a saved web link. Analyze it and return STRICT JSON matching this schema:

{{
  "transcription": "The full verbatim content. For a video: the spoken-word audio transcript AND any on-screen/overlay/caption text, each clearly labeled (e.g. 'Spoken:' / 'On-screen text:'). For images: OCR'd text. For articles: the full body text.",
  "ai_summary": "A 2-sentence summary of the link's core value or main thesis",
  "category": "Exactly one of: {category_options}",
  "routing_category": "Exactly one of: {routing_category_options}; choose the destination/use-case, not the broad topic",
  "tags": ["Tag1", "Tag2", "Tag3"],
  "vibe": "Short description of tone/aesthetic (e.g., 'Educational', 'Minimalist', 'Opinionated')",
  "author_name": "Name of the creator, author, or website publisher",
  "city": "Explicit city or null",
  "country": "Explicit country or null",
  "event_date": "Legacy single event date; use event_start for new captures",
  "event_start": "ISO date or datetime or null; only when clearly stated",
  "event_end": "ISO date or datetime or null; only when clearly stated",
  "event_status": "Upcoming, Past, Ongoing, or null",
  "relevance_score": 0,
  "routing_status": "Inbox, Needs Review, or Actionable",
  "discovery_source": "Short source label such as Instagram, Luma, or Web",
  "why_it_matched": "One short sentence explaining why this is worth saving",
  "related_contact_name": "Exact existing contact name only; otherwise null",
  "related_festival_name": "Exact existing festival name only; otherwise null",
  "related_edition_name": "Exact existing edition name only; otherwise null",
  "is_recurring_festival": false
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


def normalize_routing_category(raw_category: str) -> str:
    """Coerce Gemini's routing category to the controlled select options."""
    lookup = {opt.casefold(): opt for opt in ROUTING_CATEGORY_OPTIONS}
    return lookup.get((raw_category or "").strip().casefold(), "Other")


GEMINI_FILE_POLL_INTERVAL_S = 2
GEMINI_FILE_POLL_TIMEOUT_S = 90
# Carousels can run to 20+ slides; a single combined Gemini request can
# silently skip slides even when all images are attached, so each slide is
# OCR'd in its own call instead and the results aggregated before tagging.
# Keep the bounded OCR workload below the monitor request budget.
GEMINI_MAX_CAROUSEL_SLIDES = int(os.environ.get("GEMINI_MAX_CAROUSEL_SLIDES", "8"))
# Slide OCR calls are independent, so run them concurrently instead of
# sequentially — a 20-slide carousel run one-at-a-time (each with its own
# bounded 503 retry/backoff) was the single biggest source of
# worst-case webhook latency. Capped well under Gemini's per-project rate
# limit rather than left unbounded.
GEMINI_OCR_MAX_WORKERS = 6


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
        resp = requests.get(with_apify_auth(video_url), timeout=60, stream=True)
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


# Gemini can return short-lived 503 high-demand responses even when the
# service is healthy. Keep retries bounded, but give a transient incident
# enough time to clear before failing the capture.
GEMINI_RETRY_MAX_ATTEMPTS = 5
GEMINI_RETRY_BACKOFF_BASE_S = 5


def generate_content_with_retry(**kwargs: Any) -> Any:
    """gemini_client.models.generate_content, with a bounded retry for
    Gemini's own transient 503 UNAVAILABLE ("high demand") errors — seen
    repeatedly in production (2026-08-07): 5 of 18 carousel slide OCR calls
    and 3 of 3 attempts on one video's tagging call all hit this in the same
    session. Not retried for any other error type."""
    last_error: Optional[Exception] = None
    primary_model = kwargs.get("model")
    for attempt in range(GEMINI_RETRY_MAX_ATTEMPTS):
        try:
            return gemini_client.models.generate_content(**kwargs)
        except Exception as e:
            is_503 = "503" in str(e) or "UNAVAILABLE" in str(e)
            if not is_503 or attempt == GEMINI_RETRY_MAX_ATTEMPTS - 1:
                last_error = e
                break
            last_error = e
            wait_s = GEMINI_RETRY_BACKOFF_BASE_S * (2**attempt)
            log.warning(
                "Gemini: 503 (high demand), retrying in %ds (attempt %d/%d)",
                wait_s,
                attempt + 1,
                GEMINI_RETRY_MAX_ATTEMPTS,
            )
            time.sleep(wait_s)

    if last_error and primary_model != GEMINI_FALLBACK_MODEL:
        log.warning("Gemini: primary model unavailable; trying fallback model %s", GEMINI_FALLBACK_MODEL)
        fallback_kwargs = {**kwargs, "model": GEMINI_FALLBACK_MODEL}
        try:
            return gemini_client.models.generate_content(**fallback_kwargs)
        except Exception as fallback_error:
            if "503" not in str(fallback_error) and "UNAVAILABLE" not in str(fallback_error):
                raise
            last_error = fallback_error

    raise last_error  # unreachable only when the client returns successfully


def ocr_image_slide(image_url: str, index: int) -> Optional[str]:
    """Download one carousel slide and OCR just that image in its own Gemini
    call, labeled by position. Returns None if the slide can't be downloaded,
    isn't an image, or Gemini returns nothing — callers should skip it rather
    than fail the whole carousel over one bad slide."""
    try:
        img_resp = requests.get(with_apify_auth(image_url), timeout=15)
        img_resp.raise_for_status()
        mime_type = img_resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
        if not mime_type.startswith("image/"):
            log.warning("Gemini: slide %d has non-image content-type %s, skipping", index, mime_type)
            return None
    except requests.RequestException as e:
        log.warning("Gemini: could not download slide %d (%s): %s", index, image_url, e)
        return None

    try:
        response = generate_content_with_retry(
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
        # request (see GEMINI_MAX_CAROUSEL_SLIDES comment above). Run the
        # per-slide calls concurrently (see GEMINI_OCR_MAX_WORKERS) — order is
        # preserved by executor.map regardless of completion order, and each
        # slide is already labeled by its own index inside ocr_image_slide.
        log.info(
            "Gemini: OCR'ing %d carousel slide(s), %d concurrent",
            len(images),
            min(GEMINI_OCR_MAX_WORKERS, len(images)),
        )
        with ThreadPoolExecutor(max_workers=min(GEMINI_OCR_MAX_WORKERS, len(images))) as executor:
            ocr_results = list(
                executor.map(lambda pair: ocr_image_slide(pair[1], pair[0]), enumerate(images, start=1))
            )
        slide_texts = [text for text in ocr_results if text]
        if slide_texts:
            content += "\n\n" + "\n\n".join(slide_texts)

    prompt = TAGGING_PROMPT.format(
        url=url,
        author_name=extracted.get("author_name"),
        category_options=", ".join(CATEGORY_OPTIONS),
        routing_category_options=", ".join(ROUTING_CATEGORY_OPTIONS),
        content=content[:50000],  # keep prompt bounded
    )

    parts: list[Any] = [prompt]

    if video_part:
        parts.append(video_part)
    elif len(images) == 1:
        # Single image (plain photo post, or a cover-thumbnail fallback) —
        # cheap enough to send inline with the main tagging call.
        try:
            img_resp = requests.get(with_apify_auth(images[0]), timeout=15)
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
        response = generate_content_with_retry(
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
    result.setdefault("routing_category", "Other")
    result.setdefault("tags", [])
    result.setdefault("vibe", "")
    result.setdefault("author_name", extracted.get("author_name"))
    result.setdefault("city", None)
    result.setdefault("country", None)
    result.setdefault("event_date", None)
    result.setdefault("event_start", result.get("event_date"))
    result.setdefault("event_end", None)
    result["event_date"] = result.get("event_date") or result.get("event_start")
    result.setdefault("event_status", None)
    result.setdefault("relevance_score", 0)
    result.setdefault("routing_status", "Inbox")
    result.setdefault("discovery_source", "Web")
    result.setdefault("why_it_matched", "")
    result.setdefault("related_contact_name", None)
    result.setdefault("related_festival_name", None)
    result.setdefault("related_edition_name", None)
    result.setdefault("is_recurring_festival", False)

    # The Notion "Category" property is a fixed-option select — coerce
    # whatever Gemini returned onto one of the allowed options.
    result["category"] = normalize_category(result["category"])
    result["routing_category"] = normalize_routing_category(result["routing_category"])

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
_database_properties: Optional[dict[str, Any]] = None


def normalize_entity_name(value: str) -> str:
    return " ".join((value or "").casefold().split())


def normalize_source_url(value: Optional[str]) -> str:
    """Canonicalize a configured source URL for deterministic matching."""
    if not isinstance(value, str) or not value.strip():
        return ""
    parsed = urlparse(value.strip())
    scheme = (parsed.scheme or "https").lower()
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/") or "/"
    return f"{scheme}://{host}{path}"


def find_monitored_source_page(source_url: Optional[str]) -> Optional[str]:
    """Return one exact source-record match, never creating or guessing one."""
    wanted = normalize_source_url(source_url)
    if not wanted:
        return None
    query = {
        "data_source_id": CONTACTS_DATA_SOURCE_ID,
        "filter": {"property": "Record Type", "select": {"equals": "Monitored Source"}},
        "page_size": 100,
    }
    try:
        pages = []
        while True:
            results = notion.data_sources.query(**query)
            pages.extend(results.get("results") or [])
            if not results.get("has_more"):
                break
            next_cursor = results.get("next_cursor")
            if not next_cursor:
                log.warning("Notion: source lookup returned has_more without next_cursor")
                break
            query["start_cursor"] = next_cursor
    except Exception as e:
        log.warning("Notion: source lookup failed for %r (%s)", source_url, e)
        return None

    matches = []
    for page in pages:
        source_prop = (page.get("properties") or {}).get("Source URL", {})
        source_value = source_prop.get("url")
        if normalize_source_url(source_value) == wanted:
            matches.append(page["id"])
    return matches[0] if len(matches) == 1 else None


def list_monitored_sources() -> list[dict[str, Any]]:
    """Read the existing Contacts source registry without creating records.

    Monitoring is opt-in: only rows marked Monitored Source, enabled, and
    Active are eligible for a future poller. The cap prevents one malformed
    or unexpectedly large Contacts database from turning a scheduler tick
    into an unbounded scrape run.
    """
    query = {
        "data_source_id": CONTACTS_DATA_SOURCE_ID,
        "filter": {
            "and": [
                {"property": "Record Type", "select": {"equals": "Monitored Source"}},
                {"property": "Monitoring Enabled", "checkbox": {"equals": True}},
                {"property": "Monitoring Status", "select": {"equals": "Active"}},
            ]
        },
        "page_size": min(MONITOR_MAX_SOURCES, 100),
    }
    pages: list[dict[str, Any]] = []
    while len(pages) < MONITOR_MAX_SOURCES:
        results = notion.data_sources.query(**query)
        pages.extend(results.get("results") or [])
        if not results.get("has_more") or not results.get("next_cursor"):
            break
        query["start_cursor"] = results["next_cursor"]

    sources = []
    for page in pages[:MONITOR_MAX_SOURCES]:
        props = page.get("properties") or {}
        title_parts = (props.get("Name", {}).get("title") or [])
        name = "".join(part.get("plain_text", "") for part in title_parts).strip()
        source_url = props.get("Source URL", {}).get("url")
        source_type = (props.get("Source Type", {}).get("select") or {}).get("name")
        polling_frequency = (props.get("Polling Frequency", {}).get("select") or {}).get("name")
        monitoring_enabled = bool(props.get("Monitoring Enabled", {}).get("checkbox"))
        monitoring_status = (props.get("Monitoring Status", {}).get("select") or {}).get("name")
        include_keywords = [
            part.strip().casefold()
            for part in "".join(
                item.get("plain_text", "")
                for item in (props.get("Include Keywords", {}).get("rich_text") or [])
            ).split(",")
            if part.strip()
        ]
        exclude_keywords = [
            part.strip().casefold()
            for part in "".join(
                item.get("plain_text", "")
                for item in (props.get("Exclude Keywords", {}).get("rich_text") or [])
            ).split(",")
            if part.strip()
        ]
        require_event_intent = bool((props.get("Require Event Intent", {}).get("checkbox")))
        sources.append({
            "id": page.get("id"),
            "name": name,
            "source_url": source_url,
            "source_type": source_type,
            "polling_frequency": polling_frequency,
            "monitoring_enabled": monitoring_enabled,
            "monitoring_status": monitoring_status,
            "include_keywords": include_keywords,
            "exclude_keywords": exclude_keywords,
            "require_event_intent": require_event_intent,
        })
    return sources


PUBLIC_SOURCE_TYPES = {"Website", "Luma", "Calendar", "Newsletter", "Other"}
UNSUPPORTED_SOURCE_TYPES = {"Instagram", "TikTok", "LinkedIn", "YouTube"}
MONITOR_SKIP_PATHS = {"about", "contact", "login", "privacy", "terms", "pricing", "faq"}
MONITOR_EVENT_INTENT_KEYWORDS = [
    "event", "events", "workshop", "workshops", "exhibition", "exhibitions",
    "activation", "activations", "festival", "programming", "ticketed", "reserve",
    "sign up", "admission", "opening", "talk", "panel", "class", "concert",
]


def scrape_monitor_source(source_url: str) -> tuple[str, list[str]]:
    """Fetch a public source page and return bounded text plus page links."""
    if not FIRECRAWL_API_KEY:
        raise RuntimeError("FIRECRAWL_API_KEY is not configured")
    response = requests.post(
        f"{FIRECRAWL_API_URL.rstrip('/')}/v2/scrape",
        headers={
            "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "url": source_url,
            "formats": ["markdown", "links"],
            "onlyMainContent": True,
            "maxAge": 900000,
        },
        timeout=MONITOR_HTTP_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") or {}
    return str(data.get("markdown") or "")[:50000], [
        link for link in (data.get("links") or []) if isinstance(link, str)
    ]


def monitor_candidate_matches(
    candidate_url: str,
    include_keywords: list[str],
    exclude_keywords: list[str],
    required_keywords: list[str] | None = None,
) -> bool:
    """Apply the configured source filter to a public candidate page."""
    markdown, _links = scrape_monitor_source(candidate_url)
    return matches_monitor_keywords(
        f"{candidate_url}\n{markdown}", include_keywords, exclude_keywords, required_keywords
    )


def select_monitor_candidates(source_url: str, links: list[str]) -> list[str]:
    """Select likely content pages without following arbitrary external links."""
    source = urlparse(source_url)
    seen: set[str] = set()
    candidates: list[str] = []
    for link in links:
        parsed = urlparse(link)
        if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() != source.netloc.lower():
            continue
        normalized = normalize_source_url(link)
        if not normalized or normalized == normalize_source_url(source_url) or normalized in seen:
            continue
        path_parts = [part.casefold() for part in parsed.path.split("/") if part]
        if not path_parts or path_parts[-1] in MONITOR_SKIP_PATHS:
            continue
        if any(part in MONITOR_SKIP_PATHS for part in path_parts):
            continue
        if any(parsed.path.casefold().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".pdf")):
            continue
        seen.add(normalized)
        candidates.append(link)
        if len(candidates) >= MONITOR_MAX_ITEMS_PER_SOURCE:
            break
    return candidates


def explain_monitor_keywords(
    text: str,
    include_keywords: list[str],
    exclude_keywords: list[str],
    required_keywords: list[str] | None = None,
) -> dict[str, Any]:
    haystack = text.casefold()
    def keyword_matches(keyword: str) -> bool:
        normalized = keyword.strip().casefold()
        if not normalized:
            return False
        pattern = rf"(?<!\w){re.escape(normalized)}(?!\w)"
        return re.search(pattern, haystack) is not None

    matched_excludes = [keyword for keyword in exclude_keywords if keyword_matches(keyword)]
    matched_includes = [keyword for keyword in include_keywords if keyword_matches(keyword)]
    matched_required = [keyword for keyword in (required_keywords or []) if keyword_matches(keyword)]
    event_required = bool(required_keywords)
    return {
        "decision": (
            "reject"
            if matched_excludes
            or (include_keywords and not matched_includes)
            or (event_required and not matched_required)
            else "pass"
        ),
        "matched_include_keywords": matched_includes,
        "matched_exclude_keywords": matched_excludes,
        "matched_required_keywords": matched_required,
    }


def matches_monitor_keywords(
    text: str,
    include_keywords: list[str],
    exclude_keywords: list[str],
    required_keywords: list[str] | None = None,
) -> bool:
    return explain_monitor_keywords(text, include_keywords, exclude_keywords, required_keywords)["decision"] == "pass"


def should_skip_monitored_event(ai_data: dict[str, Any]) -> bool:
    """Keep historical event pages out of the live event-saver inbox."""
    return str(ai_data.get("event_status") or "").strip().casefold() == "past"


def poll_instagram_source(
    source_url: str,
    include_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None,
    diagnostics: bool = False,
    require_event_intent: bool = False,
) -> list[str] | list[dict[str, Any]]:
    """Return bounded post URLs from one Instagram profile via Apify."""
    include_keywords = include_keywords or []
    exclude_keywords = exclude_keywords or []
    run = run_apify_actor(
        APIFY_INSTAGRAM_ACTOR,
        {"directUrls": [source_url], "resultsLimit": MONITOR_MAX_ITEMS_PER_SOURCE},
    )
    if run is None:
        raise ValueError(f"Apify actor {APIFY_INSTAGRAM_ACTOR} run failed to start for {source_url}")
    run_status = getattr(run, "status", None)
    if run_status and run_status != "SUCCEEDED":
        raise ValueError(
            f"Apify actor {APIFY_INSTAGRAM_ACTOR} run did not succeed "
            f"(status={run_status}): {getattr(run, 'status_message', None)}"
        )
    items = apify_client.dataset(run.default_dataset_id).iterate_items(
        limit=MONITOR_MAX_ITEMS_PER_SOURCE,
        timeout=timedelta(seconds=APIFY_HTTP_TIMEOUT_S),
    )
    candidates: list[str] = []
    seen: set[str] = set()
    for item in items:
        post_url = item.get("url") or item.get("postUrl") or item.get("webUrl")
        if not isinstance(post_url, str) or classify_url(post_url) != "instagram":
            continue
        parsed = urlparse(post_url)
        if not any(part in parsed.path.split("/") for part in ("p", "reel", "tv")):
            continue
        searchable_text = " ".join(
            str(item.get(field) or "")
            for field in ("caption", "text", "alt", "title", "url")
        )
        explanation = explain_monitor_keywords(
            searchable_text,
            include_keywords,
            exclude_keywords,
            MONITOR_EVENT_INTENT_KEYWORDS if require_event_intent else None,
        )
        if diagnostics:
            diagnostic = {
                "url": post_url,
                "decision": explanation["decision"],
                "matched_include_keywords": explanation["matched_include_keywords"],
                "matched_exclude_keywords": explanation["matched_exclude_keywords"],
                "matched_required_keywords": explanation["matched_required_keywords"],
                "text_preview": searchable_text[:1000],
            }
            candidates.append(diagnostic)
            continue
        if explanation["decision"] != "pass":
            continue
        normalized = normalize_source_url(post_url)
        if normalized and normalized not in seen:
            seen.add(normalized)
            candidates.append(post_url)
        if not diagnostics and len(candidates) >= MONITOR_MAX_ITEMS_PER_SOURCE:
            break
    return candidates


def run_monitoring(
    *,
    dry_run: bool = True,
    diagnostics: bool = False,
    source_urls: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Poll eligible public sources; dry-run never writes Inbox pages."""
    if not MONITOR_RUN_LOCK.acquire(blocking=False):
        log.warning("Monitoring run skipped because another run is still in progress")
        return {"dry_run": dry_run, "status": "busy", "source_count": 0, "results": []}

    try:
        sources = list_monitored_sources()
        if source_urls:
            wanted_sources = {normalize_source_url(url) for url in source_urls}
            sources = [
                source
                for source in sources
                if normalize_source_url(source.get("source_url")) in wanted_sources
            ]
        results: list[dict[str, Any]] = []
        seen_external_ids: set[str] = set()
        for source in sources:
            source_type = source.get("source_type")
            source_url = source.get("source_url")
            result: dict[str, Any] = {"source": source, "status": "skipped", "candidates": []}
            if source_type == "Instagram" and source_url:
                try:
                    candidates = poll_instagram_source(
                        source_url,
                        source.get("include_keywords") or [],
                        source.get("exclude_keywords") or [],
                        diagnostics=diagnostics and dry_run,
                        require_event_intent=source.get("require_event_intent", False),
                    )
                    if diagnostics and dry_run:
                        result["diagnostics"] = candidates
                        result["candidates"] = [item["url"] for item in candidates]
                    else:
                        result["candidates"] = candidates
                    result["status"] = "dry_run" if dry_run else "processed"
                    if not dry_run:
                        unique_candidates = []
                        for url in candidates[:MONITOR_MAX_LIVE_ITEMS_PER_SOURCE]:
                            external_id = normalize_source_url(url)
                            if external_id and external_id in seen_external_ids:
                                continue
                            if external_id:
                                seen_external_ids.add(external_id)
                            unique_candidates.append(url)
                        result["captures"] = [
                            process_capture(
                                url=url,
                                discovery_source_url=source_url,
                                external_id_override=normalize_source_url(url),
                                skip_past_events=True,
                            )
                            for url in unique_candidates
                        ]
                except Exception as e:
                    log.exception("Instagram monitoring failed for %s", source_url)
                    result["status"] = "error"
                    result["reason"] = str(e)
            elif source_type in UNSUPPORTED_SOURCE_TYPES:
                result["reason"] = "source type adapter not implemented"
            elif source_type not in PUBLIC_SOURCE_TYPES or not source_url:
                result["reason"] = "missing URL or unsupported source type"
            else:
                try:
                    _markdown, links = scrape_monitor_source(source_url)
                    candidates = select_monitor_candidates(source_url, links)
                    include_keywords = source.get("include_keywords") or []
                    exclude_keywords = source.get("exclude_keywords") or []
                    candidates = [
                        url
                        for url in candidates
                        if monitor_candidate_matches(
                            url,
                            include_keywords,
                            exclude_keywords,
                            MONITOR_EVENT_INTENT_KEYWORDS if source.get("require_event_intent", False) else None,
                        )
                    ]
                    result["candidates"] = candidates
                    result["status"] = "dry_run" if dry_run else "processed"
                    if not dry_run:
                        unique_candidates = []
                        for url in candidates[:MONITOR_MAX_LIVE_ITEMS_PER_SOURCE]:
                            external_id = normalize_source_url(url)
                            if external_id and external_id in seen_external_ids:
                                continue
                            if external_id:
                                seen_external_ids.add(external_id)
                            unique_candidates.append(url)
                        result["captures"] = [
                            process_capture(
                                url=url,
                                discovery_source_url=source_url,
                                external_id_override=normalize_source_url(url),
                                skip_past_events=True,
                            )
                            for url in unique_candidates
                        ]
                except Exception as e:
                    log.exception("Monitoring failed for %s", source_url)
                    result["status"] = "error"
                    result["reason"] = str(e)
            results.append(result)
        return {"dry_run": dry_run, "source_count": len(sources), "results": results}
    finally:
        MONITOR_RUN_LOCK.release()


def find_exact_related_page(data_source_id: str, title_property: str, candidate: Any) -> Optional[str]:
    """Return a page ID only for one exact normalized title match."""
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    candidate = candidate.strip()[:2000]
    try:
        results = notion.data_sources.query(
            data_source_id=data_source_id,
            filter={"property": title_property, "title": {"contains": candidate}},
            page_size=100,
        )
    except Exception as e:
        log.warning("Notion: relation lookup failed for %r (%s)", candidate, e)
        return None

    matches = []
    wanted = normalize_entity_name(candidate)
    for page in results.get("results") or []:
        for prop in (page.get("properties") or {}).values():
            if prop.get("type") != "title":
                continue
            title_parts = prop.get("title") or []
            actual = "".join(part.get("plain_text", "") for part in title_parts)
            if normalize_entity_name(actual) == wanted:
                matches.append(page["id"])
            break
    return matches[0] if len(matches) == 1 else None


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


def get_database_properties() -> dict[str, Any]:
    """Return the current database schema, cached for this process.

    Expansion fields are optional until the Notion schema migration is applied.
    This lets the deployed saver remain compatible during that transition.
    """
    global _database_properties
    if _database_properties is None:
        # With the current Notion API, configurable properties live on the
        # data source rather than the parent database object.
        db = notion.data_sources.retrieve(data_source_id=get_data_source_id())
        _database_properties = db.get("properties") or {}
    return _database_properties


def add_optional_notion_properties(properties: dict[str, Any], ai_data: dict[str, Any]) -> None:
    """Add expansion fields only when the corresponding Notion property exists.

    Relations are deliberately excluded here: resolving a person/festival match
    requires an explicit identity/deduplication policy and belongs to Phase 3.
    """
    try:
        schema = get_database_properties()
    except Exception as e:
        log.warning("Notion: optional expansion schema unavailable; using core fields (%s)", e)
        return
    optional_values = {
        "City": ("rich_text", ai_data.get("city")),
        "Country": ("rich_text", ai_data.get("country")),
        "Event Date": ("date", ai_data.get("event_date")),
        "Event Start": ("date", ai_data.get("event_start") or ai_data.get("event_date")),
        "Event End": ("date", ai_data.get("event_end")),
        "Event Status": ("select", ai_data.get("event_status")),
        "Relevance Score": ("number", ai_data.get("relevance_score")),
        "Routing Status": ("select", ai_data.get("routing_status")),
        "Routing Category": ("select", ai_data.get("routing_category")),
        "Discovery Source": ("rich_text", ai_data.get("discovery_source")),
        "Why It Matched": ("rich_text", ai_data.get("why_it_matched")),
    }
    for name, (property_type, value) in optional_values.items():
        if name not in schema or value in (None, ""):
            continue
        if schema[name].get("type") != property_type:
            log.warning("Notion: ignoring %s because its type is not %s", name, property_type)
            continue
        if property_type == "rich_text":
            properties[name] = {"rich_text": [{"type": "text", "text": {"content": str(value)[:2000]}}]}
        elif property_type == "select":
            allowed = {option.get("name") for option in schema[name].get("select", {}).get("options", [])}
            if allowed and str(value) not in allowed:
                log.warning("Notion: ignoring %s value %r; option is not configured", name, value)
                continue
            properties[name] = {"select": {"name": str(value)}}
        elif property_type == "date":
            properties[name] = {"date": {"start": str(value)}}
        elif property_type == "number":
            try:
                properties[name] = {"number": max(0, min(100, int(value)))}
            except (TypeError, ValueError):
                log.warning("Notion: ignoring non-numeric Relevance Score %r", value)


def add_exact_relations(properties: dict[str, Any], ai_data: dict[str, Any]) -> None:
    """Link only safe existing canonical records; never create or guess one.

    Contacts stay manual-confirmation-only because an extracted organizer,
    author, venue, or speaker is not necessarily a canonical contact. Festival
    and Edition links require an explicit recurring-festival classification.
    """
    try:
        schema = get_database_properties()
        relation_targets = {}
        if ai_data.get("is_recurring_festival") is True:
            relation_targets = {
                "Related Festival": (FESTIVALS_DATA_SOURCE_ID, "Festival", ai_data.get("related_festival_name")),
                "Related Edition": (EDITIONS_DATA_SOURCE_ID, "Edition", ai_data.get("related_edition_name")),
            }
        for property_name, (data_source_id, title_property, candidate) in relation_targets.items():
            if schema.get(property_name, {}).get("type") != "relation":
                continue
            page_id = find_exact_related_page(data_source_id, title_property, candidate)
            if page_id:
                properties[property_name] = {"relation": [{"id": page_id}]}
    except Exception as e:
        log.warning("Notion: relation matching unavailable; leaving relations empty (%s)", e)


def find_existing_page(external_id: str) -> Optional[dict[str, Any]]:
    """Return {"id", "title"} for the page already saved for this URL, if any.
    Checked at the top of process_url (before any Apify/Gemini work) so a
    retried webhook call (e.g. a phone client re-sending after its own
    timeout fires) short-circuits immediately instead of redoing the slowest,
    most expensive stages just to discover it's a duplicate at the very end."""
    results = notion.data_sources.query(
        data_source_id=get_data_source_id(),
        filter={"property": "External ID", "rich_text": {"equals": external_id[:2000]}},
        page_size=1,
    )
    pages = results.get("results") or []
    if not pages:
        return None
    page = pages[0]
    title_parts = (page.get("properties", {}).get("Title", {}).get("title") or [])
    title = title_parts[0]["plain_text"] if title_parts else external_id
    return {"id": page["id"], "title": title}


def stable_text_external_id(text: str, title: str = "") -> str:
    """Build a deterministic dedup key for text-only captures."""
    digest = hashlib.sha256(f"{title}\n{text}".encode("utf-8")).hexdigest()
    return f"text:{digest}"


def push_to_notion(
    url: Optional[str],
    platform: str,
    title: str,
    ai_data: dict[str, Any],
    external_id: Optional[str] = None,
    discovery_source_url: Optional[str] = None,
) -> str:
    """Create a Notion page with mapped properties + transcription page content.
    Returns the new page ID. Callers are expected to have already checked
    find_existing_page — this always creates."""
    properties = {
        "Title": {"title": [{"text": {"content": title[:2000] or url}}]},
        "URL": {"url": url} if url else {"url": None},
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
        "External ID": {
            "rich_text": [
                {
                    "type": "text",
                    "text": {"content": (external_id or url or "")[:2000]},
                }
            ]
        },
    }
    source_page_id = find_monitored_source_page(discovery_source_url)
    add_optional_notion_properties(properties, ai_data)
    add_exact_relations(properties, ai_data)

    if source_page_id:
        try:
            schema = get_database_properties()
            if schema.get("Monitored Source", {}).get("type") == "relation":
                properties["Monitored Source"] = {"relation": [{"id": source_page_id}]}
        except Exception as e:
            log.warning("Notion: could not map monitored source relation (%s)", e)
    elif discovery_source_url:
        try:
            schema = get_database_properties()
            routing_schema = schema.get("Routing Status", {})
            allowed = {option.get("name") for option in routing_schema.get("select", {}).get("options", [])}
            if routing_schema.get("type") == "select" and "To Sort" in allowed:
                properties["Routing Status"] = {"select": {"name": "To Sort"}}
        except Exception as e:
            log.warning("Notion: could not mark unmatched source as To Sort (%s)", e)

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

def process_capture(
    *,
    url: Optional[str] = None,
    raw_text: Optional[str] = None,
    title_hint: Optional[str] = None,
    discovery_source_url: Optional[str] = None,
    external_id_override: Optional[str] = None,
    skip_past_events: bool = False,
) -> dict[str, Any]:
    """Full extraction -> Gemini -> Notion pipeline. Each stage's exception is
    caught and logged, then turned into a real {"ok": False, "error": ...}
    result instead of being swallowed, so the caller can report what actually
    happened."""
    if not url and not raw_text:
        return {"ok": False, "error": "A capture must include 'url' or non-empty 'text'"}

    title_hint = (title_hint or "").strip()
    external_id = external_id_override or url or stable_text_external_id(raw_text or "", title_hint)
    source_for_ai = url or "text://capture"
    log.info("=== Processing %s ===", source_for_ai)

    try:
        existing = find_existing_page(external_id)
    except Exception:
        log.exception("Dedup check failed for %s, proceeding as new save", source_for_ai)
        existing = None
    if existing:
        log.info("Notion: %s already saved as page %s, skipping re-extraction", source_for_ai, existing["id"])
        return {"ok": True, "title": existing["title"], "notion_page_id": existing["id"]}

    if raw_text is not None:
        platform = "general"
        extracted = {
            "text": raw_text.strip(),
            "title": title_hint or "Text capture",
            "author_name": None,
            "images": [],
        }
    else:
        try:
            platform, extracted = route_and_extract(url or "")
        except Exception as e:
            log.exception("Extraction failed for %s", source_for_ai)
            return {"ok": False, "error": f"Extraction failed: {e}"}

    try:
        ai_data = call_gemini(source_for_ai, extracted)
    except Exception as e:
        log.exception("Gemini tagging failed for %s", source_for_ai)
        return {"ok": False, "error": f"Gemini tagging failed: {e}"}

    if skip_past_events and should_skip_monitored_event(ai_data):
        log.info("Skipping past event capture for %s", source_for_ai)
        return {
            "ok": True,
            "status": "skipped_past_event",
            "title": extracted.get("title") or title_hint or url or "Text capture",
        }

    try:
        page_id = push_to_notion(
            url,
            platform,
            extracted.get("title") or title_hint or url or "Text capture",
            ai_data,
            external_id=external_id,
            discovery_source_url=discovery_source_url,
        )
    except Exception as e:
        log.exception("Notion push failed for %s", source_for_ai)
        return {"ok": False, "error": f"Notion push failed: {e}"}

    log.info("=== Done: %s ===", source_for_ai)
    return {
        "ok": True,
        "title": extracted.get("title") or title_hint or url or "Text capture",
        "notion_page_id": page_id,
    }


def process_url(url: str) -> dict[str, Any]:
    """Backward-compatible URL entry point."""
    return process_capture(url=url)


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
    raw_text = payload.get("text")
    if not url and not raw_text:
        raise HTTPException(status_code=400, detail="Missing 'url' or non-empty 'text' field in payload")

    log.info("Accepted webhook capture (%s)", "url" if url else "text")
    # process_url does blocking I/O (requests/apify_client/notion_client are all
    # sync); offload to a threadpool so the event loop can still serve /health etc.
    result = await run_in_threadpool(
        process_capture,
        url=url,
        raw_text=raw_text,
        title_hint=payload.get("title"),
        discovery_source_url=payload.get("discovery_source_url"),
    )
    if not result["ok"]:
        raise HTTPException(status_code=502, detail=result["error"])
    return {"status": "saved", "title": result["title"], "notion_page_id": result["notion_page_id"]}


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/monitor/status")
async def monitor_status(x_api_key: Optional[str] = Header(default=None)):
    """Return the opt-in source set for a future scheduler/monitor worker.

    This endpoint intentionally performs no scraping and creates no Notion
    records. It is useful for deployment verification and for a scheduler to
    confirm that the existing Contacts registry is configured as expected.
    """
    if not x_api_key or not hmac.compare_digest(x_api_key, SECRET_WEBHOOK_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    try:
        sources = await run_in_threadpool(list_monitored_sources)
    except Exception as e:
        log.exception("Monitoring source registry lookup failed")
        raise HTTPException(status_code=502, detail=f"Monitoring source lookup failed: {e}")
    return {
        "status": "ready",
        "scraping_started": False,
        "source_count": len(sources),
        "sources": sources,
        "max_sources_per_run": MONITOR_MAX_SOURCES,
    }


@app.post("/monitor/run")
async def monitor_run(
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
):
    """Run the bounded public-source poller; writes require explicit dry_run=false."""
    if not x_api_key or not hmac.compare_digest(x_api_key, SECRET_WEBHOOK_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    dry_run = payload.get("dry_run", True) is not False
    diagnostics = payload.get("diagnostics", False) is True
    source_urls = payload.get("source_urls")
    if not isinstance(source_urls, list) or not all(isinstance(url, str) for url in source_urls):
        source_urls = None
    try:
        return await run_in_threadpool(
            run_monitoring,
            dry_run=dry_run,
            diagnostics=diagnostics,
            source_urls=source_urls,
        )
    except Exception as e:
        log.exception("Monitoring run failed")
        raise HTTPException(status_code=502, detail=f"Monitoring run failed: {e}")
