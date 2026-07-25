"""
Universal Web Saver — webhook backend for the "Save to Notion" Android Share Sheet.

Pipeline:
  1. Android (HTTP Shortcuts) POSTs a URL to /webhook with an X-API-Key header.
  2. We validate the key, kick off a BackgroundTask, and return 200 immediately.
  3. The background task routes the URL by domain (Instagram/TikTok/LinkedIn via
     Apify, YouTube via transcript API, everything else via Jina Reader /
     trafilatura), sends the extracted content to Gemini for structured tagging,
     and pushes the result into a Notion database page.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import hmac
import logging
import os
import re
from datetime import date, datetime
from typing import Any, Optional
from urllib.parse import urlparse

import requests
import trafilatura
from apify_client import ApifyClient
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from google import genai
from google.genai import types as genai_types
from notion_client import Client as NotionClient
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

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


def extract_youtube_video_id(url: str) -> Optional[str]:
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")
    if "youtube.com" in parsed.netloc:
        if parsed.path == "/watch":
            match = re.search(r"[?&]v=([^&]+)", url)
            return match.group(1) if match else None
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/shorts/")[1].split("/")[0]
    return None


def extract_youtube(url: str) -> dict[str, Any]:
    """Fetch transcript + title for a YouTube video. No images/video download."""
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise ValueError(f"Could not parse YouTube video ID from {url}")

    log.info("YouTube: fetching transcript for video_id=%s", video_id)
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en", "en-US", "en-GB"])
        transcript_text = " ".join(snippet.text for snippet in transcript_list)
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        log.warning("YouTube: no English transcript available (%s)", e)
        transcript_text = ""

    # Lightweight title fetch via oEmbed — no API key required.
    title = url
    try:
        oembed = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=10,
        )
        oembed.raise_for_status()
        data = oembed.json()
        title = data.get("title", url)
        author_name = data.get("author_name")
    except requests.RequestException as e:
        log.warning("YouTube: oEmbed title fetch failed (%s)", e)
        author_name = None

    return {
        "text": transcript_text,
        "title": title,
        "author_name": author_name,
        "images": [],
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

    run_input = {"directUrls": [url], "urls": [url], "postUrls": [url]}
    run = apify_client.actor(actor_id).call(run_input=run_input)
    items = list(apify_client.dataset(run["defaultDatasetId"]).iterate_items())

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
    images = []
    if item.get("displayUrl"):
        images.append(item["displayUrl"])
    if item.get("videoUrl"):
        images.append(item["videoUrl"])
    if item.get("images"):
        images.extend(item["images"])
    if item.get("coverUrl"):
        images.append(item["coverUrl"])

    title = caption[:120] if caption else f"{platform.title()} post by {handle or 'unknown'}"

    return {
        "text": caption,
        "title": title,
        "author_name": handle,
        "images": [img for img in images if img],
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
knowledge base. You will be given the extracted text (and possibly images) from a
saved web link. Analyze it and return STRICT JSON matching this schema:

{{
  "transcription": "The verbatim OCR text, spoken words, or the full article body text",
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


def call_gemini(url: str, extracted: dict[str, Any]) -> dict[str, Any]:
    """Send extracted text (and optional images) to Gemini and get back
    structured JSON tags."""
    prompt = TAGGING_PROMPT.format(
        url=url,
        author_name=extracted.get("author_name"),
        category_options=", ".join(CATEGORY_OPTIONS),
        content=(extracted.get("text") or "")[:50000],  # keep prompt bounded
    )

    parts: list[Any] = [prompt]

    # For IG/TikTok posts, pull in a couple of images for vision context.
    for image_url in (extracted.get("images") or [])[:3]:
        try:
            img_resp = requests.get(image_url, timeout=15)
            img_resp.raise_for_status()
            mime_type = img_resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
            if mime_type.startswith("image/"):
                parts.append(
                    genai_types.Part.from_bytes(data=img_resp.content, mime_type=mime_type)
                )
        except requests.RequestException as e:
            log.warning("Gemini: could not download media %s (%s)", image_url, e)

    log.info("Gemini: sending %d part(s) for analysis", len(parts))
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=parts,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

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


def push_to_notion(url: str, platform: str, title: str, ai_data: dict[str, Any]) -> str:
    """Create a Notion page with mapped properties + transcription page content.
    Returns the new page ID."""
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

def process_url(url: str) -> None:
    """Full extraction -> Gemini -> Notion pipeline. Runs in the background;
    all exceptions are caught and logged so the request/response cycle is
    never affected."""
    log.info("=== Processing %s ===", url)
    try:
        platform, extracted = route_and_extract(url)
    except Exception:
        log.exception("Extraction failed for %s", url)
        return

    try:
        ai_data = call_gemini(url, extracted)
    except Exception:
        log.exception("Gemini tagging failed for %s", url)
        return

    try:
        push_to_notion(url, platform, extracted.get("title") or url, ai_data)
    except Exception:
        log.exception("Notion push failed for %s", url)
        return

    log.info("=== Done: %s ===", url)


# --------------------------------------------------------------------------- #
# FastAPI routes
# --------------------------------------------------------------------------- #

@app.post("/webhook")
async def webhook(
    request: Request,
    background_tasks: BackgroundTasks,
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
    background_tasks.add_task(process_url, url)
    return {"status": "Processing web content"}


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}
