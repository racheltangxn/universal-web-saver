# Universal Web Saver

Python service for turning supported web, social, and video links into structured, searchable Notion records. It routes sources to format-specific extractors, adds transcription and tagging when available, and protects against duplicate captures.

## What it does

- Accepts authenticated webhook requests containing a URL.
- Routes YouTube, social, article, and other supported sources to the appropriate extractor.
- Uses Gemini for structured metadata and optional transcription/tagging.
- Writes records to Notion with duplicate protection.
- Includes an optional monitor for recurring sources.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in the environment variables with your own credentials.
uvicorn main:app --reload
```

The service expects credentials through environment variables. Never commit `.env`, runtime logs, or local virtual-environment files; they are excluded by `.gitignore`.

## Deployment

The included `Dockerfile` can be used with a container host such as Cloud Run. Configure secrets in the deployment environment and send the webhook key through the `X-API-Key` header.

## Scope

This repository contains the application code and deployment scaffolding. It does not include captured content, Notion data, runtime logs, or production credentials.
