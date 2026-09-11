"""Run one bounded monitoring pass for the Universal Web Saver."""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import classify_url, list_monitored_sources, run_monitoring


logging.basicConfig(level=logging.INFO)
meetup_sources = [
    source["source_url"]
    for source in list_monitored_sources()
    if classify_url(source.get("source_url") or "") == "meetup"
]
result = run_monitoring(dry_run=False, source_urls=meetup_sources)
print(json.dumps(result, default=str))
