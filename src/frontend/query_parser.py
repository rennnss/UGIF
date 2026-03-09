"""
Natural language query parser for the UGIF frontend.

Extracts location, date range, and task type from free-text queries using
regex patterns + optional spaCy NER. No LLM API key required.

Examples::

    parse_query("flood damage in Chennai August 2023")
    # → {'location': 'Chennai', 'start_date': '2023-08-01',
    #    'end_date': '2023-08-31', 'task': 'flood damage assessment'}
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, Optional


# ── Month name → number mapping ─────────────────────────────────────────────
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# ── Task keyword map ─────────────────────────────────────────────────────────
_TASK_KEYWORDS = {
    "flood": "flood damage assessment",
    "earthquake": "earthquake damage assessment",
    "fire": "wildfire damage assessment",
    "cyclone": "cyclone damage assessment",
    "hurricane": "hurricane damage assessment",
    "landslide": "landslide damage assessment",
    "tsunami": "tsunami damage assessment",
    "damage": "damage assessment",
    "change": "change detection",
}


def parse_query(text: str) -> Dict[str, Optional[str]]:
    """Parse a natural language disaster query into a structured dict.

    Args:
        text: Free-form query string.

    Returns:
        Dict with keys: ``location``, ``start_date``, ``end_date``, ``task``.
    """
    text_lower = text.lower()

    # ── Extract task type ────────────────────────────────────────────
    task = "change detection"
    for kw, label in _TASK_KEYWORDS.items():
        if kw in text_lower:
            task = label
            break

    # ── Extract year ─────────────────────────────────────────────────
    year_match = re.search(r"\b(20\d{2})\b", text)
    year = int(year_match.group(1)) if year_match else datetime.now().year

    # ── Extract month ────────────────────────────────────────────────
    month_num = None
    for name, num in _MONTHS.items():
        if name in text_lower:
            month_num = num
            break

    if month_num:
        import calendar
        _, last_day = calendar.monthrange(year, month_num)
        start_date = f"{year}-{month_num:02d}-01"
        end_date   = f"{year}-{month_num:02d}-{last_day:02d}"
    else:
        start_date = f"{year}-01-01"
        end_date   = f"{year}-12-31"

    # ── Extract location using spaCy (fallback to regex) ─────────────
    location = _extract_location(text)

    return {
        "location":   location,
        "start_date": start_date,
        "end_date":   end_date,
        "task":       task,
    }


def _extract_location(text: str) -> Optional[str]:
    """Extract location mention from text using spaCy GPE/LOC entities,
    falling back to a simple 'in <Place>' regex pattern."""
    # 1. Try spaCy
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC"):
                return ent.text
    except Exception:
        pass

    # 2. Regex fallback: "in LOCATION"
    m = re.search(r"\bin\s+([A-Z][a-zA-Z\s,]+?)(?:\s+\d{4}|\s*$)", text)
    if m:
        return m.group(1).strip().rstrip(",")

    return None
