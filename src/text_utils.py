import re
from typing import Optional

TIME_PATTERNS = [
    r"(\d+)\s*-\s*(\d+)\s*(?:hours|hrs|h)\b",
    r"(\d+)\s*(?:hours|hrs|h)\b",
    r"(\d+)\s*-\s*(\d+)\s*(?:mins|minutes|m)\b",
    r"(\d+)\s*(?:mins|minutes|m)\b",
]

def extract_time_budget_mins(text: str) -> Optional[int]:
    """
    Parse an approximate time budget in minutes from free text.
    Returns the upper bound if a range is given. Examples:
      '1-2 hour' -> 120, '≤ 40 mins' -> 40, 'at most 90 mins' -> 90
    """
    if not text:
        return None
    t = text.lower()
    # explicit upper bounds like 'at most 90 mins'
    m = re.search(r"(?:at most|atmost|<=|less than|no more than|under|within)\s*(\d+)\s*(mins|minutes|m|hours|hrs|h)", t)
    if m:
        val = int(m.group(1))
        unit = m.group(2)
        return val * 60 if unit.startswith("h") else val
    # ranges
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*(hours|hrs|h)", t)
    if m:
        return int(m.group(2)) * 60
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*(mins|minutes|m)", t)
    if m:
        return int(m.group(2))
    # singles
    m = re.search(r"(\d+)\s*(hours|hrs|h)", t)
    if m:
        return int(m.group(1)) * 60
    m = re.search(r"(\d+)\s*(mins|minutes|m)", t)
    if m:
        return int(m.group(1))
    # common phrases
    if "1-2 hour" in t or "1 to 2 hour" in t:
        return 120
    if "half an hour" in t:
        return 30
    return None


