"""Conservative text normalization service for NeuroQuery."""

import re

# Common typographic ligatures produced during PDF text extraction
LIGATURE_MAP = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "--",
    "\xa0": " ",  # Non-breaking space
}

LIGATURE_REGEX = re.compile("|".join(re.escape(k) for k in LIGATURE_MAP.keys()))


def normalize_text(text: str) -> str:
    """
    Applies conservative text normalization preserving document semantics.

    Preserves:
    - Punctuation, symbols, mathematical notation, currency.
    - Numbers, dates, email addresses, URLs.
    - Word casing and grammatical structure.
    - Paragraph breaks and document flow.

    Fixes:
    - Inconsistent line endings (\r\n, \r -> \n).
    - Hard hyphenation split across lines (e.g. 'intelli-\ngence' -> 'intelligence').
    - Unicode typographic ligatures.
    - Multiple consecutive spaces/tabs within a line.
    - Excessive empty lines (>2 newlines collapsed to 2).
    """
    if not text:
        return ""

    # 1. Normalize line endings
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. De-ligaturize common PDF ligature characters
    normalized = LIGATURE_REGEX.sub(lambda m: LIGATURE_MAP[m.group(0)], normalized)

    # 3. Reconnect words broken by line-break hyphens: 'transfor-\nmation' -> 'transformation'
    # Only reconnect if both parts are alphabetic words
    normalized = re.sub(r"([A-Za-z]{2,})-\n([A-Za-z]{2,})", r"\1\2", normalized)

    # 4. Collapse horizontal whitespace (spaces, tabs) while preserving newlines
    lines = []
    for line in normalized.split("\n"):
        clean_line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(clean_line)

    normalized = "\n".join(lines)

    # 5. Collapse excessive vertical blank lines (>2 newlines collapsed to standard double newline for paragraphs)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    return normalized.strip()
