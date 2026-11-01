"""Tests for conservative text preprocessing and normalization."""

from app.services.preprocessing import normalize_text


def test_normalize_line_endings():
    raw = "Line 1\r\nLine 2\rLine 3\nLine 4"
    result = normalize_text(raw)
    assert "\r" not in result
    assert result == "Line 1\nLine 2\nLine 3\nLine 4"


def test_normalize_broken_pdf_hyphens():
    raw = "The deep learning transfor-\nmation has accelerated research."
    result = normalize_text(raw)
    assert "transformation" in result
    assert "transfor-\nmation" not in result


def test_normalize_ligatures():
    raw = "The ﬁrst ﬂight required a strong coeﬃcient."
    result = normalize_text(raw)
    assert "first" in result
    assert "flight" in result
    assert "coefficient" in result


def test_normalize_excessive_whitespace():
    raw = "Word1    Word2\t\tWord3\n\n\n\nWord4"
    result = normalize_text(raw)
    assert "Word1 Word2 Word3" in result
    assert "\n\nWord4" in result
    assert "\n\n\n" not in result


def test_preserves_punctuation_and_numbers():
    raw = "On 2026-11-01, revenue was $14,500.50 (+12.4%). Formula: E=mc^2!"
    result = normalize_text(raw)
    assert result == "On 2026-11-01, revenue was $14,500.50 (+12.4%). Formula: E=mc^2!"
