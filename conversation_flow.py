"""Parsing helpers for the structured, one-question-at-a-time patient-detail
chat flow. Deliberately simple regex/keyword parsing (not LLM-based extraction)
so behavior is predictable and debuggable live during a demo.
"""
import re


def parse_age(text: str):
    match = re.search(r"\d{1,3}", text)
    if not match:
        return None
    age = int(match.group())
    return age if 1 <= age <= 120 else None


def parse_sex(text: str):
    t = text.strip().lower()
    if "female" in t or t in ("f", "woman"):
        return "Female"
    if "male" in t or t in ("m", "man"):
        return "Male"
    return None


def parse_cognitive_score(text: str):
    """Returns (value, was_skipped). value is None if skipped or unparseable."""
    t = text.strip().lower()
    if any(word in t for word in ("skip", "none", "n/a", "no", "don't", "dont", "unknown")):
        return None, True
    match = re.search(r"\d+(\.\d+)?", text)
    if match:
        return float(match.group()), False
    return None, False
