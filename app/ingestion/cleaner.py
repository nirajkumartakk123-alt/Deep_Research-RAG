"""
Text cleaning/normalization applied after extraction, before chunking.

Deliberately conservative: strips null bytes and control characters,
collapses redundant whitespace, and trims lines - but does NOT do
anything "smart" like removing repeated headers/footers or de-hyphenating
line-wrapped words, since those are corpus-specific heuristics that can
silently destroy legitimate content. Add those later if a real document
set demonstrates the need, with a test proving they don't over-trigger.
"""
import re


def clean_text(text: str | None) -> str:
    if not text:
        return ""

    text = text.replace("\x00", "")
    text = re.sub(r"[\r\f\v]", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))

    return text.strip()