from app.ingestion.cleaner import clean_text


def test_clean_text_removes_null_bytes():
    assert "\x00" not in clean_text("hello\x00world")


def test_clean_text_collapses_repeated_spaces():
    assert clean_text("hello    world") == "hello world"


def test_clean_text_collapses_excess_blank_lines():
    assert clean_text("line1\n\n\n\nline2") == "line1\n\nline2"


def test_clean_text_handles_empty_and_none():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_clean_text_strips_leading_trailing_whitespace():
    assert clean_text("   hello world   ") == "hello world"