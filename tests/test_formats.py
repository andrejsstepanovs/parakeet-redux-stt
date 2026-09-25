from stt2_service.routes import (
    _clean_text,
    _result_to_segments,
    _segments_to_srt,
    _segments_to_vtt,
)

RESULT = {
    "text": "Hello world. Second sentence.",
    "segments": [
        {
            "start": 0.0,
            "end": 1.5,
            "text": "Hello world.",
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.7},
                {"word": "world.", "start": 0.7, "end": 1.5},
            ],
        },
        {
            "start": 1.5,
            "end": 3.0,
            "text": "Second sentence.",
            "words": [],
        },
    ],
}


def test_clean_text_collapses_spacing_and_underscore():
    assert _clean_text("  hello\u2581world   ") == "hello world"
    assert _clean_text("don 't") == "don't"


def test_result_to_segments_maps_words():
    segments = _result_to_segments(RESULT)
    assert len(segments) == 2
    assert segments[0]["segment"] == "Hello world."
    assert segments[0]["words"][0]["word"] == "Hello"
    assert segments[1]["words"] == []


def test_result_to_segments_skips_empty():
    assert _result_to_segments({"segments": [{"start": 0, "end": 1, "text": "  "}]}) == []


def test_srt_formatting():
    srt = _segments_to_srt(_result_to_segments(RESULT))
    assert "1\n00:00:00,000 --> 00:00:01,500\nHello world." in srt
    assert "2\n00:00:01,500 --> 00:00:03,000\nSecond sentence." in srt


def test_vtt_formatting():
    vtt = _segments_to_vtt(_result_to_segments(RESULT))
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in vtt
