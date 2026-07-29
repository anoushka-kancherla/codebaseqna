from types import SimpleNamespace

from api.stream_parser import parse_stream, split_json_header

HEADER = {
    "confidence": "high",
    "confidence_reason": "clear evidence",
    "uncertainty_type": "none",
    "files_read": ["a.py"],
    "files_read_reasons": ["entry point"],
    "call_graph_complete": True,
    "caveats": "",
}


def test_split_json_header_fenced():
    text = "```json\n" + __import__("json").dumps(HEADER) + "\n```\nHere is the prose answer."
    header, prose = split_json_header(text)
    assert header == HEADER
    assert prose == "Here is the prose answer."


def test_split_json_header_raw():
    text = __import__("json").dumps(HEADER) + "\nHere is the prose answer."
    header, prose = split_json_header(text)
    assert header == HEADER
    assert prose == "Here is the prose answer."


def test_split_json_header_missing_returns_empty():
    header, prose = split_json_header("Just prose, no header.")
    assert header == {}
    assert prose == "Just prose, no header."


def _event(type_, **kwargs):
    return SimpleNamespace(type=type_, **kwargs)


def test_parse_stream_separates_thinking_and_text():
    events = [
        _event("content_block_start", content_block=SimpleNamespace(type="thinking")),
        _event("content_block_delta", delta=SimpleNamespace(thinking="pondering...")),
        _event("content_block_start", content_block=SimpleNamespace(type="text")),
        _event("content_block_delta", delta=SimpleNamespace(text='{"confidence": "high"} The answer is here.')),
        _event("message_delta", usage=SimpleNamespace(output_tokens=42, input_tokens=10)),
    ]
    result = parse_stream(events)
    assert result.thinking == "pondering..."
    assert result.json_header == {"confidence": "high"}
    assert result.prose == "The answer is here."
    assert result.usage == {"output_tokens": 42, "input_tokens": 10}
    assert result.tool_results == []
