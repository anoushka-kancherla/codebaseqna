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


def test_split_json_header_fenced_at_end_falls_back():
    # Observed in a real run: despite the system prompt, the model narrated
    # its tool use in prose first and put the JSON header at the end instead
    # of the start. The anchored patterns miss this; the fallback shouldn't.
    prose_text = "I'll investigate the codebase first.\n\nHere's what I found."
    text = prose_text + "\n\n```json\n" + __import__("json").dumps(HEADER) + "\n```"
    header, prose = split_json_header(text)
    assert header == HEADER
    assert prose == prose_text


def test_split_json_header_raw_mid_prose_falls_back():
    # Observed in another real run: no fences at all, the raw {...} object
    # dropped directly into the middle of running prose.
    before = "Some narration before the header."
    after = "## More prose after the header."
    text = before + __import__("json").dumps(HEADER) + after
    header, prose = split_json_header(text)
    assert header == HEADER
    assert prose == (before + after)


def test_split_json_header_ignores_unrelated_braces_in_prose():
    text = 'Some prose with an unrelated {"foo": "bar"} dict, no real header here.'
    header, prose = split_json_header(text)
    assert header == {}
    assert prose == text


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
    assert result.raw_text == '{"confidence": "high"} The answer is here.'
    assert result.usage == {"output_tokens": 42, "input_tokens": 10}
    assert result.tool_results == []


def test_parse_stream_captures_thinking_signature_for_multi_turn_replay():
    # Adaptive thinking mode interleaves a signature_delta (no .thinking attr)
    # into the thinking content block; it must be captured (not dropped) since
    # multi-turn continuation needs to replay the thinking block verbatim,
    # signature included.
    events = [
        _event("content_block_start", content_block=SimpleNamespace(type="thinking")),
        _event("content_block_delta", delta=SimpleNamespace(thinking="pondering...")),
        _event("content_block_delta", delta=SimpleNamespace(signature="sig-xyz")),
        _event("content_block_start", content_block=SimpleNamespace(type="text")),
        _event("content_block_delta", delta=SimpleNamespace(text="The answer is here.")),
    ]
    result = parse_stream(events)
    assert result.thinking == "pondering..."
    assert result.thinking_signature == "sig-xyz"
    assert result.prose == "The answer is here."


def test_parse_stream_flattens_pydantic_usage():
    import json as json_mod

    class _FakeUsage:
        def model_dump(self):
            return {"output_tokens": 12, "output_tokens_details": {"nested": 1}}

    events = [
        _event("content_block_start", content_block=SimpleNamespace(type="text")),
        _event("content_block_delta", delta=SimpleNamespace(text="hi")),
        _event("message_delta", usage=_FakeUsage()),
    ]
    result = parse_stream(events)
    json_mod.dumps(result.usage)  # must not raise
    assert result.usage == {"output_tokens": 12, "output_tokens_details": {"nested": 1}}
