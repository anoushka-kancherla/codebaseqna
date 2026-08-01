from __future__ import annotations
from dataclasses import dataclass
import json
import re


@dataclass
class ParsedResponse:
    thinking: str
    json_header: dict
    prose: str
    tool_results: list[dict]
    usage: dict


def split_json_header(text: str) -> tuple[dict, str]:
    # Try code-fenced first
    match = re.match(r"^\s*```(?:json)?\s*(\{.*?\})\s*```(.*)", text, re.DOTALL)
    if not match:
        # Try raw JSON
        match = re.match(r"^\s*(\{.*?\})(.*)", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1)), match.group(2).strip()
        except json.JSONDecodeError:
            pass

    # Fallback: despite the system prompt asking for the JSON block first,
    # a real run showed the model narrating its tool use in prose before
    # emitting the header at the end instead. Look for a fenced JSON block
    # anywhere and treat the rest of the text as prose.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            header = json.loads(fenced.group(1))
            prose = (text[:fenced.start()] + text[fenced.end():]).strip()
            return header, prose
        except json.JSONDecodeError:
            pass

    # Fallback: a raw (unfenced) JSON object dropped directly into running
    # prose with no delimiter at all, observed in another real run. Scan flat
    # (non-nested) {...} spans and take the first one that both parses and
    # carries a key our header schema actually uses, so an unrelated brace
    # elsewhere in the prose can't be mistaken for the header.
    for raw_match in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            candidate = json.loads(raw_match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and ("confidence" in candidate or "result" in candidate):
            prose = (text[:raw_match.start()] + text[raw_match.end():]).strip()
            return candidate, prose

    return {}, text.strip()


def parse_stream(stream) -> ParsedResponse:
    thinking_parts, text_parts, tool_results = [], [], []
    current_block = None
    usage = {}
    for event in stream:
        if event.type == "content_block_start":
            current_block = event.content_block.type
        elif event.type == "content_block_delta":
            # Adaptive thinking blocks interleave a signature_delta (no .thinking
            # attribute, used for multi-turn continuity) alongside thinking_delta.
            if current_block == "thinking" and hasattr(event.delta, "thinking"):
                thinking_parts.append(event.delta.thinking)
            elif current_block == "text" and hasattr(event.delta, "text"):
                text_parts.append(event.delta.text)
        elif event.type == "message_delta":
            # usage is a pydantic model with nested pydantic sub-objects (e.g.
            # output_tokens_details) — vars() leaves those un-flattened and
            # unserializable, so prefer model_dump() when it's available.
            usage = event.usage.model_dump() if hasattr(event.usage, "model_dump") else vars(event.usage)
    full_text = "".join(text_parts)
    json_header, prose = split_json_header(full_text)
    return ParsedResponse(
        thinking="".join(thinking_parts),
        json_header=json_header, prose=prose,
        tool_results=tool_results, usage=usage
    )
