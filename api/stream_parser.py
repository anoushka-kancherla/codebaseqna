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
    return {}, text.strip()


def parse_stream(stream) -> ParsedResponse:
    thinking_parts, text_parts, tool_results = [], [], []
    current_block = None
    usage = {}
    for event in stream:
        if event.type == "content_block_start":
            current_block = event.content_block.type
        elif event.type == "content_block_delta":
            if current_block == "thinking":
                thinking_parts.append(event.delta.thinking)
            elif current_block == "text":
                text_parts.append(event.delta.text)
        elif event.type == "message_delta":
            usage = vars(event.usage)
    full_text = "".join(text_parts)
    json_header, prose = split_json_header(full_text)
    return ParsedResponse(
        thinking="".join(thinking_parts),
        json_header=json_header, prose=prose,
        tool_results=tool_results, usage=usage
    )
