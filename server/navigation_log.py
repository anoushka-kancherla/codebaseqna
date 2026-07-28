from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import time
import uuid


@dataclass
class NavigationEntry:
    timestamp: float
    event: Literal["file_read", "tree_read", "tool_call"]
    path: str | None = None
    tool_name: str | None = None
    tool_args: dict | None = None
    lines: int | None = None
    size_bytes: int | None = None


class NavigationLog:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.entries: list[NavigationEntry] = []

    def record_file_read(self, path: str, content: str) -> None:
        self.entries.append(NavigationEntry(
            timestamp=time.time(), event="file_read", path=path,
            lines=content.count("\n") + 1,
            size_bytes=len(content.encode("utf-8"))
        ))

    def record_tree_read(self) -> None:
        self.entries.append(NavigationEntry(timestamp=time.time(), event="tree_read"))

    def record_tool_call(self, tool_name: str, args: dict) -> None:
        self.entries.append(NavigationEntry(
            timestamp=time.time(), event="tool_call",
            tool_name=tool_name, tool_args=args
        ))

    def to_dict(self) -> list[dict]:
        return [vars(e) for e in self.entries]
