from __future__ import annotations

import json
from pathlib import Path

from lorenzo.models import MemoryItem


class JsonlMemoryStore:
    """Persistent memory store backed by JSONL."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def add(self, item: MemoryItem) -> None:
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    def list_all(self) -> list[MemoryItem]:
        items: list[MemoryItem] = []
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    items.append(MemoryItem.from_dict(raw))
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Skip malformed rows and keep the store readable.
                    continue
        return items

    def count(self) -> int:
        return len(self.list_all())

    def replace_all(self, items: list[MemoryItem]) -> None:
        with self.path.open("w", encoding="utf-8") as fp:
            for item in items:
                fp.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
