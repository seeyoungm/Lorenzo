from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"


class InputType(str, Enum):
    GOAL_STATEMENT = "goal_statement"
    QUESTION = "question"
    MEMORY_RECALL = "memory_recall"
    MEMORY_CANDIDATE = "memory_candidate"
    MEMORY_INSTRUCTION = "memory_instruction"
    FACT = "fact"
    EVENT = "event"
    PREFERENCE = "preference"
    COMMITMENT = "commitment"
    STATEMENT = "statement"


@dataclass(slots=True)
class MemoryItem:
    content: str
    memory_type: MemoryType
    importance: float = 5.0
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    memory_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "importance": self.importance,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MemoryItem":
        return cls(
            memory_id=raw["memory_id"],
            content=raw["content"],
            memory_type=MemoryType(raw["memory_type"]),
            importance=float(raw.get("importance", 5.0)),
            metadata=dict(raw.get("metadata", {})),
            tags=list(raw.get("tags", [])),
            created_at=datetime.fromisoformat(raw["created_at"]),
        )


@dataclass(slots=True)
class ProcessedInput:
    raw_text: str
    input_type: InputType
    input_types: list[InputType] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    goal_confidence: str = "none"
    preference_confidence: str = "none"
    fact_confidence: str = "none"


@dataclass(slots=True)
class RetrievedMemory:
    memory: MemoryItem
    total_score: float
    similarity_score: float
    recency_score: float
    importance_score: float
    retrieval_reason: str = ""


@dataclass(slots=True)
class ReasoningPlan:
    strategy: str
    rationale: str
    steps: list[str]


@dataclass(slots=True)
class TurnResult:
    user_input: str
    response: str
    retrieved_memories: list[RetrievedMemory]
    plan: ReasoningPlan
