from __future__ import annotations

from typing import Protocol

from lorenzo.models import MemoryItem, ProcessedInput, ReasoningPlan, RetrievedMemory


class InputProcessorPort(Protocol):
    def process(self, text: str) -> ProcessedInput:
        ...


class MemoryStorePort(Protocol):
    def add(self, item: MemoryItem) -> None:
        ...

    def list_all(self) -> list[MemoryItem]:
        ...

    def replace_all(self, items: list[MemoryItem]) -> None:
        ...

    def count(self) -> int:
        ...


class MemoryRetrieverPort(Protocol):
    def retrieve(
        self,
        query: str,
        memories: list[MemoryItem],
        top_k: int = 5,
        now=None,
        mode: str = "auto",
    ) -> list[RetrievedMemory]:
        ...

    def text_similarity(self, left: str, right: str) -> float:
        ...


class ReasoningPlannerPort(Protocol):
    def plan(self, processed: ProcessedInput, retrieved: list[RetrievedMemory]) -> ReasoningPlan:
        ...


class LanguageAdapterPort(Protocol):
    def generate(
        self,
        user_input: str,
        processed: ProcessedInput,
        retrieved: list[RetrievedMemory],
        plan: ReasoningPlan,
    ) -> str:
        ...
