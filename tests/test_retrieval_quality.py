from datetime import datetime, timezone

from lorenzo.memory.retriever import MemoryRetriever
from lorenzo.models import MemoryItem, MemoryType


def test_goal_intent_reranking_prefers_semantic_when_base_equal() -> None:
    now = datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc)
    retriever = MemoryRetriever()

    semantic = MemoryItem(
        content="build modular ai memory goal",
        memory_type=MemoryType.SEMANTIC,
        importance=7.0,
        created_at=now,
    )
    episodic = MemoryItem(
        content="build modular ai memory goal",
        memory_type=MemoryType.EPISODIC,
        importance=7.0,
        created_at=now,
    )

    ranked = retriever.retrieve("목표를 갖는 modular ai를 만들고 싶어", [episodic, semantic], top_k=2, now=now)

    assert ranked[0].memory.memory_type is MemoryType.SEMANTIC
    assert "type_alignment" in ranked[0].retrieval_reason
    assert "intent=goal" in ranked[0].retrieval_reason
    assert "semantic(raw=" in ranked[0].retrieval_reason


def test_multilingual_semantic_similarity_dominates_over_keyword_overlap() -> None:
    now = datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc)
    retriever = MemoryRetriever()

    target = MemoryItem(
        content="User goal: build a long-term memory AI architecture",
        memory_type=MemoryType.SEMANTIC,
        importance=8.0,
        created_at=now,
    )
    misleading = MemoryItem(
        content="menu memory module keyword trap",
        memory_type=MemoryType.SEMANTIC,
        importance=8.0,
        created_at=now,
    )

    ranked = retriever.retrieve("장기 기억 AI 목표 구조", [misleading, target], top_k=2, now=now)

    assert ranked[0].memory.content == target.content


def test_embedding_cache_is_reused_for_memory_items() -> None:
    now = datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc)
    retriever = MemoryRetriever()

    memories = [
        MemoryItem(content="alpha memory", memory_type=MemoryType.SEMANTIC, created_at=now),
        MemoryItem(content="beta memory", memory_type=MemoryType.SEMANTIC, created_at=now),
    ]

    retriever.retrieve("alpha", memories, top_k=2, now=now)
    cache_size_first = len(retriever._memory_embedding_cache)

    retriever.retrieve("alpha again", memories, top_k=2, now=now)
    cache_size_second = len(retriever._memory_embedding_cache)

    assert cache_size_first == 2
    assert cache_size_second == 2


def test_preference_query_prefers_preference_tagged_semantic_memory() -> None:
    now = datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc)
    retriever = MemoryRetriever()

    preference = MemoryItem(
        content="User preference: concise bullet style",
        memory_type=MemoryType.SEMANTIC,
        tags=["preference", "semantic"],
        importance=7.0,
        created_at=now,
    )
    generic_goal = MemoryItem(
        content="User goal: build memory system",
        memory_type=MemoryType.SEMANTIC,
        tags=["goal", "semantic"],
        importance=7.0,
        created_at=now,
    )

    ranked = retriever.retrieve("내 선호 답변 스타일이 뭐야?", [generic_goal, preference], top_k=2, now=now)

    assert ranked[0].memory is preference
    assert "intent=preference" in ranked[0].retrieval_reason


def test_conflict_penalty_demotes_stale_fact_memory() -> None:
    now = datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc)
    retriever = MemoryRetriever()

    stale = MemoryItem(
        content="User fact: 프로젝트 예산은 50만원이다",
        memory_type=MemoryType.SEMANTIC,
        tags=["fact", "semantic"],
        importance=8.0,
        created_at=now,
    )
    latest = MemoryItem(
        content="User fact: 프로젝트 예산은 80만원이다",
        memory_type=MemoryType.SEMANTIC,
        tags=["fact", "semantic"],
        importance=8.0,
        created_at=now,
    )

    ranked = retriever.retrieve("현재 프로젝트 예산이 얼마야?", [stale, latest], top_k=2, now=now)

    assert ranked[0].memory is latest
    assert "conflict_penalty" in ranked[1].retrieval_reason
