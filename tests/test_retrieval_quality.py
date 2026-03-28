from datetime import datetime, timedelta, timezone

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


def test_strong_only_mode_excludes_weak_memory_hints() -> None:
    now = datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc)
    retriever = MemoryRetriever()

    strong = MemoryItem(
        content="User fact: 팀 미팅은 금요일이다",
        memory_type=MemoryType.SEMANTIC,
        tags=["fact", "semantic", "strong"],
        metadata={"memory_tier": "strong"},
        created_at=now,
    )
    weak = MemoryItem(
        content="Weak goal hint: 장기 기억 AI를 만들고 싶어",
        memory_type=MemoryType.SEMANTIC,
        tags=["goal", "semantic", "weak", "weak_hint"],
        metadata={"memory_tier": "weak"},
        created_at=now,
    )

    ranked = retriever.retrieve("장기 기억 AI 목표", [strong, weak], top_k=2, now=now, mode="strong_only")

    assert ranked
    assert ranked[0].memory is strong
    assert all(item.memory is not weak for item in ranked)


def test_auto_mode_expands_to_weak_memory_on_low_confidence() -> None:
    now = datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc)
    retriever = MemoryRetriever(
        fallback_similarity_threshold=0.65,
        fallback_score_threshold=0.40,
    )

    strong = MemoryItem(
        content="User fact: 회의실은 3층이다",
        memory_type=MemoryType.SEMANTIC,
        tags=["fact", "semantic", "strong"],
        metadata={"memory_tier": "strong"},
        created_at=now,
    )
    weak = MemoryItem(
        content="Weak preference hint: 가능하면 짧은 답변을 선호할 듯해",
        memory_type=MemoryType.SEMANTIC,
        tags=["preference", "semantic", "weak", "weak_hint"],
        metadata={"memory_tier": "weak"},
        created_at=now,
    )

    ranked = retriever.retrieve("내 답변 선호가 뭐였지?", [strong, weak], top_k=2, now=now, mode="auto")

    assert ranked
    assert ranked[0].memory is weak
    assert "mode=with_fallback" in ranked[0].retrieval_reason
    assert "recall_mode=true" in ranked[0].retrieval_reason


def test_intent_aware_fallback_prefers_weak_preference_for_preference_recall() -> None:
    now = datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc)
    retriever = MemoryRetriever()

    strong = MemoryItem(
        content="User fact: 회의실은 3층이다",
        memory_type=MemoryType.SEMANTIC,
        tags=["fact", "semantic", "strong"],
        metadata={"memory_tier": "strong"},
        created_at=now,
    )
    weak_preference = MemoryItem(
        content="Weak preference hint: 답변은 짧은 bullet 형식이 좋다",
        memory_type=MemoryType.SEMANTIC,
        tags=["preference", "semantic", "weak", "weak_hint", "weak_preference_hint"],
        metadata={"memory_tier": "weak"},
        created_at=now,
    )
    weak_fact = MemoryItem(
        content="Weak fact hint: 예산은 120 정도",
        memory_type=MemoryType.SEMANTIC,
        tags=["fact", "semantic", "weak", "weak_hint", "weak_fact_hint"],
        metadata={"memory_tier": "weak"},
        created_at=now,
    )

    ranked = retriever.retrieve(
        "내 선호가 뭐였지?",
        [strong, weak_fact, weak_preference],
        top_k=3,
        now=now,
        mode="auto",
    )

    assert ranked
    assert ranked[0].memory is weak_preference
    assert "intent=preference_recall" in ranked[0].retrieval_reason


def test_intent_aware_fallback_prefers_weak_fact_for_fact_recall() -> None:
    now = datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc)
    retriever = MemoryRetriever()

    strong = MemoryItem(
        content="User commitment: 나는 내일까지 문서를 제출하겠다",
        memory_type=MemoryType.SEMANTIC,
        tags=["commitment", "semantic", "strong"],
        metadata={"memory_tier": "strong"},
        created_at=now,
    )
    weak_goal = MemoryItem(
        content="Weak goal hint: 장기 기억 AI를 만들고 싶다",
        memory_type=MemoryType.SEMANTIC,
        tags=["goal", "semantic", "weak", "weak_hint"],
        metadata={"memory_tier": "weak"},
        created_at=now,
    )
    weak_fact = MemoryItem(
        content="Weak fact hint: latest budget is around 130",
        memory_type=MemoryType.SEMANTIC,
        tags=["fact", "semantic", "weak", "weak_hint", "weak_fact_hint"],
        metadata={"memory_tier": "weak"},
        created_at=now,
    )

    ranked = retriever.retrieve(
        "What was my latest budget?",
        [strong, weak_goal, weak_fact],
        top_k=3,
        now=now,
        mode="auto",
    )

    assert ranked
    assert ranked[0].memory is weak_fact
    assert "intent=fact_recall" in ranked[0].retrieval_reason


def test_conditional_weak_override_promotes_weak_when_strong_is_insufficient() -> None:
    now = datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc)
    retriever = MemoryRetriever(
        weak_memory_fallback_penalty=0.80,
        weak_override_low_score_threshold=0.60,
        weak_override_high_similarity_threshold=0.45,
        weak_override_type_alignment_threshold=0.80,
    )

    strong = MemoryItem(
        content="User fact: 회의실은 3층이다",
        memory_type=MemoryType.SEMANTIC,
        tags=["fact", "semantic", "strong"],
        metadata={"memory_tier": "strong"},
        created_at=now,
    )
    weak = MemoryItem(
        content="Weak preference hint: 답변은 짧은 bullet 형식을 선호함",
        memory_type=MemoryType.SEMANTIC,
        tags=["preference", "semantic", "weak", "weak_hint"],
        metadata={"memory_tier": "weak"},
        created_at=now,
    )

    ranked = retriever.retrieve(
        "내 답변 선호가 뭐였지?",
        [strong, weak],
        top_k=2,
        now=now,
        mode="with_fallback",
    )

    assert ranked
    assert ranked[0].memory is weak
    assert (
        "weak_override=applied" in ranked[0].retrieval_reason
        or retriever.last_weak_override_success_count == 0
    )


def test_fact_recall_prefers_latest_value_across_multilingual_weak_hints() -> None:
    now = datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc)
    retriever = MemoryRetriever()

    older_en = MemoryItem(
        content="Weak fact hint: i think it was around 530 budget",
        memory_type=MemoryType.SEMANTIC,
        tags=["fact", "semantic", "weak", "weak_hint", "weak_fact_hint"],
        metadata={"memory_tier": "weak"},
        created_at=now,
    )
    latest_kr = MemoryItem(
        content="Weak fact hint: key=budget; value=550; source=예산은 아마 550쯤",
        memory_type=MemoryType.SEMANTIC,
        tags=["fact", "semantic", "weak", "weak_hint", "weak_fact_hint"],
        metadata={"memory_tier": "weak", "fact_key": "budget", "fact_value": "550"},
        created_at=now + timedelta(minutes=5),
    )

    ranked = retriever.retrieve(
        "what was my latest budget?",
        [older_en, latest_kr],
        top_k=2,
        now=now + timedelta(minutes=10),
        mode="with_fallback",
    )

    assert ranked
    assert ranked[0].memory is latest_kr
