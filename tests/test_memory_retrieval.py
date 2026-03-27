from datetime import datetime, timedelta, timezone

from lorenzo.memory.retriever import MemoryRetriever
from lorenzo.models import MemoryItem, MemoryType


def test_retrieval_prioritizes_similarity_when_query_matches() -> None:
    now = datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc)
    retriever = MemoryRetriever(
        importance_weight=0.30,
        recency_weight=0.25,
        similarity_weight=0.45,
    )

    memories = [
        MemoryItem(
            content="나는 장기 기억이 있는 AI를 만들고 싶어",
            memory_type=MemoryType.SEMANTIC,
            importance=7.0,
            created_at=now - timedelta(hours=2),
        ),
        MemoryItem(
            content="오늘 점심 메뉴를 고민하고 있어",
            memory_type=MemoryType.EPISODIC,
            importance=10.0,
            created_at=now - timedelta(minutes=10),
        ),
    ]

    ranked = retriever.retrieve("장기 기억 있는 AI", memories, top_k=2, now=now)

    assert len(ranked) == 2
    assert ranked[0].memory.content == "나는 장기 기억이 있는 AI를 만들고 싶어"
    assert ranked[0].similarity_score > ranked[1].similarity_score


def test_retrieval_includes_importance_and_recency_when_similarity_equal() -> None:
    now = datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc)
    retriever = MemoryRetriever(
        importance_weight=0.40,
        recency_weight=0.40,
        similarity_weight=0.20,
    )

    memories = [
        MemoryItem(
            content="완전히 다른 문장 A",
            memory_type=MemoryType.EPISODIC,
            importance=2.0,
            created_at=now - timedelta(days=7),
        ),
        MemoryItem(
            content="완전히 다른 문장 B",
            memory_type=MemoryType.EPISODIC,
            importance=9.0,
            created_at=now - timedelta(hours=1),
        ),
    ]

    ranked = retriever.retrieve("쿼리와 무관한 내용", memories, top_k=2, now=now)

    assert ranked[0].memory.content == "완전히 다른 문장 B"
    assert ranked[0].importance_score > ranked[1].importance_score
    assert ranked[0].recency_score > ranked[1].recency_score
