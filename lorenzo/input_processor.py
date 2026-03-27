from __future__ import annotations

import re

from lorenzo.models import InputType, ProcessedInput


class InputProcessor:
    _entity_re = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
    _time_like_re = re.compile(r"\b\d{1,2}(:\d{2})?\s*(am|pm)?\b|\d{1,2}\s*시")

    _goal_keywords = ["want", "goal", "하고 싶", "만들고 싶", "목표"]
    _memory_keywords = ["기억", "저장", "잊지마", "remember", "save"]
    _short_temporal_markers = ["오늘", "내일", "모레", "다음", "later", "tomorrow"]
    _short_schedule_markers = ["일정", "약속", "알림", "리마인드", "remind"]
    _event_keywords = ["meeting", "deadline", "appointment", "event", "회의", "마감", "일정", "약속"]
    _fact_keywords = ["사실", "중요", "important", "항상", "never", "절대"]
    _preference_keywords = ["prefer", "preference", "선호", "좋아", "싫어"]
    _commitment_keywords = ["i will", "commit", "약속", "반드시", "할게", "하겠다"]
    _memory_recall_keywords = [
        "recall",
        "what did i",
        "remember what",
        "기억나",
        "기억해",
        "뭐였지",
        "였지",
        "이전에",
    ]

    def process(self, text: str) -> ProcessedInput:
        normalized = text.strip()
        lower = normalized.lower()

        labels: list[InputType] = []

        has_goal = self._contains_any(lower, self._goal_keywords)
        has_memory_trigger = self._contains_any(lower, self._memory_keywords)
        is_question = normalized.endswith("?")
        has_memory_recall = is_question and self._contains_any(lower, self._memory_recall_keywords)
        is_short_memory_candidate = self._is_short_memory_candidate(normalized, lower, is_question)

        # Memory intent comes before question intent.
        if has_goal:
            labels.append(InputType.GOAL_STATEMENT)

        if self._contains_any(lower, self._preference_keywords):
            labels.append(InputType.PREFERENCE)

        if self._contains_any(lower, self._commitment_keywords):
            labels.append(InputType.COMMITMENT)

        if self._contains_any(lower, self._event_keywords):
            labels.append(InputType.EVENT)

        if self._is_fact_like(normalized, lower):
            labels.append(InputType.FACT)

        if has_memory_trigger:
            labels.append(InputType.MEMORY_INSTRUCTION)
            labels.append(InputType.MEMORY_CANDIDATE)
        elif is_short_memory_candidate:
            labels.append(InputType.MEMORY_CANDIDATE)
            labels.append(InputType.EVENT)

        if has_memory_recall:
            labels.append(InputType.MEMORY_RECALL)

        if is_question:
            labels.append(InputType.QUESTION)

        if not labels:
            labels.append(InputType.STATEMENT)

        labels = self._dedup_keep_order(labels)
        primary = self._select_primary(labels)

        entities = self._extract_entities(normalized)
        return ProcessedInput(
            raw_text=normalized,
            input_type=primary,
            input_types=labels,
            entities=entities,
        )

    def _contains_any(self, text: str, keywords: list[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    def _is_short_memory_candidate(self, text: str, lower: str, is_question: bool) -> bool:
        if not is_question:
            return False

        compact = "".join(text.split())
        if len(compact) > 20:
            return False

        has_time_hint = bool(self._time_like_re.search(lower))
        has_temporal_marker = self._contains_any(lower, self._short_temporal_markers)
        has_schedule_marker = self._contains_any(lower, self._short_schedule_markers)
        return has_time_hint or (has_temporal_marker and has_schedule_marker)

    def _is_fact_like(self, text: str, lower: str) -> bool:
        if self._contains_any(lower, self._fact_keywords):
            return True
        has_number = bool(re.search(r"\d", text))
        has_copula = any(token in lower for token in [" is ", " are ", "이다", "입니다", "였다", "was", "were"])
        return has_number and has_copula

    def _dedup_keep_order(self, labels: list[InputType]) -> list[InputType]:
        seen: set[InputType] = set()
        deduped: list[InputType] = []
        for label in labels:
            if label in seen:
                continue
            seen.add(label)
            deduped.append(label)
        return deduped

    def _select_primary(self, labels: list[InputType]) -> InputType:
        priority = [
            InputType.GOAL_STATEMENT,
            InputType.PREFERENCE,
            InputType.COMMITMENT,
            InputType.FACT,
            InputType.EVENT,
            InputType.MEMORY_CANDIDATE,
            InputType.MEMORY_RECALL,
            InputType.MEMORY_INSTRUCTION,
            InputType.QUESTION,
            InputType.STATEMENT,
        ]
        for item in priority:
            if item in labels:
                return item
        return InputType.STATEMENT

    def _extract_entities(self, text: str) -> list[str]:
        return sorted(set(self._entity_re.findall(text)))
