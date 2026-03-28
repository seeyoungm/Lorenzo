from __future__ import annotations

import re

from lorenzo.models import InputType, ProcessedInput


class InputProcessor:
    _entity_re = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
    _time_like_re = re.compile(r"\b\d{1,2}(:\d{2})?\s*(am|pm)?\b|\d{1,2}\s*시")
    _memory_command_re = re.compile(
        r"(기억해|기억해줘|기억해둬|저장해|저장해줘|잊지마|잊지 말아|remember|save|don't forget)"
    )

    _goal_keywords = [
        "goal",
        "objective",
        "target",
        "mission",
        "vision",
        "목표",
        "장기 목표",
        "궁극",
        "비전",
        "지향",
    ]
    _goal_transform_markers = [
        "만들",
        "구축",
        "완성",
        "달성",
        "개발",
        "개선",
        "향상",
        "높이",
        "확장",
        "전환",
        "build",
        "create",
        "design",
        "achieve",
        "transform",
        "improve",
        "optimize",
        "scale",
        "ship",
    ]
    _goal_future_markers = [
        "앞으로",
        "향후",
        "장기",
        "long-term",
        "future",
        "next",
        "roadmap",
        "올해",
        "내년",
        "계획",
        "plan",
        "intend",
        "aim to",
    ]
    _goal_persistence_markers = [
        "장기",
        "지속",
        "계속",
        "꾸준",
        "핵심",
        "전략",
        "궁극",
        "long-term",
        "roadmap",
        "persistent",
        "core",
    ]
    _goal_future_action_markers = [
        "할 거",
        "할 것이다",
        "하려고",
        "하려 한다",
        "하겠다",
        "할게",
        "will",
        "going to",
        "plan to",
        "intend to",
        "aim to",
    ]
    _goal_desire_markers = ["하고 싶", "고 싶", "싶어", "원해", "want to", "would like to"]
    _goal_wish_markers = ["좋겠다", "하면 좋겠", "wish", "would be nice", "바라"]
    _goal_opinion_markers = ["생각", "의견", "느낌", "같아", "i think", "in my opinion", "seems"]
    _goal_temporary_markers = ["오늘", "지금", "이번 주말", "잠깐", "당장", "today", "tonight", "right now"]
    _goal_vague_markers = ["해볼게", "생각해볼게", "고려해볼게", "maybe", "might", "could", "try to"]
    _goal_meta_markers = ["답변", "설명", "질문", "대화", "프롬프트", "respond", "answer style", "tone"]
    _memory_keywords = ["기억", "저장", "잊지마", "remember", "save"]
    _short_temporal_markers = ["오늘", "내일", "모레", "다음", "later", "tomorrow"]
    _short_schedule_markers = ["일정", "약속", "알림", "리마인드", "remind"]
    _event_keywords = ["meeting", "deadline", "appointment", "event", "회의", "마감", "일정", "약속"]
    _fact_keywords = ["사실", "중요", "important", "항상", "never", "절대"]
    _preference_keywords = ["prefer", "preference", "선호", "좋아해", "싫어해", "i like", "i dislike"]
    _preference_context_markers = ["답변", "스타일", "형식", "방식", "tone", "style", "format", "behavior"]
    _commitment_subject_markers = ["나는", "내가", "저는", "제가", "assistant", "ai", "에이전트"]
    _commitment_future_markers = ["will", "i'll", "going to", "할게", "하겠다", "하겠습니다", "할 예정", "예정이야", "하기로"]
    _commitment_action_markers = [
        "제출",
        "완료",
        "끝내",
        "보내",
        "정리",
        "수정",
        "업데이트",
        "배포",
        "작성",
        "send",
        "finish",
        "deliver",
        "submit",
        "update",
        "ship",
        "deploy",
    ]
    _commitment_certainty_markers = ["반드시", "꼭", "확실히", "will", "하겠다", "하겠습니다", "할게", "going to"]
    _commitment_time_markers = ["내일", "모레", "오늘", "tomorrow", "today", "이번 주", "next week", "까지", "by ", "before "]
    _commitment_promise_markers = ["약속", "promise", "commit", "맹세", "보장"]
    _commitment_vague_markers = ["해볼게", "생각해볼게", "시도해볼게", "고려해볼게", "검토해볼게", "해보려고", "try", "consider"]
    _commitment_conversational_markers = ["이렇게 답변할게", "답변할게", "답변해줄게", "설명해줄게", "말해줄게"]
    _commitment_speculative_markers = [
        "아마",
        "maybe",
        "might",
        "could",
        "possibly",
        "perhaps",
        "일지도",
        "할 수도",
        "인듯",
        "듯해",
    ]
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

        has_commitment = self._is_commitment_like(normalized, lower)
        has_preference = self._is_preference_like(normalized, lower, has_commitment=has_commitment)
        goal_confidence = self._goal_confidence(normalized, lower, has_commitment=has_commitment)
        has_goal = goal_confidence == "strong"
        has_memory_trigger = self._is_memory_command_like(normalized, lower)
        is_question = normalized.endswith("?")
        has_memory_recall = is_question and self._contains_any(lower, self._memory_recall_keywords)
        is_short_memory_candidate = self._is_short_memory_candidate(normalized, lower, is_question)

        # Memory intent comes before question intent.
        if has_commitment:
            labels.append(InputType.COMMITMENT)

        if has_goal:
            labels.append(InputType.GOAL_STATEMENT)

        if has_preference:
            labels.append(InputType.PREFERENCE)

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
            goal_confidence=goal_confidence,
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

    def _goal_confidence(self, text: str, lower: str, has_commitment: bool) -> str:
        if has_commitment:
            return "none"
        if text.endswith("?"):
            return "none"

        has_keyword = self._contains_any(lower, self._goal_keywords)
        has_transform = self._contains_any(lower, self._goal_transform_markers)
        has_future = self._contains_any(lower, self._goal_future_markers)
        has_future_action = self._contains_any(lower, self._goal_future_action_markers)
        has_persistence = self._contains_any(lower, self._goal_persistence_markers)
        has_desire = self._contains_any(lower, self._goal_desire_markers)
        goal_candidate = has_keyword or (has_desire and has_transform)

        negative_wish = self._contains_any(lower, self._goal_wish_markers)
        negative_opinion = self._contains_any(lower, self._goal_opinion_markers) and not has_keyword
        negative_temporary = self._contains_any(lower, self._goal_temporary_markers)
        negative_vague = self._contains_any(lower, self._goal_vague_markers)
        negative_meta = self._contains_any(lower, self._goal_meta_markers)
        has_negative = (
            negative_wish or negative_opinion or negative_temporary or negative_vague or negative_meta
        )

        # Strong goal: future + transformation intent + persistence, without negative cues.
        if (
            goal_candidate
            and (has_future or has_future_action or has_desire)
            and has_transform
            and has_persistence
            and not has_negative
        ):
            return "strong"

        if goal_candidate:
            return "weak"
        return "none"

    def _is_memory_command_like(self, text: str, lower: str) -> bool:
        if not self._memory_command_re.search(lower):
            return False

        compact = lower.strip()

        if "i remember" in compact or "기억이" in compact or "장기 기억" in compact:
            return False

        if compact.startswith(("기억", "저장", "remember", "save", "please remember", "please save")):
            return True

        if any(token in compact for token in ["기억해", "저장해", "잊지마", "don't forget"]):
            return True

        return compact.endswith("?")

    def _is_preference_like(self, text: str, lower: str, has_commitment: bool) -> bool:
        if has_commitment:
            return False
        if self._contains_any(lower, self._preference_keywords):
            return True
        has_context = self._contains_any(lower, self._preference_context_markers)
        has_tone_word = any(token in lower for token in ["좋아", "싫어", "원해", "want"])
        if has_context and has_tone_word and "할게" not in lower and "하겠다" not in lower:
            return True
        return False

    def _is_commitment_like(self, text: str, lower: str) -> bool:
        # Negative rules first: vague/ conversational/ speculative phrases are not commitments.
        if self._contains_any(lower, self._commitment_vague_markers):
            return False
        if self._contains_any(lower, self._commitment_speculative_markers):
            return False
        if self._contains_any(lower, self._commitment_conversational_markers):
            return False
        if "답변" in lower and any(token in lower for token in ["할게", "줄게", "해줄게"]):
            return False

        # Required conditions:
        # 1) explicit future action
        # 2) clear subject (self or agent)
        # 3) high certainty signal
        # 4) (time constraint OR explicit promise tone)
        has_subject = self._contains_any(lower, self._commitment_subject_markers) or bool(
            re.search(r"\b(i|i'm|i am|we|we'll)\b", lower)
        )
        has_future_form = self._contains_any(lower, self._commitment_future_markers)
        has_action = self._contains_any(lower, self._commitment_action_markers)
        explicit_future_action = has_future_form and has_action

        high_certainty = self._contains_any(lower, self._commitment_certainty_markers)
        time_constraint = bool(self._time_like_re.search(lower)) or self._contains_any(
            lower,
            self._commitment_time_markers,
        )
        explicit_promise_tone = self._contains_any(lower, self._commitment_promise_markers)

        return has_subject and explicit_future_action and high_certainty and (
            time_constraint or explicit_promise_tone
        )

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
            InputType.COMMITMENT,
            InputType.GOAL_STATEMENT,
            InputType.PREFERENCE,
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
