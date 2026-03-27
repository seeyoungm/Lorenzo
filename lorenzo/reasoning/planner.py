from __future__ import annotations

from lorenzo.models import InputType, ProcessedInput, ReasoningPlan, RetrievedMemory


class ReasoningPlanner:
    def plan(self, processed: ProcessedInput, retrieved: list[RetrievedMemory]) -> ReasoningPlan:
        if processed.input_type is InputType.GOAL_STATEMENT:
            return ReasoningPlan(
                strategy="goal_planning",
                rationale="사용자 목표형 입력이므로 실행 가능한 단계 기반 응답을 우선한다.",
                steps=[
                    "관련 기억에서 기존 목표/제약 추출",
                    "현재 목표를 단기 실행 계획으로 분해",
                    "다음 행동 1~3개를 제안",
                ],
            )

        if processed.input_type is InputType.MEMORY_RECALL:
            return ReasoningPlan(
                strategy="memory_recall",
                rationale="기억 회상 질의이므로 관련 기억의 우선순위와 충돌 이력을 함께 확인한다.",
                steps=[
                    "Top 기억 후보 확인",
                    "충돌 해결 이력과 최신성 점검",
                    "근거 포함 회상 답변 생성",
                ],
            )

        if retrieved and retrieved[0].similarity_score >= 0.35:
            return ReasoningPlan(
                strategy="memory_grounded_response",
                rationale="유사 기억이 존재하므로 기억 기반 정합성을 우선한다.",
                steps=[
                    "상위 기억 핵심 요약",
                    "입력과 기억의 연결점 명시",
                    "맥락 일관성 있는 응답 생성",
                ],
            )

        return ReasoningPlan(
            strategy="direct_response",
            rationale="강한 유사 기억이 부족하여 일반 응답 전략을 사용한다.",
            steps=[
                "입력 의도 파악",
                "간결한 직접 답변 작성",
            ],
        )
