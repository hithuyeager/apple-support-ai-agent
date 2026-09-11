from typing import Any

from src.classification.qwen import QwenClassifier
from src.retrival.tfidf_retriever import TfidfRetriever
from src.generation.generator import SupportGenerator
from src.evaluation.escalation import EscalationGate


class SupportAgent:
    """End-to-end Apple support agent with escalation and grounding."""

    def __init__(
        self,
        classifier: QwenClassifier | None = None,
        retriever: TfidfRetriever | None = None,
        generator: SupportGenerator | None = None,
        escalation_gate: EscalationGate | None = None,
    ):
        self.classifier = classifier or QwenClassifier()
        self.retriever = retriever or TfidfRetriever()
        self.generator = generator or SupportGenerator()
        self.escalation_gate = escalation_gate or EscalationGate()

    def handle(
        self,
        customer_message: str,
        retrieval_top_k: int = 10,
        responses_per_conversation: int = 3,
        generation_evidence_k: int = 5,
    ) -> dict[str, Any]:

        if not customer_message.strip():
            raise ValueError("customer_message cannot be empty")

        # 1. Classify.
        classification = self.classifier.classify(
            customer_message
        )

        # 2. Retrieve historical evidence.
        evidence = self.retriever.search(
            customer_message,
            top_k=retrieval_top_k,
            responses_per_conversation=responses_per_conversation,
        )

        # 3. Initial escalation decision.
        escalation = self.escalation_gate.decide(
            customer_message=customer_message,
            intent=classification.intent,
            confidence=classification.confidence,
            evidence=evidence,
        )

        # 4. Never generate an autonomous reply for an already-escalated case.
        if escalation.should_escalate:
            return {
                "message": customer_message,
                "intent": classification.intent,
                "confidence": classification.confidence,
                "evidence": evidence,
                "should_escalate": True,
                "escalation_reason": escalation.reason,
                "risk_level": escalation.risk_level,
                "draft_response": None,
            }

        # 5. Generate only from the highest-ranked evidence.
        generation_evidence = evidence[:generation_evidence_k]

        draft_response = self.generator.generate(
            customer_message=customer_message,
            intent=classification.intent,
            evidence=generation_evidence,
        )

        # 6. Generator can reject the case when evidence is insufficient.
        if draft_response.startswith(
            self.generator.ESCALATION_PREFIX
        ):
            return {
                "message": customer_message,
                "intent": classification.intent,
                "confidence": classification.confidence,
                "evidence": evidence,
                "should_escalate": True,
                "escalation_reason": (
                    "Historical evidence was retrieved, but it did not "
                    "contain enough information to safely draft a grounded "
                    "response."
                ),
                "risk_level": "medium",
                "draft_response": None,
            }

        return {
            "message": customer_message,
            "intent": classification.intent,
            "confidence": classification.confidence,
            "evidence": evidence,
            "should_escalate": False,
            "escalation_reason": escalation.reason,
            "risk_level": escalation.risk_level,
            "draft_response": draft_response,
        }