from dataclasses import dataclass
from typing import Any


@dataclass
class EscalationDecision:
    should_escalate: bool
    reason: str
    risk_level: str


class EscalationGate:
    """
    Conservative human-handoff policy for the support agent.

    Escalate when:
    - the request is security/privacy sensitive
    - the request is billing/purchase related
    - the classifier is uncertain
    - there is no useful historical evidence
    - the retrieved evidence is too weak
    """

    HIGH_RISK_INTENTS = {
        "privacy_security",
        "billing_purchase",
    }

    SECURITY_RISK_PHRASES = {
        "hacked",
        "hack",
        "someone got into",
        "someone accessed",
        "someone access",
        "unauthorized access",
        "unauthorised access",
        "unauthorized login",
        "unauthorised login",
        "unknown login",
        "unknown device",
        "account compromised",
        "account was compromised",
        "account has been compromised",
        "someone changed my password",
        "password was changed",
        "password has been changed",
        "someone changed my security",
        "don't recognize this login",
        "dont recognize this login",
        "do not recognize this login",
        "don't recognize this device",
        "dont recognize this device",
        "do not recognize this device",
    }

    def __init__(
        self,
        min_classifier_confidence: float = 0.70,
        min_retrieval_score: float = 0.15,
    ):
        self.min_classifier_confidence = min_classifier_confidence
        self.min_retrieval_score = min_retrieval_score

    def decide(
        self,
        customer_message: str,
        intent: str,
        confidence: float,
        evidence: list[dict[str, Any]],
    ) -> EscalationDecision:

        if not customer_message.strip():
            return EscalationDecision(
                should_escalate=True,
                reason="Empty customer message cannot be safely handled.",
                risk_level="high",
            )

        normalized_message = customer_message.lower()

        # Safety override:
        # A message can be security-sensitive even when the classifier
        # labels it as ordinary account authentication.
        for phrase in self.SECURITY_RISK_PHRASES:
            if phrase in normalized_message:
                return EscalationDecision(
                    should_escalate=True,
                    reason=(
                        "Message indicates possible account compromise or "
                        "unauthorized access and requires human review."
                    ),
                    risk_level="high",
                )

        # High-risk intents are escalated by policy.
        if intent in self.HIGH_RISK_INTENTS:
            return EscalationDecision(
                should_escalate=True,
                reason=(
                    f"Intent '{intent}' may require account-specific or "
                    "policy-sensitive human action."
                ),
                risk_level="high",
            )

        # Low classifier confidence means we should not act autonomously.
        if confidence < self.min_classifier_confidence:
            return EscalationDecision(
                should_escalate=True,
                reason=(
                    f"Classifier confidence ({confidence:.2f}) is below "
                    f"the safety threshold "
                    f"({self.min_classifier_confidence:.2f})."
                ),
                risk_level="medium",
            )

        # No evidence means there is nothing historical to ground the reply in.
        if not evidence:
            return EscalationDecision(
                should_escalate=True,
                reason="No historical support evidence was retrieved.",
                risk_level="high",
            )

        best_score = max(
            float(item.get("final_score", 0.0))
            for item in evidence
        )

        if best_score < self.min_retrieval_score:
            return EscalationDecision(
                should_escalate=True,
                reason=(
                    f"Historical evidence is too weak "
                    f"(best retrieval score {best_score:.2f})."
                ),
                risk_level="medium",
            )

        return EscalationDecision(
            should_escalate=False,
            reason=(
                "Intent is within the current auto-handling scope and "
                "sufficient historical evidence was found."
            ),
            risk_level="low",
        )