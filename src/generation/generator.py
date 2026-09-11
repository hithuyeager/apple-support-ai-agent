from typing import Any

import ollama


class SupportGenerator:
    """Generate customer-facing replies grounded in historical evidence."""

    ESCALATION_PREFIX = "ESCALATE:"

    def __init__(self, model: str = "qwen3.5:4b"):
        self.model = model

    def generate(
        self,
        customer_message: str,
        intent: str,
        evidence: list[dict[str, Any]],
    ) -> str:

        if not customer_message.strip():
            raise ValueError("customer_message cannot be empty")

        if not evidence:
            return (
                f"{self.ESCALATION_PREFIX} "
                "No historical support evidence is available."
            )

        prompt = self._build_prompt(
            customer_message=customer_message,
            intent=intent,
            evidence=evidence,
        )

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an Apple customer support response "
                        "drafting assistant. You must strictly follow "
                        "the historical evidence provided by the user."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": 0,
            },
            think=False,
        )

        content = response["message"]["content"].strip()

        if not content:
            raise RuntimeError("Qwen returned an empty response")

        return content

    def _build_prompt(
        self,
        customer_message: str,
        intent: str,
        evidence: list[dict[str, Any]],
    ) -> str:

        formatted_evidence = self._format_evidence(evidence)

        return f"""
You are drafting a customer support reply using ONLY the historical
Apple Support evidence provided below.

CUSTOMER MESSAGE:
{customer_message}

CLASSIFIED INTENT:
{intent}

HISTORICAL EVIDENCE:
{formatted_evidence}

STRICT RULES:

1. Use ONLY information supported by the historical evidence.

2. Do NOT use your general knowledge about Apple, iPhone, iOS,
   settings, troubleshooting, policies, refunds, or products.

3. Do NOT invent troubleshooting steps.

4. Do NOT invent URLs.

5. Do NOT invent policies or guarantees.

6. Do NOT claim that Apple performed an action.

7. The historical customer message is context.
   The historical Apple Support response is the actual source
   for what Apple historically suggested.

8. If the evidence does NOT contain enough information to answer
   the customer's request safely, return EXACTLY:

ESCALATE: insufficient historical evidence

9. If the evidence does contain enough information, write a concise
   customer-facing response grounded in that evidence.

10. Return ONLY the response text. Do not explain your reasoning.
""".strip()

    def _format_evidence(
        self,
        evidence: list[dict[str, Any]],
    ) -> str:

        sections = []

        for index, item in enumerate(evidence, start=1):
            customer = item.get("matched_customer", {})
            support = item.get("support_response", {})

            sections.append(
                f"""
EVIDENCE {index}
Relevance score: {item.get("final_score", 0.0):.4f}

Historical customer:
{customer.get("text", "")}

Historical Apple Support response:
{support.get("text", "")}
""".strip()
            )

        return "\n\n".join(sections)