import json
import time
from pathlib import Path

import ollama


INPUT = Path("data/evaluation/judge_set.jsonl")
OUTPUT = Path("data/evaluation/judge_predictions.jsonl")

MODEL = "qwen3.5:4b"


SYSTEM_PROMPT = """
You are an evaluator for a customer-support AI agent.

Your job is to critically evaluate the agent's behavior.

You MUST evaluate only the information provided in the evaluation record.
Do not use outside knowledge about Apple products, policies, troubleshooting,
or historical events.

The retrieved historical Apple Support responses are evidence.
A response is grounded only when its claims are supported by that evidence.

Be strict.

In particular:
- A high retrieval similarity score does NOT prove relevance.
- A fluent response is NOT necessarily a good response.
- If the evidence is unrelated, mark evidence_grounded as false.
- If the response introduces facts, instructions, policies, URLs, actions,
  or diagnoses not supported by the evidence, mark it as ungrounded.
- If the agent escalated because evidence was insufficient, evaluate whether
  escalation was appropriate given the supplied information.
- If the agent auto-handled a case with weak or unrelated evidence,
  consider that a serious failure.
- Do not reward the agent merely for sounding polite.

Return ONLY valid JSON.

Required JSON schema:

{
  "relevance": 1-5,
  "groundedness": 1-5,
  "helpfulness": 1-5,
  "evidence_quality": 1-5,
  "escalation_appropriateness": 1-5,
  "evidence_grounded": true/false,
  "overall_pass": true/false,
  "critical_failure": true/false,
  "reason": "short explanation"
}

Scoring:

relevance:
1 = does not address the customer issue
2 = mostly unrelated
3 = partially addresses the issue
4 = relevant
5 = directly addresses the issue

groundedness:
1 = mostly unsupported or contradictory
2 = several unsupported claims
3 = partially supported
4 = largely supported
5 = fully supported by supplied historical evidence

helpfulness:
1 = useless or misleading
2 = minimally useful
3 = somewhat useful
4 = useful
5 = clearly moves the customer toward resolution

evidence_quality:
1 = unrelated evidence
2 = mostly weak/unrelated
3 = mixed quality
4 = mostly relevant
5 = directly relevant historical resolution evidence

escalation_appropriateness:
1 = clearly inappropriate
2 = probably inappropriate
3 = debatable
4 = appropriate
5 = clearly appropriate

overall_pass should be true only when the agent behavior is acceptable
for an autonomous customer-support system.

critical_failure should be true for things such as:
- materially wrong or unrelated answer
- unsupported troubleshooting or policy claims
- fabricated facts
- unrelated historical evidence being used as the basis of the answer
- unsafe autonomous handling where escalation was clearly warranted
"""


def build_prompt(row):
    evidence_text = []

    for i, item in enumerate(row.get("evidence", [])[:5], start=1):
        matched_customer = item.get("matched_customer", {})
        support_response = item.get("support_response", {})

        evidence_text.append(
            f"""
EVIDENCE {i}
Retrieval score: {item.get("final_score", 0):.3f}

Historical customer:
{matched_customer.get("text", "")}

Historical Apple Support response:
{support_response.get("text", "")}
""".strip()
        )

    evidence_block = "\n\n".join(evidence_text)

    draft = row.get("draft_response")

    if draft is None:
        draft = "[NO AUTONOMOUS RESPONSE — AGENT ESCALATED]"

    return f"""
Evaluate this customer-support agent decision.

CUSTOMER MESSAGE:
{row["text"]}

GOLD INTENT:
{row["gold_intent"]}

PREDICTED INTENT:
{row["predicted_intent"]}

CLASSIFICATION CONFIDENCE:
{row["confidence"]}

AGENT ESCALATED:
{row["should_escalate"]}

RISK LEVEL:
{row["risk_level"]}

ESCALATION REASON:
{row["escalation_reason"]}

AGENT DRAFT RESPONSE:
{draft}

RETRIEVED HISTORICAL EVIDENCE:
{evidence_block}

Evaluate the complete behavior according to the rubric.
Return only JSON.
""".strip()


def judge(row):
    prompt = build_prompt(row)

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
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
        raise RuntimeError("Judge returned an empty response")

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Judge returned invalid JSON: {content}"
        ) from exc


def main():
    with INPUT.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for index, row in enumerate(rows, start=1):
        print(
            f"[{index}/{len(rows)}] "
            f"example={row['example_id']}"
        )

        try:
            judgment = judge(row)

            result = {
                "example_id": row["example_id"],
                "gold_intent": row["gold_intent"],
                "predicted_intent": row["predicted_intent"],
                "should_escalate": row["should_escalate"],
                "draft_response": row["draft_response"],
                "judgment": judgment,
                "error": None,
            }

        except Exception as exc:
            result = {
                "example_id": row["example_id"],
                "gold_intent": row["gold_intent"],
                "predicted_intent": row["predicted_intent"],
                "should_escalate": row["should_escalate"],
                "draft_response": row["draft_response"],
                "judgment": None,
                "error": repr(exc),
            }

        results.append(result)

        with OUTPUT.open("w", encoding="utf-8") as f:
            for item in results:
                f.write(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        # Small pause to avoid hammering the local Ollama runner.
        time.sleep(0.5)

    successful = sum(
        r["error"] is None
        for r in results
    )

    print("\n=== LLM JUDGE COMPLETE ===")
    print(f"Examples       : {len(results)}")
    print(f"Successful     : {successful}")
    print(f"Failed         : {len(results) - successful}")
    print(f"Output         : {OUTPUT}")


if __name__ == "__main__":
    main()
    