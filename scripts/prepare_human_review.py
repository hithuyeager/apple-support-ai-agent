import csv
import json
from pathlib import Path


JUDGE_SET_PATH = Path("data/evaluation/judge_set.jsonl")
OUTPUT_PATH = Path("data/evaluation/human_review.csv")

NUM_CASES = 20


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    cases = load_jsonl(JUDGE_SET_PATH)

    if len(cases) < NUM_CASES:
        raise ValueError(
            f"Expected at least {NUM_CASES} judge cases, found {len(cases)}"
        )

    # Prefer a balanced review set:
    # - 10 weaker-evidence cases
    # - 10 stronger-evidence cases
    #
    # Evidence strength is based on the top retrieved evidence score.
    scored = []

    for case in cases:
        evidence = case.get("evidence", [])

        top_score = 0.0
        if evidence:
            top_score = max(
                float(item.get("final_score", 0.0))
                for item in evidence
            )

        scored.append((top_score, case))

    scored.sort(key=lambda x: x[0])

    weak_cases = [case for _, case in scored[:10]]
    strong_cases = [case for _, case in scored[-10:]]

    selected = weak_cases + strong_cases

    # Remove accidental duplicates while preserving order.
    unique_cases = []
    seen = set()

    for case in selected:
        example_id = case["example_id"]

        if example_id not in seen:
            seen.add(example_id)
            unique_cases.append(case)

    if len(unique_cases) < NUM_CASES:
        # Fill from remaining cases if necessary.
        for _, case in scored:
            example_id = case["example_id"]

            if example_id not in seen:
                seen.add(example_id)
                unique_cases.append(case)

            if len(unique_cases) == NUM_CASES:
                break

    fieldnames = [
        "example_id",
        "text",
        "gold_intent",
        "predicted_intent",
        "confidence",
        "should_escalate",
        "risk_level",
        "draft_response",
        "top_evidence_score",
        "top_evidence",
        "human_evidence_grounded",
        "human_notes",
    ]

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for case in unique_cases:
            evidence = case.get("evidence", [])

            top_evidence_score = 0.0
            top_evidence_text = ""

            if evidence:
                top = max(
                    evidence,
                    key=lambda item: float(item.get("final_score", 0.0))
                )

                top_evidence_score = float(
                    top.get("final_score", 0.0)
                )

                matched_customer = top.get("matched_customer", {})
                support_response = top.get("support_response", {})

                top_evidence_text = (
                    f"Historical customer: "
                    f"{matched_customer.get('text', '')}\n"
                    f"Historical Apple Support response: "
                    f"{support_response.get('text', '')}"
                )

            writer.writerow(
                {
                    "example_id": case["example_id"],
                    "text": case["text"],
                    "gold_intent": case["gold_intent"],
                    "predicted_intent": case["predicted_intent"],
                    "confidence": case["confidence"],
                    "should_escalate": case["should_escalate"],
                    "risk_level": case["risk_level"],
                    "draft_response": case["draft_response"],
                    "top_evidence_score": round(top_evidence_score, 4),
                    "top_evidence": top_evidence_text,
                    "human_evidence_grounded": "",
                    "human_notes": "",
                }
            )

    print(f"Judge cases available : {len(cases)}")
    print(f"Human review cases     : {len(unique_cases)}")
    print(f"Wrote                  : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()