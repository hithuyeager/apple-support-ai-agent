import csv
import json
import random
from pathlib import Path


JUDGE_PATH = Path("data/evaluation/judge_predictions.jsonl")
OUTPUT_PATH = Path("data/evaluation/human_review.csv")

TARGET = 20
SEED = 42


def load_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def main():
    rows = load_jsonl(JUDGE_PATH)

    # Only use the exact 30 cases that were judged.
    auto_handled = [
        row
        for row in rows
        if row.get("should_escalate") is False
        and row.get("draft_response")
    ]

    if len(auto_handled) < TARGET:
        raise RuntimeError(
            f"Only {len(auto_handled)} auto-handled "
            f"cases available in the judge set."
        )

    # Sort by retrieval strength.
    auto_handled.sort(
        key=lambda row: (
            row.get("evidence", [{}])[0].get(
                "final_score",
                0.0,
            )
            if row.get("evidence")
            else 0.0
        )
    )

    # Take half weak and half strong retrieval cases.
    weak = auto_handled[:10]
    strong = auto_handled[-10:]

    selected = weak + strong

    # Shuffle so the reviewer doesn't know which are
    # intentionally weak/strong.
    rng = random.Random(SEED)
    rng.shuffle(selected)

    fieldnames = [
        "example_id",
        "customer_message",
        "predicted_intent",
        "confidence",
        "draft_response",
        "evidence_1",
        "evidence_2",
        "evidence_3",
        "judge_evidence_grounded",
        "human_evidence_grounded",
        "human_notes",
    ]

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in selected:
            evidence = row.get("evidence", [])

            writer.writerow({
                "example_id": row["example_id"],
                "customer_message": row["text"],
                "predicted_intent": row[
                    "predicted_intent"
                ],
                "confidence": row["confidence"],
                "draft_response": row.get(
                    "draft_response",
                    "",
                ),

                "evidence_1": (
                    evidence[0].get("support_response", "")
                    if len(evidence) > 0
                    else ""
                ),

                "evidence_2": (
                    evidence[1].get("support_response", "")
                    if len(evidence) > 1
                    else ""
                ),

                "evidence_3": (
                    evidence[2].get("support_response", "")
                    if len(evidence) > 2
                    else ""
                ),

                "judge_evidence_grounded": row.get(
                    "judge_evidence_grounded",
                    "",
                ),

                "human_evidence_grounded": "",

                "human_notes": "",
            })

    print("=== HUMAN REVIEW SET ===")
    print(f"Judge cases available : {len(rows)}")
    print(f"Auto-handled cases    : {len(auto_handled)}")
    print(f"Selected               : {len(selected)}")
    print(f"Output                 : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()