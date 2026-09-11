import json
import random
from pathlib import Path


INPUT = Path("data/evaluation/agent_predictions.jsonl")
OUTPUT = Path("data/evaluation/judge_set.jsonl")

SEED = 42
TARGET_SIZE = 30


def load_rows():
    with INPUT.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    rows = load_rows()

    # Deterministic RNG.
    rng = random.Random(SEED)

    selected = []
    selected_ids = set()

    def add_rows(candidates, count):
        candidates = list(candidates)
        rng.shuffle(candidates)

        for row in candidates:
            if len(selected) >= TARGET_SIZE:
                return

            example_id = row["example_id"]

            if example_id in selected_ids:
                continue

            selected.append(row)
            selected_ids.add(example_id)

    # ------------------------------------------------------------------
    # Bucket 1: classification errors
    # Important because these expose intent-boundary failures.
    # ------------------------------------------------------------------
    classification_errors = [
        r for r in rows
        if r["gold_intent"] != r["predicted_intent"]
    ]

    add_rows(classification_errors, 10)

    # ------------------------------------------------------------------
    # Bucket 2: escalated cases
    # Important for evaluating escalation safety.
    # ------------------------------------------------------------------
    escalated = [
        r for r in rows
        if r["should_escalate"]
    ]

    add_rows(escalated, 7)

    # ------------------------------------------------------------------
    # Bucket 3: strong retrieval + correct classification
    # These are our likely "good path" examples.
    # ------------------------------------------------------------------
    strong_retrieval = [
        r for r in rows
        if (
            not r["should_escalate"]
            and r["gold_intent"] == r["predicted_intent"]
            and r["evidence"]
            and r["evidence"][0]["final_score"] >= 0.50
        )
    ]

    add_rows(strong_retrieval, 8)

    # ------------------------------------------------------------------
    # Bucket 4: weaker retrieval + auto-handled
    # These are especially important for grounding failures.
    # ------------------------------------------------------------------
    weak_retrieval = [
        r for r in rows
        if (
            not r["should_escalate"]
            and r["draft_response"]
            and r["evidence"]
            and r["evidence"][0]["final_score"] < 0.50
        )
    ]

    add_rows(weak_retrieval, 5)

    # ------------------------------------------------------------------
    # Fill any remaining slots deterministically.
    # ------------------------------------------------------------------
    remaining = [
        r for r in rows
        if r["example_id"] not in selected_ids
    ]

    add_rows(remaining, TARGET_SIZE - len(selected))

    # Sort by example ID so the output is easy to inspect.
    selected.sort(key=lambda r: r["example_id"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=== JUDGE SET CREATED ===")
    print(f"Source examples : {len(rows)}")
    print(f"Judge examples  : {len(selected)}")
    print(f"Output          : {OUTPUT}")

    print("\nCategory counts:")

    print(
        "Classification errors:",
        sum(
            r["gold_intent"] != r["predicted_intent"]
            for r in selected
        ),
    )

    print(
        "Escalated:",
        sum(r["should_escalate"] for r in selected),
    )

    print(
        "Correct + strong retrieval:",
        sum(
            (
                not r["should_escalate"]
                and r["gold_intent"] == r["predicted_intent"]
                and r["evidence"]
                and r["evidence"][0]["final_score"] >= 0.50
            )
            for r in selected
        ),
    )

    print(
        "Weak retrieval:",
        sum(
            (
                not r["should_escalate"]
                and r["draft_response"]
                and r["evidence"]
                and r["evidence"][0]["final_score"] < 0.50
            )
            for r in selected
        ),
    )


if __name__ == "__main__":
    main()