import json
from pathlib import Path
from statistics import mean, median


AGENT_PATH = Path("data/evaluation/agent_predictions.jsonl")
JUDGE_PATH = Path("data/evaluation/judge_predictions.jsonl")
HUMAN_PATH = Path("data/evaluation/human_review.csv")


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    agent_rows = load_jsonl(AGENT_PATH)
    judge_rows = load_jsonl(JUDGE_PATH)

    judge_by_id = {
        str(row["example_id"]): row
        for row in judge_rows
    }

    print("=" * 70)
    print("RETRIEVAL SCORE ANALYSIS")
    print("=" * 70)

    scores = []

    for row in agent_rows:
        evidence = row.get("evidence", [])

        if not evidence:
            best_score = 0.0
        else:
            best_score = max(
                float(item.get("final_score", 0.0))
                for item in evidence
            )

        scores.append(
            {
                "example_id": row["example_id"],
                "score": best_score,
                "predicted_intent": row["predicted_intent"],
                "gold_intent": row["gold_intent"],
                "auto_handled": not row["should_escalate"],
            }
        )

    values = [item["score"] for item in scores]

    print(f"\nTotal examples : {len(scores)}")
    print(f"Min score      : {min(values):.4f}")
    print(f"Median score   : {median(values):.4f}")
    print(f"Mean score     : {mean(values):.4f}")
    print(f"Max score      : {max(values):.4f}")

    print("\n--- Score Distribution ---")

    buckets = [
        (0.00, 0.20),
        (0.20, 0.30),
        (0.30, 0.40),
        (0.40, 0.45),
        (0.45, 0.50),
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 0.70),
        (0.70, 1.01),
    ]

    for low, high in buckets:
        bucket = [
            item
            for item in scores
            if low <= item["score"] < high
        ]

        if bucket:
            print(
                f"{low:.2f} - {high:.2f}: "
                f"{len(bucket):3d} examples"
            )

    # ---------------------------------------------------------
    # Analyze the 30 LLM-judged examples.
    # ---------------------------------------------------------

    judged = []

    for item in scores:
        example_id = str(item["example_id"])

        if example_id not in judge_by_id:
            continue

        judgment = judge_by_id[example_id]["judgment"]

        grounded = bool(
            judgment["evidence_grounded"]
        )

        judged.append(
            {
                **item,
                "judge_grounded": grounded,
            }
        )

    print("\n--- LLM Judge Cases ---")

    grounded_scores = [
        item["score"]
        for item in judged
        if item["judge_grounded"]
    ]

    ungrounded_scores = [
        item["score"]
        for item in judged
        if not item["judge_grounded"]
    ]

    print(
        f"Grounded cases   : {len(grounded_scores)}"
    )

    if grounded_scores:
        print(
            f"  mean           : {mean(grounded_scores):.4f}"
        )
        print(
            f"  median         : {median(grounded_scores):.4f}"
        )
        print(
            f"  min            : {min(grounded_scores):.4f}"
        )
        print(
            f"  max            : {max(grounded_scores):.4f}"
        )

    print(
        f"\nUngrounded cases : {len(ungrounded_scores)}"
    )

    if ungrounded_scores:
        print(
            f"  mean           : {mean(ungrounded_scores):.4f}"
        )
        print(
            f"  median         : {median(ungrounded_scores):.4f}"
        )
        print(
            f"  min            : {min(ungrounded_scores):.4f}"
        )
        print(
            f"  max            : {max(ungrounded_scores):.4f}"
        )

    # ---------------------------------------------------------
    # Threshold analysis.
    # ---------------------------------------------------------

    print("\n--- Candidate Thresholds ---")

    print(
        "\nThreshold | Auto-handled | "
        "Grounded among auto | Ungrounded among auto"
    )
    print("-" * 70)

    for threshold in [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]:
        auto = [
            item
            for item in judged
            if item["score"] >= threshold
        ]

        grounded_auto = [
            item
            for item in auto
            if item["judge_grounded"]
        ]

        ungrounded_auto = [
            item
            for item in auto
            if not item["judge_grounded"]
        ]

        if auto:
            grounded_rate = (
                len(grounded_auto) / len(auto) * 100
            )
        else:
            grounded_rate = 0.0

        print(
            f"{threshold:8.2f} | "
            f"{len(auto):12d} | "
            f"{grounded_rate:18.1f}% | "
            f"{len(ungrounded_auto):18d}"
        )

    # ---------------------------------------------------------
    # Show borderline / surprising examples.
    # ---------------------------------------------------------

    print("\n--- Highest-scoring judge cases marked UNGROUNDED ---")

    bad_high_score = sorted(
        [
            item
            for item in judged
            if not item["judge_grounded"]
        ],
        key=lambda item: item["score"],
        reverse=True,
    )

    for item in bad_high_score[:10]:
        print(
            f"\nExample {item['example_id']} "
            f"score={item['score']:.4f}"
        )
        print(
            f"Gold={item['gold_intent']} "
            f"Predicted={item['predicted_intent']}"
        )

    print("\n--- Lowest-scoring judge cases marked GROUNDED ---")

    good_low_score = sorted(
        [
            item
            for item in judged
            if item["judge_grounded"]
        ],
        key=lambda item: item["score"],
    )

    for item in good_low_score[:10]:
        print(
            f"\nExample {item['example_id']} "
            f"score={item['score']:.4f}"
        )
        print(
            f"Gold={item['gold_intent']} "
            f"Predicted={item['predicted_intent']}"
        )


if __name__ == "__main__":
    main()