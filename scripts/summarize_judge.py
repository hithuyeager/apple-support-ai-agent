import json
import statistics
from collections import Counter
from pathlib import Path


INPUT_PATH = Path("data/evaluation/judge_predictions.jsonl")


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(value, total):
    return (value / total * 100) if total else 0.0


def main():
    rows = load_jsonl(INPUT_PATH)

    if not rows:
        raise RuntimeError("No judge predictions found.")

    score_fields = [
        "relevance",
        "groundedness",
        "helpfulness",
        "evidence_quality",
        "escalation_appropriateness",
    ]

    print("=== LLM JUDGE SUMMARY ===")
    print(f"Examples              : {len(rows)}")

    print("\n--- Mean Scores ---")

    for field in score_fields:
        values = [
            float(row["judgment"][field])
            for row in rows
            if field in row.get("judgment", {})
        ]

        if values:
            print(
                f"{field:25}: "
                f"{statistics.mean(values):.2f}/5"
            )

    print("\n--- Binary Metrics ---")

    grounded = [
        row["judgment"]["evidence_grounded"]
        for row in rows
        if "evidence_grounded" in row.get("judgment", {})
    ]

    passes = [
        row["judgment"]["overall_pass"]
        for row in rows
        if "overall_pass" in row.get("judgment", {})
    ]

    critical = [
        row["judgment"]["critical_failure"]
        for row in rows
        if "critical_failure" in row.get("judgment", {})
    ]

    grounded_yes = sum(value is True for value in grounded)
    pass_yes = sum(value is True for value in passes)
    critical_yes = sum(value is True for value in critical)

    print(
        f"Evidence grounded      : "
        f"{grounded_yes}/{len(grounded)} "
        f"({pct(grounded_yes, len(grounded)):.1f}%)"
    )

    print(
        f"Overall pass            : "
        f"{pass_yes}/{len(passes)} "
        f"({pct(pass_yes, len(passes)):.1f}%)"
    )

    print(
        f"Critical failures       : "
        f"{critical_yes}/{len(critical)} "
        f"({pct(critical_yes, len(critical)):.1f}%)"
    )

    print("\n--- Escalation Appropriateness ---")

    escalation_values = [
        int(row["judgment"]["escalation_appropriateness"])
        for row in rows
        if "escalation_appropriateness" in row.get("judgment", {})
    ]

    if escalation_values:
        print(
            f"Mean                    : "
            f"{statistics.mean(escalation_values):.2f}/5"
        )
        print(
            f"Median                  : "
            f"{statistics.median(escalation_values):.2f}/5"
        )

    print("\n--- Risk / Escalation Breakdown ---")

    escalation_counter = Counter(
        row.get("should_escalate")
        for row in rows
    )

    print(f"Auto-handled            : {escalation_counter.get(False, 0)}")
    print(f"Escalated               : {escalation_counter.get(True, 0)}")

    print("\n--- Critical Failure Cases ---")

    failures = [
        row
        for row in rows
        if row["judgment"].get("critical_failure") is True
    ]

    if not failures:
        print("None")
    else:
        for row in failures:
            judgment = row["judgment"]

            print(
                f"\nExample {row['example_id']}"
            )
            print(
                f"Reason: {judgment.get('reason', 'N/A')}"
            )

    print("\n--- Weak Evidence Cases ---")

    weak = [
        row
        for row in rows
        if row["judgment"].get("evidence_grounded") is False
    ]

    if not weak:
        print("None")
    else:
        for row in weak:
            judgment = row["judgment"]

            print(
                f"\nExample {row['example_id']}"
            )
            print(
                f"Reason: {judgment.get('reason', 'N/A')}"
            )


if __name__ == "__main__":
    main()