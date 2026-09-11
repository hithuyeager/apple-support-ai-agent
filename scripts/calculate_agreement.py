import csv
import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score


HUMAN_PATH = Path("data/evaluation/human_review.csv")
JUDGE_PATH = Path("data/evaluation/judge_predictions.jsonl")


def load_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def main():
    judge_rows = load_jsonl(JUDGE_PATH)

    # The judge predictions contain only the 30-case judge set.
    judge_by_id = {
        str(row["example_id"]): row
        for row in judge_rows
    }

    human_rows = []

    with HUMAN_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            human_rows.append(row)

    # Keep only cases that were actually evaluated by the LLM judge.
    matched_rows = [
        row
        for row in human_rows
        if str(row["example_id"]) in judge_by_id
    ]

    if not matched_rows:
        raise RuntimeError(
            "No human-review examples overlap with "
            "the 30-case judge set."
        )

    human_labels = []
    judge_labels = []

    disagreements = []

    for row in matched_rows:
        example_id = str(row["example_id"])

        human_value = (
            row["human_evidence_grounded"]
            .strip()
            .upper()
        )

        if human_value not in {"TRUE", "FALSE"}:
            raise ValueError(
                f"Invalid human label for example "
                f"{example_id}: {human_value}"
            )

        judge_value = judge_by_id[example_id][
            "judgment"
        ]["evidence_grounded"]

        human_label = human_value == "TRUE"
        judge_label = bool(judge_value)

        human_labels.append(human_label)
        judge_labels.append(judge_label)

        if human_label != judge_label:
            disagreements.append({
                "example_id": example_id,
                "human": human_label,
                "judge": judge_label,
                "customer_message": row[
                    "text"
                ],
                "human_notes": row.get(
                    "human_notes",
                    "",
                ),
                "judge_reason": judge_by_id[
                    example_id
                ]["judgment"].get(
                    "reason",
                    "",
                ),
            })

    total = len(human_labels)

    agreement_count = sum(
        human == judge
        for human, judge in zip(
            human_labels,
            judge_labels,
        )
    )

    agreement_pct = (
        agreement_count / total * 100
    )

    kappa = cohen_kappa_score(
        human_labels,
        judge_labels,
    )

    print("=== HUMAN / LLM JUDGE AGREEMENT ===")

    print(
        f"Judge cases available  : "
        f"{len(judge_rows)}"
    )

    print(
        f"Human cases            : "
        f"{len(human_rows)}"
    )

    print(
        f"Matched cases          : "
        f"{total}"
    )

    print(
        f"Raw agreement          : "
        f"{agreement_count}/{total} "
        f"({agreement_pct:.1f}%)"
    )

    print(
        f"Cohen's kappa          : "
        f"{kappa:.3f}"
    )

    print("\n--- Label Counts ---")

    print(
        f"Human TRUE             : "
        f"{sum(human_labels)}"
    )

    print(
        f"Human FALSE            : "
        f"{total - sum(human_labels)}"
    )

    print(
        f"Judge TRUE             : "
        f"{sum(judge_labels)}"
    )

    print(
        f"Judge FALSE            : "
        f"{total - sum(judge_labels)}"
    )

    print("\n--- Disagreements ---")

    if not disagreements:
        print("None")

    else:
        for item in disagreements:
            print(
                f"\nExample {item['example_id']}"
            )

            print(
                f"Human : {item['human']}"
            )

            print(
                f"Judge : {item['judge']}"
            )

            print(
                f"Customer: "
                f"{item['customer_message']}"
            )

            print(
                f"Human notes: "
                f"{item['human_notes']}"
            )

            print(
                f"Judge reason: "
                f"{item['judge_reason']}"
            )


if __name__ == "__main__":
    main()