import json
from pathlib import Path


PREDICTION_FILE = Path("data/evaluation/qwen_predictions.jsonl")


def main():
    errors = []

    with PREDICTION_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if row["gold"] != row["predicted"]:
                errors.append(row)

    print(f"Total errors: {len(errors)}")
    print("=" * 80)

    for i, row in enumerate(errors, 1):
        print(f"\nERROR {i}")
        print(f"Message    : {row['text']}")
        print(f"Expected   : {row['gold']}")
        print(f"Predicted  : {row['predicted']}")
        print(f"Confidence : {row['confidence']}")
        print("-" * 80)


if __name__ == "__main__":
    main()