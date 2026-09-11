import json
from pathlib import Path
from collections import Counter

PATH = Path("data/evaluation/agent_predictions.jsonl")


def main():
    rows = [json.loads(line) for line in PATH.open() if line.strip()]

    successful = [r for r in rows if not r["error"]]
    classification_correct = [
        r for r in successful
        if r["gold_intent"] == r["predicted_intent"]
    ]

    print("=== AGENT EVALUATION SUMMARY ===")
    print(f"Examples             : {len(rows)}")
    print(f"Successful runs      : {len(successful)}")
    print(f"Failed runs          : {len(rows) - len(successful)}")

    if successful:
        print(
            f"Classification acc.  : "
            f"{100 * len(classification_correct) / len(successful):.2f}%"
        )

        escalations = Counter(
            "escalate" if r["should_escalate"] else "auto_handle"
            for r in successful
        )

        print(f"Auto-handled         : {escalations['auto_handle']}")
        print(f"Escalated            : {escalations['escalate']}")

        intents = Counter(r["predicted_intent"] for r in successful)
        print("\nPredicted intents:")
        for intent, count in intents.most_common():
            print(f"  {intent:20s} {count}")


if __name__ == "__main__":
    main()
