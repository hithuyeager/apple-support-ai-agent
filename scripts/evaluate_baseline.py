import json
from pathlib import Path
from collections import Counter

from src.classification.baseline import RuleBasedClassifier

EVAL_FILE = Path("data/evaluation/eval_set.jsonl")


def load_eval_set():
    examples = []

    with EVAL_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                examples.append(json.loads(line))

    return examples


def main():
    examples = load_eval_set()
    classifier = RuleBasedClassifier()

    y_true = []
    y_pred = []

    for example in examples:
        text = example["text"]
        true_intent = example["intent"]

        result = classifier.classify(text)

        y_true.append(true_intent)
        y_pred.append(result.intent)

    correct = sum(
        true == pred
        for true, pred in zip(y_true, y_pred)
    )

    total = len(y_true)
    accuracy = correct / total if total else 0

    print("\n=== BASELINE EVALUATION ===")
    print(f"Total examples : {total}")
    print(f"Correct        : {correct}")
    print(f"Incorrect      : {total - correct}")
    print(f"Accuracy       : {accuracy:.2%}")

    print("\n=== PREDICTED DISTRIBUTION ===")
    for intent, count in Counter(y_pred).most_common():
        print(f"{intent:20} {count}")

    print("\n=== ERRORS ===")

    for example, true, pred in zip(examples, y_true, y_pred):
        if true != pred:
            print("\nText:", example["text"])
            print("Expected:", true)
            print("Predicted:", pred)


if __name__ == "__main__":
    main()