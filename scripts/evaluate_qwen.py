import json
from collections import Counter
from pathlib import Path

from src.classification.qwen import QwenClassifier


EVAL_FILE = Path("data/evaluation/eval_set.jsonl")
PREDICTION_FILE = Path("data/evaluation/qwen_predictions.jsonl")

MODEL = "qwen3.5:4b"


def load_eval_set():
    examples = []

    with EVAL_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    return examples


def get_text(row):
    for key in ("text", "customer_text", "message"):
        if key in row:
            return row[key]

    raise KeyError(f"Could not find message text in row: {row}")


def get_label(row):
    for key in ("intent", "label", "gold_intent"):
        if key in row:
            return row[key]

    raise KeyError(f"Could not find gold label in row: {row}")


def load_cached_predictions():
    predictions = {}

    if not PREDICTION_FILE.exists():
        return predictions

    with PREDICTION_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                predictions[row["index"]] = row

    return predictions


def append_prediction(row):
    PREDICTION_FILE.parent.mkdir(parents=True, exist_ok=True)

    with PREDICTION_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def calculate_metrics(results):
    total = len(results)
    correct = sum(
        result["gold"] == result["predicted"]
        for result in results
    )

    accuracy = correct / total if total else 0.0

    labels = sorted(
        set(result["gold"] for result in results)
        | set(result["predicted"] for result in results)
    )

    print()
    print("=== QWEN EVALUATION ===")
    print(f"Model          : {MODEL}")
    print(f"Total examples : {total}")
    print(f"Correct        : {correct}")
    print(f"Incorrect      : {total - correct}")
    print(f"Accuracy       : {accuracy:.2%}")

    print()
    print("=== PER-INTENT METRICS ===")

    f1_scores = []

    for label in labels:
        tp = sum(
            r["gold"] == label and r["predicted"] == label
            for r in results
        )

        fp = sum(
            r["gold"] != label and r["predicted"] == label
            for r in results
        )

        fn = sum(
            r["gold"] == label and r["predicted"] != label
            for r in results
        )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        f1_scores.append(f1)

        print(
            f"{label:20}"
            f"P={precision:.2f} "
            f"R={recall:.2f} "
            f"F1={f1:.2f}"
        )

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    print()
    print(f"Macro-F1       : {macro_f1:.2%}")


def main():
    examples = load_eval_set()
    cached = load_cached_predictions()

    classifier = QwenClassifier(model=MODEL)

    results = []

    for index, row in enumerate(examples):
        gold = get_label(row)

        if index in cached:
            prediction = cached[index]
        else:
            text = get_text(row)

            print(
                f"[{index + 1}/{len(examples)}] "
                f"classifying..."
            )

            result = classifier.classify(text)

            prediction = {
                "index": index,
                "text": text,
                "gold": gold,
                "predicted": result.intent,
                "confidence": result.confidence,
            }

            append_prediction(prediction)

        results.append(prediction)

    calculate_metrics(results)


if __name__ == "__main__":
    main()