import json
import time
from pathlib import Path

from src.agent import SupportAgent


EVAL_PATH = Path("data/evaluation/eval_set.jsonl")
OUTPUT_PATH = Path("data/evaluation/agent_predictions.jsonl")


def load_eval_set():
    with EVAL_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    examples = load_eval_set()
    agent = SupportAgent()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = len(examples)
    started = time.time()

    with OUTPUT_PATH.open("w") as out:
        for i, example in enumerate(examples, start=1):
            message = example["text"]

            print(f"[{i}/{total}] {message[:100]}")

            try:
                result = agent.handle(message)

                record = {
                    "example_id": example.get("example_id", i),
                    "conversation_id": example.get("conversation_id"),
                    "tweet_id": example.get("tweet_id"),
                    "text": message,
                    "gold_intent": example.get("intent"),
                    "predicted_intent": result["intent"],
                    "confidence": result["confidence"],
                    "should_escalate": result["should_escalate"],
                    "risk_level": result["risk_level"],
                    "escalation_reason": result["escalation_reason"],
                    "draft_response": result["draft_response"],
                    "evidence": result["evidence"][:5],
                    "error": None,
                }

            except Exception as exc:
                record = {
                    "example_id": example.get("example_id", i),
                    "conversation_id": example.get("conversation_id"),
                    "tweet_id": example.get("tweet_id"),
                    "text": message,
                    "gold_intent": example.get("intent"),
                    "predicted_intent": None,
                    "confidence": None,
                    "should_escalate": None,
                    "risk_level": None,
                    "escalation_reason": None,
                    "draft_response": None,
                    "evidence": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }

                print(f"  ERROR: {record['error']}")

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    elapsed = time.time() - started
    print()
    print("=== EVALUATION COMPLETE ===")
    print(f"Examples : {total}")
    print(f"Output   : {OUTPUT_PATH}")
    print(f"Time     : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
