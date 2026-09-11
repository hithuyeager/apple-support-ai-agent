import json
import random
from pathlib import Path


INPUT_FILE = Path("data/processed/customer_messages.jsonl")
OUTPUT_FILE = Path("data/evaluation/eval_candidates.jsonl")

SAMPLE_SIZE = 150
SEED = 42


def load_messages():
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    messages = load_messages()

    random.seed(SEED)

    # First, shuffle deterministically so selection is reproducible.
    shuffled = messages.copy()
    random.shuffle(shuffled)

    # Spread samples across the 30 conversations.
    by_conversation = {}

    for message in shuffled:
        conversation_id = message["conversation_id"]
        by_conversation.setdefault(conversation_id, []).append(message)

    candidates = []

    conversation_ids = sorted(by_conversation)

    # Round-robin selection gives every conversation representation.
    index = 0

    while len(candidates) < min(SAMPLE_SIZE, len(messages)):
        added_this_round = False

        for conversation_id in conversation_ids:
            messages_in_conversation = by_conversation[conversation_id]

            if index < len(messages_in_conversation):
                candidates.append(messages_in_conversation[index])
                added_this_round = True

                if len(candidates) == SAMPLE_SIZE:
                    break

        if not added_this_round:
            break

        index += 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for message in candidates:
            record = {
                **message,
                "intent": None,
            }

            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print("=== EVALUATION CANDIDATES ===")
    print(f"Source messages : {len(messages)}")
    print(f"Candidates      : {len(candidates)}")
    print(f"Seed            : {SEED}")
    print(f"Output          : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()