import json
import random
from pathlib import Path


DATASET = Path("data/processed/customer_messages.jsonl")

messages = []

with DATASET.open("r", encoding="utf-8") as file:
    for line in file:
        messages.append(json.loads(line))


print("=== DATASET OVERVIEW ===")
print(f"Total messages: {len(messages)}")
print(f"Conversations: {len(set(m['conversation_id'] for m in messages))}")
print()


# Messages per conversation
conversation_counts = {}

for message in messages:
    conversation_id = message["conversation_id"]
    conversation_counts[conversation_id] = (
        conversation_counts.get(conversation_id, 0) + 1
    )

print("=== MESSAGES PER CONVERSATION ===")

for conversation_id, count in sorted(conversation_counts.items()):
    print(f"Conversation {conversation_id}: {count}")

print()


# Length statistics
lengths = [len(message["text"]) for message in messages]

print("=== TEXT LENGTH ===")
print(f"Shortest: {min(lengths)} characters")
print(f"Longest : {max(lengths)} characters")
print(f"Average : {sum(lengths) / len(lengths):.2f} characters")
print()


# Random samples
print("=== RANDOM SAMPLES ===")

random.seed(42)

for message in random.sample(messages, min(20, len(messages))):
    print(f"\n[{message['tweet_id']}]")
    print(message["text"])

print()


# Longest messages
print("=== LONGEST MESSAGES ===")

for message in sorted(
    messages,
    key=lambda m: len(m["text"]),
    reverse=True,
)[:10]:
    print(f"\n[{message['tweet_id']}] ({len(message['text'])} chars)")
    print(message["text"])

print()


# Empty / malformed checks
print("=== VALIDATION ===")

print(
    "Empty text:",
    sum(not message["text"].strip() for message in messages)
)

print(
    "Missing conversation ID:",
    sum(message.get("conversation_id") is None for message in messages)
)

print(
    "Missing tweet ID:",
    sum(not message.get("tweet_id") for message in messages)
)

print(
    "Missing timestamp:",
    sum(not message.get("timestamp") for message in messages)
)