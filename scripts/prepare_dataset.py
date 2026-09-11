import json
import re
from pathlib import Path


RAW_FILE = Path("data/raw/apple_convos.txt")
OUTPUT_FILE = Path("data/processed/customer_messages.jsonl")


def parse_conversations():
    conversations = []
    current_conversation = None
    current_message = None

    with RAW_FILE.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")

            # Conversation boundary
            conversation_match = re.match(
                r"CONVERSATION\s+(\d+)\s+\|\s+(\d+)\s+messages",
                line,
            )

            if conversation_match:
                if current_message:
                    current_conversation["messages"].append(current_message)
                    current_message = None

                if current_conversation:
                    conversations.append(current_conversation)

                current_conversation = {
                    "conversation_id": int(conversation_match.group(1)),
                    "declared_message_count": int(conversation_match.group(2)),
                    "messages": [],
                }

                continue

            # Speaker boundary
            if line in ("[CUSTOMER]", "[APPLE SUPPORT]"):
                if current_message:
                    current_conversation["messages"].append(current_message)

                current_message = {
                    "speaker": (
                        "customer"
                        if line == "[CUSTOMER]"
                        else "apple_support"
                    )
                }

                continue

            if current_message is None:
                continue

            # Message metadata
            if line.startswith("Tweet ID"):
                current_message["tweet_id"] = line.split(":", 1)[1].strip()

            elif line.startswith("Time"):
                current_message["timestamp"] = line.split(":", 1)[1].strip()

            elif line.startswith("Text"):
                current_message["text"] = line.split(":", 1)[1].strip()

            # Ignore formatting/empty lines outside the fields we need.

    # Flush final message/conversation
    if current_message and current_conversation:
        current_conversation["messages"].append(current_message)

    if current_conversation:
        conversations.append(current_conversation)

    return conversations


def extract_customer_messages(conversations):
    customer_messages = []

    for conversation in conversations:
        for message in conversation["messages"]:
            if message["speaker"] != "customer":
                continue

            if not message.get("text"):
                continue

            customer_messages.append(
                {
                    "conversation_id": conversation["conversation_id"],
                    "tweet_id": message.get("tweet_id"),
                    "timestamp": message.get("timestamp"),
                    "text": message["text"],
                }
            )

    return customer_messages


def validate(customer_messages):
    tweet_ids = [
        message["tweet_id"]
        for message in customer_messages
        if message.get("tweet_id")
    ]

    duplicate_ids = len(tweet_ids) - len(set(tweet_ids))

    missing_text = sum(
        1
        for message in customer_messages
        if not message.get("text")
    )

    missing_tweet_id = sum(
        1
        for message in customer_messages
        if not message.get("tweet_id")
    )

    return {
        "customer_messages": len(customer_messages),
        "unique_tweet_ids": len(set(tweet_ids)),
        "duplicate_tweet_ids": duplicate_ids,
        "missing_text": missing_text,
        "missing_tweet_id": missing_tweet_id,
    }


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")

    conversations = parse_conversations()
    customer_messages = extract_customer_messages(conversations)

    stats = validate(customer_messages)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for message in customer_messages:
            file.write(
                json.dumps(message, ensure_ascii=False) + "\n"
            )

    print("\n=== INGESTION COMPLETE ===")
    print(f"Conversations parsed : {len(conversations)}")
    print(f"Customer messages    : {stats['customer_messages']}")
    print(f"Unique tweet IDs     : {stats['unique_tweet_ids']}")
    print(f"Duplicate tweet IDs  : {stats['duplicate_tweet_ids']}")
    print(f"Missing text         : {stats['missing_text']}")
    print(f"Missing tweet IDs    : {stats['missing_tweet_id']}")
    print(f"Output               : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()