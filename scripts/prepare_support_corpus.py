import json
import re
from pathlib import Path


RAW_FILE = Path("data/raw/apple_convos.txt")
OUTPUT_FILE = Path("data/processed/support_responses.jsonl")


def parse_conversations():
    conversations = []
    current_conversation = None
    current_message = None

    with RAW_FILE.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")

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
                    "declared_message_count": int(
                        conversation_match.group(2)
                    ),
                    "messages": [],
                }
                continue

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

            if line.startswith("Tweet ID"):
                current_message["tweet_id"] = line.split(
                    ":", 1
                )[1].strip()

            elif line.startswith("Time"):
                current_message["timestamp"] = line.split(
                    ":", 1
                )[1].strip()

            elif line.startswith("Text"):
                current_message["text"] = line.split(
                    ":", 1
                )[1].strip()

    if current_message and current_conversation:
        current_conversation["messages"].append(current_message)

    if current_conversation:
        conversations.append(current_conversation)

    return conversations


def extract_support_responses(conversations):
    support_responses = []

    for conversation in conversations:
        for message in conversation["messages"]:
            if message["speaker"] != "apple_support":
                continue

            if not message.get("text"):
                continue

            support_responses.append(
                {
                    "conversation_id": conversation["conversation_id"],
                    "tweet_id": message.get("tweet_id"),
                    "timestamp": message.get("timestamp"),
                    "text": message["text"],
                }
            )

    return support_responses


def validate(support_responses):
    tweet_ids = [
        response["tweet_id"]
        for response in support_responses
        if response.get("tweet_id")
    ]

    duplicate_ids = len(tweet_ids) - len(set(tweet_ids))

    missing_text = sum(
        1
        for response in support_responses
        if not response.get("text")
    )

    missing_tweet_id = sum(
        1
        for response in support_responses
        if not response.get("tweet_id")
    )

    return {
        "support_responses": len(support_responses),
        "unique_tweet_ids": len(set(tweet_ids)),
        "duplicate_tweet_ids": duplicate_ids,
        "missing_text": missing_text,
        "missing_tweet_id": missing_tweet_id,
    }


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}"
        )

    conversations = parse_conversations()

    support_responses = extract_support_responses(
        conversations
    )

    stats = validate(support_responses)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for response in support_responses:
            file.write(
                json.dumps(
                    response,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("\n=== SUPPORT CORPUS COMPLETE ===")
    print(f"Conversations parsed : {len(conversations)}")
    print(f"Support responses    : {stats['support_responses']}")
    print(f"Unique tweet IDs     : {stats['unique_tweet_ids']}")
    print(f"Duplicate tweet IDs  : {stats['duplicate_tweet_ids']}")
    print(f"Missing text         : {stats['missing_text']}")
    print(f"Missing tweet IDs    : {stats['missing_tweet_id']}")
    print(f"Output               : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
    