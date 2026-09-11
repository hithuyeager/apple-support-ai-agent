import json
import re
from pathlib import Path


RAW_FILE = Path("data/raw/apple_convos.txt")
OUTPUT_FILE = Path("data/processed/resolution_corpus.jsonl")


TWEET_ID_RE = re.compile(r"Tweet ID\s*:\s*(\S+)")
TIME_RE = re.compile(r"Time\s*:\s*(.+)")
TEXT_RE = re.compile(r"Text\s*:\s*(.*)")


def parse_raw_file():
    conversations = []

    current_conversation = None
    current_author = None
    current_tweet = None

    def finish_tweet():
        nonlocal current_tweet

        if current_tweet is not None:
            current_conversation["messages"].append(current_tweet)

        current_tweet = None

    def finish_conversation():
        nonlocal current_conversation

        finish_tweet()

        if current_conversation is not None:
            conversations.append(current_conversation)

        current_conversation = None

    with RAW_FILE.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            conversation_match = re.match(
                r"CONVERSATION\s+(\d+)\s+\|\s+(\d+)\s+messages",
                line,
            )

            if conversation_match:
                finish_conversation()

                current_conversation = {
                    "conversation_id": int(conversation_match.group(1)),
                    "messages": [],
                }

                continue

            if line == "[CUSTOMER]":
                finish_tweet()
                current_author = "customer"

                current_tweet = {
                    "author": "customer",
                }

                continue

            if line == "[APPLE SUPPORT]":
                finish_tweet()
                current_author = "apple_support"

                current_tweet = {
                    "author": "apple_support",
                }

                continue

            if current_tweet is None:
                continue

            tweet_id_match = TWEET_ID_RE.match(line)

            if tweet_id_match:
                current_tweet["tweet_id"] = tweet_id_match.group(1)
                continue

            time_match = TIME_RE.match(line)

            if time_match:
                current_tweet["timestamp"] = time_match.group(1)
                continue

            text_match = TEXT_RE.match(line)

            if text_match:
                current_tweet["text"] = text_match.group(1)
                continue

        finish_conversation()

    return conversations


def build_resolution_corpus(conversations):
    records = []

    for conversation in conversations:
        messages = conversation["messages"]

        for i, message in enumerate(messages):
            if message["author"] != "apple_support":
                continue

            # Look at nearby customer messages.
            previous_customer_messages = []

            for previous in reversed(messages[:i]):
                if previous["author"] == "customer":
                    previous_customer_messages.append(previous["text"])

                if len(previous_customer_messages) == 3:
                    break

            previous_customer_messages.reverse()

            if not previous_customer_messages:
                continue

            records.append(
                {
                    "conversation_id": conversation["conversation_id"],
                    "support_tweet_id": message.get("tweet_id"),
                    "support_timestamp": message.get("timestamp"),
                    "support_response": message.get("text", ""),
                    "customer_context": previous_customer_messages,
                }
            )

    return records


def main():
    conversations = parse_raw_file()

    records = build_resolution_corpus(conversations)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("=== RESOLUTION CORPUS ===")
    print(f"Conversations : {len(conversations)}")
    print(f"Resolutions   : {len(records)}")
    print(f"Output        : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()