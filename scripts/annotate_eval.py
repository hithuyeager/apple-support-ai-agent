import json
from pathlib import Path


INPUT_FILE = Path("data/evaluation/eval_candidates.jsonl")
OUTPUT_FILE = Path("data/evaluation/eval_set.jsonl")


INTENTS = [
    "software_os",
    "device_hardware",
    "connectivity",
    "account_auth",
    "apps_services",
    "setup_activation",
    "billing_purchase",
    "howto_feature",
    "privacy_security",
    "other",
]


def load_candidates():
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_existing_labels():
    if not OUTPUT_FILE.exists():
        return {}

    labeled = {}

    with OUTPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            labeled[record["tweet_id"]] = record

    return labeled


def save_labels(records):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def print_intents():
    print("\nIntent options:")

    for index, intent in enumerate(INTENTS, start=1):
        print(f"  {index:2}. {intent}")


def main():
    candidates = load_candidates()
    existing = load_existing_labels()

    records = []

    for candidate in candidates:
        if candidate["tweet_id"] in existing:
            records.append(existing[candidate["tweet_id"]])
        else:
            records.append(candidate)

    labeled_count = sum(
        1 for record in records
        if record.get("intent")
    )

    print("=== EVALUATION ANNOTATION ===")
    print(f"Total candidates : {len(records)}")
    print(f"Already labeled  : {labeled_count}")
    print(f"Remaining        : {len(records) - labeled_count}")

    for index, record in enumerate(records):

        if record.get("intent"):
            continue

        print("\n" + "=" * 80)
        print(f"Message {index + 1}/{len(records)}")
        print(f"Tweet ID : {record['tweet_id']}")
        print(f"Conversation : {record['conversation_id']}")
        print("-" * 80)
        print(record["text"])
        print("-" * 80)

        print_intents()

        while True:
            choice = input("\nIntent number (q = quit): ").strip()

            if choice.lower() == "q":
                save_labels(records)

                print("\nProgress saved.")
                print(
                    f"Labeled: "
                    f"{sum(1 for r in records if r.get('intent'))}"
                    f"/{len(records)}"
                )

                return

            if not choice.isdigit():
                print("Enter a number from 1-10.")
                continue

            choice = int(choice)

            if not 1 <= choice <= len(INTENTS):
                print("Enter a number from 1-10.")
                continue

            record["intent"] = INTENTS[choice - 1]

            save_labels(records)

            print(f"✓ {record['intent']}")
            break

    print("\n" + "=" * 80)
    print("ANNOTATION COMPLETE")
    print("=" * 80)
    print(f"Total labeled : {len(records)}")
    print(f"Output        : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()