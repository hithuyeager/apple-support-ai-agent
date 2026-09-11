from src.agent import SupportAgent


def main():
    agent = SupportAgent()

    message = (
        "My iPhone cannot activate. "
        "The activation server is unavailable."
    )

    result = agent.handle(message)

    print("\n=== SUPPORT AGENT ===")
    print(f"Message    : {result['message']}")
    print(f"Intent     : {result['intent']}")
    print(f"Confidence : {result['confidence']:.2f}")

    print("\n=== TOP EVIDENCE ===")
    for index, evidence in enumerate(result["evidence"][:5], start=1):
        print(f"\n--- Evidence {index} ---")
        print(
            f"Final score: "
            f"{evidence['final_score']:.4f}"
        )
        print(
            "Historical customer:"
            f"\n{evidence['matched_customer']['text']}"
        )
        print(
            "Historical Apple Support:"
            f"\n{evidence['support_response']['text']}"
        )

    print("\n=== DRAFT RESPONSE ===")
    print(result["draft_response"])


if __name__ == "__main__":
    main()
