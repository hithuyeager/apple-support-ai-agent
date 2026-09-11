from src.retrival.tfidf_retriever import TfidfRetriever


def main():
    retriever = TfidfRetriever()

    query = (
        "My iPhone cannot activate. "
        "The activation server is unavailable."
    )

    results = retriever.search(
        query,
        top_k=5,
    )

    print("\n=== RETRIEVAL TEST ===")
    print(f"Query: {query}\n")

    for rank, result in enumerate(results, start=1):
        print(f"--- Result {rank} ---")

        print(
        f"Customer similarity: "
        f"{result['customer_similarity']:.4f}"
        )

        print(
        f"Response similarity: "
        f"{result['response_similarity']:.4f}"
        )

        print(
        f"Conversation: "
        f"{result['conversation_id']}"
        )

        print(
        "Matched customer:"
        )

        print(
        result["matched_customer"]["text"]
        )

        print(
        "\nHistorical Apple Support:"
        )

        print(
        result["support_response"]["text"]
        )

        print()
if __name__ == "__main__":
    main()