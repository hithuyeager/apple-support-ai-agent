import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CUSTOMER_FILE = Path(
    "data/processed/customer_messages.jsonl"
)

SUPPORT_FILE = Path(
    "data/processed/support_responses.jsonl"
)


class TfidfRetriever:
    """
    Resolution-oriented TF-IDF retriever.

    The historical corpus is noisy, so retrieval is performed in
    two stages:

    1. Find historically similar customer messages.
    2. Rank the support responses associated with those conversations
       primarily by how relevant the response is to the current query.

    Unlike v1, customer similarity is NOT allowed to dominate the
    final evidence score.
    """

    def __init__(
        self,
        customer_file: Path = CUSTOMER_FILE,
        support_file: Path = SUPPORT_FILE,
    ):
        self.customer_file = customer_file
        self.support_file = support_file

        self.customer_messages = []
        self.support_by_conversation = {}

        self.customer_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )

        self.customer_matrix = None

        self._load_data()
        self._build_customer_index()

    # ---------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------

    def _load_data(self):
        with self.customer_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                if not line.strip():
                    continue

                self.customer_messages.append(
                    json.loads(line)
                )

        with self.support_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                if not line.strip():
                    continue

                response = json.loads(line)

                conversation_id = response[
                    "conversation_id"
                ]

                self.support_by_conversation.setdefault(
                    conversation_id,
                    [],
                ).append(response)

    # ---------------------------------------------------------
    # Customer index
    # ---------------------------------------------------------

    def _build_customer_index(self):
        texts = [
            message["text"]
            for message in self.customer_messages
        ]

        self.customer_matrix = (
            self.customer_vectorizer.fit_transform(
                texts
            )
        )

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        responses_per_conversation: int = 3,
    ):
        """
        Retrieve historical support resolutions relevant to query.
        """

        if not query.strip():
            return []

        # -------------------------------------------------
        # Stage 1:
        # Find similar historical customer messages.
        # -------------------------------------------------

        query_vector = (
            self.customer_vectorizer.transform(
                [query]
            )
        )

        similarities = cosine_similarity(
            query_vector,
            self.customer_matrix,
        )[0]

        ranked_indices = similarities.argsort()[::-1]

        conversation_matches = []
        seen_conversations = set()

        for index in ranked_indices:
            customer_similarity = float(
                similarities[index]
            )

            if customer_similarity <= 0:
                break

            customer = self.customer_messages[index]

            conversation_id = customer[
                "conversation_id"
            ]

            if conversation_id in seen_conversations:
                continue

            seen_conversations.add(
                conversation_id
            )

            conversation_matches.append(
                {
                    "customer_similarity":
                        customer_similarity,
                    "conversation_id":
                        conversation_id,
                    "matched_customer":
                        customer,
                }
            )

            if len(conversation_matches) >= top_k:
                break

        # -------------------------------------------------
        # Stage 2:
        # Rank support responses.
        #
        # IMPORTANT:
        # The incoming query is compared directly against
        # the support response.
        #
        # This makes response relevance the main signal.
        # -------------------------------------------------

        results = []

        for match in conversation_matches:
            conversation_id = match[
                "conversation_id"
            ]

            support_responses = (
                self.support_by_conversation.get(
                    conversation_id,
                    [],
                )
            )

            response_results = (
                self._rank_support_responses(
                    query=query,
                    support_responses=support_responses,
                    top_k=responses_per_conversation,
                )
            )

            for response in response_results:
                response_similarity = float(
                    response["similarity"]
                )

                customer_similarity = float(
                    match["customer_similarity"]
                )

                # V2:
                #
                # Response relevance is dominant.
                # Customer similarity is only a supporting signal.
                #
                # 80% response relevance
                # 20% customer-problem similarity
                final_score = (
                    0.8 * response_similarity
                    + 0.2 * customer_similarity
                )

                results.append(
                    {
                        "customer_similarity":
                            customer_similarity,

                        "response_similarity":
                            response_similarity,

                        "final_score":
                            final_score,

                        "conversation_id":
                            conversation_id,

                        "matched_customer":
                            match[
                                "matched_customer"
                            ],

                        "support_response":
                            response[
                                "response"
                            ],
                    }
                )

        # -------------------------------------------------
        # Remove duplicate support responses.
        # -------------------------------------------------

        unique_results = []
        seen_responses = set()

        for result in results:
            text = (
                result[
                    "support_response"
                ]["text"]
                .strip()
                .lower()
            )

            if text in seen_responses:
                continue

            seen_responses.add(text)
            unique_results.append(result)

        # -------------------------------------------------
        # Final ranking.
        # -------------------------------------------------

        unique_results.sort(
            key=lambda result: result[
                "final_score"
            ],
            reverse=True,
        )

        return unique_results

    # ---------------------------------------------------------
    # Support-response ranking
    # ---------------------------------------------------------

    def _rank_support_responses(
        self,
        query: str,
        support_responses: list,
        top_k: int,
    ):
        """
        Rank historical Apple Support responses directly
        against the incoming customer query.
        """

        if not support_responses:
            return []

        texts = [
            response["text"]
            for response in support_responses
        ]

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )

        response_matrix = (
            vectorizer.fit_transform(texts)
        )

        query_vector = vectorizer.transform(
            [query]
        )

        similarities = cosine_similarity(
            query_vector,
            response_matrix,
        )[0]

        ranked_indices = (
            similarities.argsort()[::-1]
        )

        results = []

        for index in ranked_indices[:top_k]:
            results.append(
                {
                    "similarity":
                        float(
                            similarities[index]
                        ),
                    "response":
                        support_responses[index],
                }
            )

        return results