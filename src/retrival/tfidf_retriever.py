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
    def __init__(
        self,
        customer_file: Path = CUSTOMER_FILE,
        support_file: Path = SUPPORT_FILE,
    ):
        self.customer_file = customer_file
        self.support_file = support_file

        self.customer_messages = []
        self.support_by_conversation = {}

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )

        self.customer_matrix = None

        self._load_data()
        self._build_index()

    def _load_data(self):
        # Load historical customer messages
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

        # Load historical Apple Support responses
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

    def _build_index(self):
        texts = [
            message["text"]
            for message in self.customer_messages
        ]

        self.customer_matrix = (
            self.vectorizer.fit_transform(texts)
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        responses_per_conversation: int = 3,
    ):
        """
        Retrieve historically similar customer problems
        and rank Apple Support responses from the
        corresponding conversations.
        """

        if not query.strip():
            return []

        # -------------------------------------------------
        # Stage 1:
        # Find historically similar customer messages
        # -------------------------------------------------

        query_vector = self.vectorizer.transform(
            [query]
        )

        similarities = cosine_similarity(
            query_vector,
            self.customer_matrix,
        )[0]

        ranked_indices = similarities.argsort()[::-1]

        conversation_matches = []
        seen_conversations = set()

        for index in ranked_indices:
            similarity = float(
                similarities[index]
            )

            if similarity <= 0:
                break

            customer = self.customer_messages[index]

            conversation_id = customer[
                "conversation_id"
            ]

            # Only keep one representative customer
            # message from each conversation.
            if conversation_id in seen_conversations:
                continue

            seen_conversations.add(
                conversation_id
            )

            conversation_matches.append(
                {
                    "customer_similarity": similarity,
                    "conversation_id": conversation_id,
                    "matched_customer": customer,
                }
            )

            if len(conversation_matches) >= top_k:
                break

        # -------------------------------------------------
        # Stage 2:
        # Rank support responses inside each matched
        # conversation.
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
                    matched_customer=(
                        match[
                            "matched_customer"
                        ]["text"]
                    ),
                    support_responses=(
                        support_responses
                    ),
                    top_k=responses_per_conversation,
                )
            )

            for response in response_results:
                results.append(
                    {
                        "customer_similarity": (
                            match[
                                "customer_similarity"
                            ]
                        ),
                        "response_similarity": (
                            response[
                                "similarity"
                            ]
                        ),
                        "conversation_id": (
                            conversation_id
                        ),
                        "matched_customer": (
                            match[
                                "matched_customer"
                            ]
                        ),
                        "support_response": (
                            response[
                                "response"
                            ]
                        ),
                    }
                )

        # -------------------------------------------------
        # Final ranking
        # -------------------------------------------------

        results.sort(
            key=lambda result: (
                result[
                    "customer_similarity"
                ]
                * result[
                    "response_similarity"
                ]
            ),
            reverse=True,
        )

        return results

    def _rank_support_responses(
        self,
        query: str,
        matched_customer: str,
        support_responses: list,
        top_k: int,
    ):
        """
        Rank Apple Support responses using TF-IDF
        similarity against the incoming query plus
        the matched historical customer message.
        """

        if not support_responses:
            return []

        texts = [
            response["text"]
            for response in support_responses
        ]

        response_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )

        response_matrix = (
            response_vectorizer.fit_transform(
                texts
            )
        )

        combined_query = (
            f"{query} {matched_customer}"
        )

        query_vector = (
            response_vectorizer.transform(
                [combined_query]
            )
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
                    "similarity": float(
                        similarities[index]
                    ),
                    "response": (
                        support_responses[index]
                    ),
                }
            )

        return results