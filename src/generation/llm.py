"""
LLM Service
============
Interfaces with the **Ollama** local API to generate grounded answers
based on retrieved context chunks.
"""

import logging
from typing import Optional

import ollama

from src.utils.errors import LLMServiceError
from src.core.config import settings

logger = logging.getLogger(__name__)

# System-level instruction constraining the model to the provided context
_SYSTEM_PROMPT: str = (
    "You are a precise document assistant. "
    "Answer the user's question using ONLY the context provided below. "
    "If the answer is not contained in the context, respond with: "
    "'The requested information is not found in the uploaded documents.' "
    "When answering, cite the relevant source passages. "
    "Be concise, accurate, and factual."
)


class LLMService:
    """
    Sends context-augmented prompts to Ollama and returns the generated answer.
    """

    def __init__(self) -> None:
        try:
            self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)
            self.model = settings.MODEL_NAME
            logger.info("Ollama client initialised (model=%s, url=%s)", self.model, settings.OLLAMA_BASE_URL)
        except Exception as exc:
            logger.error("Failed to initialise Ollama client: %s", exc)
            raise LLMServiceError(f"Ollama client init failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    #  Answer Generation
    # ------------------------------------------------------------------ #

    def generate_answer(self, question: str, context: str) -> str:
        """
        Call the Ollama API with the user's *question* and retrieved *context*.

        The system prompt forces the model to answer **only** from the
        supplied context and to cite sources.

        Args:
            question: The user's natural-language question.
            context:  Concatenated text passages retrieved from FAISS.

        Returns:
            The generated answer string.

        Raises:
            LLMServiceError: If the Ollama API call fails.
        """
        user_message: str = (
            f"Context:\n"
            f"---\n"
            f"{context}\n"
            f"---\n\n"
            f"Question: {question}"
        )

        logger.info(
            "Calling Ollama API (model=%s) — question length=%d, context length=%d",
            self.model,
            len(question),
            len(context),
        )

        try:
            # We use chat since it maintains dialogue structure via role messages
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            
            # Extract content from response dictionary safely
            answer: str = response.get('message', {}).get('content', '')

            # Extract token details for logging
            input_tokens = response.get('prompt_eval_count', 0)
            output_tokens = response.get('eval_count', 0)
            logger.info(
                "Ollama response received — input_tokens=%d, output_tokens=%d",
                input_tokens,
                output_tokens,
            )

            return answer

        except Exception as exc:
            logger.error("Error during Ollama call: %s", exc)
            raise LLMServiceError(
                f"Failed to get a response from Ollama: {exc}"
            ) from exc
