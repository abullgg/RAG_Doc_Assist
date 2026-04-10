"""
LLM Service
============
Interfaces with the **Anthropic Claude** API to generate grounded answers
based on retrieved context chunks.
"""

import logging
import os
from typing import Optional

import anthropic
from dotenv import load_dotenv

# Load .env so ANTHROPIC_API_KEY is available
load_dotenv()

logger = logging.getLogger(__name__)

# Claude model to use for answer generation
_MODEL_ID: str = "claude-3-5-sonnet-20241022"
_MAX_TOKENS: int = 1024

# System-level instruction that constrains the model to the provided context
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
    Sends context-augmented prompts to Claude and returns the generated answer.
    """

    def __init__(self) -> None:
        api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            logger.error(
                "ANTHROPIC_API_KEY is not set. "
                "Set it in the environment or in a .env file."
            )
            raise EnvironmentError(
                "Missing ANTHROPIC_API_KEY environment variable. "
                "Please set it before starting the server."
            )

        self.client: anthropic.Anthropic = anthropic.Anthropic(api_key=api_key)
        logger.info("Anthropic client initialised (model=%s)", _MODEL_ID)

    # ------------------------------------------------------------------ #
    #  Answer Generation
    # ------------------------------------------------------------------ #

    def generate_answer(self, question: str, context: str) -> str:
        """
        Call the Claude API with the user's *question* and retrieved *context*.

        The system prompt forces the model to answer **only** from the
        supplied context and to cite sources.

        Args:
            question: The user's natural-language question.
            context:  Concatenated text passages retrieved from FAISS.

        Returns:
            The generated answer string.

        Raises:
            RuntimeError: If the Anthropic API call fails.
        """
        # Compose the user message with context + question
        user_message: str = (
            f"Context:\n"
            f"---\n"
            f"{context}\n"
            f"---\n\n"
            f"Question: {question}"
        )

        logger.info(
            "Calling Claude API — question length=%d, context length=%d",
            len(question),
            len(context),
        )

        try:
            response = self.client.messages.create(
                model=_MODEL_ID,
                max_tokens=_MAX_TOKENS,
                system=_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_message},
                ],
            )

            # Extract the text content from the response
            answer: str = response.content[0].text

            # Log token usage for cost tracking / debugging
            usage = response.usage
            logger.info(
                "Claude response received — input_tokens=%d, output_tokens=%d",
                usage.input_tokens,
                usage.output_tokens,
            )

            return answer

        except anthropic.APIError as exc:
            logger.error("Anthropic API error: %s", exc)
            raise RuntimeError(
                f"Failed to get a response from Claude: {exc}"
            ) from exc
        except Exception as exc:
            logger.error("Unexpected error during LLM call: %s", exc)
            raise RuntimeError(
                f"Unexpected error generating answer: {exc}"
            ) from exc
