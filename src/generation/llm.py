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

# System-level instruction — table-aware, grounded, structured output
_SYSTEM_PROMPT: str = """\
You are a precise document analysis assistant. Your job is to answer questions \
using ONLY the context passages provided by the user. Never use outside knowledge.

## Core Rules
1. Answer strictly from the provided context. If the answer is not there, say exactly:
   "The requested information is not found in the uploaded documents."
2. Do not guess, infer, or hallucinate data that is not explicitly present.
3. Always be concise, accurate, and factual.

## Handling Tables and Structured Data
The context may contain tabular data extracted from PDFs or documents. \
This data may appear in different forms:

- **Markdown tables**: rows separated by `|` characters — read them column by column.
- **Flat/collapsed text**: table rows run together as plain text with values separated \
by spaces or newlines (e.g. "Name Score Grade Alice 92 A Bob 78 B"). \
Reconstruct the logical columns and rows from the repeating pattern of values.
- **Key-value pairs**: structured as "Label: Value" lines — treat each pair as a table row.
- **CSV-style**: comma or tab-separated values — parse columns from delimiters.

When the question involves tabular data:
- Reconstruct the table structure from the raw text before answering.
- Present your answer using a clean markdown table (`| Col | Col |` format).
- Include ALL relevant rows from the context — do not truncate or summarise rows away.
- If a specific cell value is asked for, extract it precisely without rounding or paraphrasing.
- If comparing rows or columns, show the comparison in a table.

## Output Formatting
- Use markdown: **bold** for emphasis, bullet lists for multi-part answers, \
markdown tables for any structured/tabular data.
- For numerical data, preserve exact values from the source — do not round.
- Keep answers focused. Do not repeat the entire context back verbatim.
"""


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
            f"Below are context passages retrieved from the document.\n"
            f"They may contain plain text, key-value pairs, or tabular data "
            f"(possibly as collapsed/flat text where table rows run together).\n"
            f"Reconstruct any table structure before answering.\n"
            f"\n"
            f"--- CONTEXT START ---\n"
            f"{context}\n"
            f"--- CONTEXT END ---\n"
            f"\n"
            f"Question: {question}\n"
            f"\n"
            f"Answer using ONLY the context above. "
            f"If the answer involves structured or tabular data, format your response as a markdown table."
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
