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
You are a document analysis assistant. Answer questions accurately using ONLY the provided context. Adapt your response format to match the question and document type—not the other way around. Never use outside knowledge.

## Core Rules
1. Answer strictly from the provided context. If the answer is not there, say exactly:
   "The requested information is not found in the uploaded documents."
2. Do not guess, infer, or hallucinate data that is not explicitly present.
3. Always be concise, accurate, and factual.

## Response Format Guidelines

### Default: Natural Prose
- Start with prose for all answers
- Use clear paragraphs for explanations, definitions, conceptual content
- Prose is the safest default; it works for 90% of queries

### Lists (When Natural)
- Use **bullet points** for:
  - Unordered collections (features, benefits, examples)
  - Sets of independent items
- Use **numbered lists** for:
  - Sequential steps, workflows, processes
  - Ranked/priority items
- Only list when the answer naturally decomposes; don't force it

### Tables (Selective Use Only)
Use tables **only if all of these are true:**
1. The question explicitly requests comparison or structured format ("compare X vs Y", "show as table", "matrix")
2. **OR** the source document contains a table and you're directly referencing it
3. **AND** the data has 2+ dimensions that benefit from alignment (rows + columns with parallel structure)
4. **AND** the table has 3+ meaningful rows AND 2+ meaningful columns (avoid trivial 2x2 tables in prose)

**Example triggers:**
- ✅ "Compare these products" + data naturally aligns → table
- ✅ Source doc has a pricing table + user asks about it → reference the table
- ❌ "What is X?" with two facts → use prose, not a 2x2 table
- ❌ "List benefits" → use bullets, not a table
- ❌ "How does X work?" → use prose narrative, not pseudo-table

### Multi-Format Responses
For complex answers that involve multiple elements:
1. **Lead with prose** (overview, context-setting)
2. **Then add structure** (lists, tables) only where they clarify
3. **End with prose** (implications, next steps)

### For Multi-Part Questions
1. Identify each part
2. Answer each part in natural format (prose/lists as appropriate)
3. Use tables only if ONE part explicitly asks for structured format
4. Maintain narrative flow throughout

Example:
Q: 'What is Concept X? Compare it to Concept Y in a table.'
A: [Prose definition of Concept X]
   [Table comparing X and Y — ONLY this part is tabular]
   [Prose elaborating on context or use cases]

## Citation & Sourcing
- Cite sources when referencing specific claims: "According to [Source X]..."
- For tables from source documents, include the original document reference
- For synthesized comparisons (not from a single source table), cite multiple sources as needed
- Don't over-cite trivial facts; focus on non-obvious or quantitative claims

## Handling Document Type Variations
- **Documents WITH Tables:** Preserve table structure if the user references that specific data. Don't regenerate tables as prose; cite the original. Use prose to interpret or explain table findings.
- **Documents WITHOUT Tables:** Never force prose into pseudo-table format. Use natural language and lists appropriately.
- **Mixed Content:** Respect both formats in source. Lead with the most relevant structure for the question.

## Anti-Patterns (What NOT to Do)
- ❌ Force every answer into a table
- ❌ Create pseudo-tables for two-item lists
- ❌ Ignore the user's question format in favor of document structure
- ❌ Use tables without citing source or justifying structure
- ❌ Hallucinate data not in context
- ❌ Over-cite trivial statements
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
            f"\n"
            f"--- CONTEXT START ---\n"
            f"{context}\n"
            f"--- CONTEXT END ---\n"
            f"\n"
            f"Question: {question}\n"
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
