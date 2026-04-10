from anthropic import Anthropic
from src.core.config import settings

class LLMService:
    def __init__(self):
        self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    def generate_answer(self, prompt: str, context: str) -> str:
        # Placeholder for LLM call
        # message = self.client.messages.create(...)
        return "This is a placeholder answer from Claude."
