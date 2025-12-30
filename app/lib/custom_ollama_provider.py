from typing import AsyncIterator, Dict, Any, List, Optional
from ai_sdk.providers.language_model import LanguageModel
from ollama import AsyncClient
from ai_sdk.types import CoreUserMessage, CoreAssistantMessage, CoreSystemMessage

class CustomOllamaProvider(LanguageModel):
    """
    Adapter to allow ai-sdk-python to talk to the native Ollama API (non-OpenAI).
    """
    def __init__(self, model: str, host: str, headers: Optional[Dict[str, str]] = None):
        self.model = model
        self.client = AsyncClient(host=host, headers=headers)

    def _convert_messages(self, messages: List[Any]) -> List[Dict[str, str]]:
        """Convert ai_sdk CoreMessage objects to Ollama native message dicts."""
        converted = []
        if not messages:
            return converted

        for m in messages:
            role = "user"
            content = ""
            
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
            elif isinstance(m, CoreSystemMessage):
                role = "system"
                content = m.content
            elif isinstance(m, CoreUserMessage):
                role = "user"
                if isinstance(m.content, str):
                    content = m.content
                else:
                    content = str(m.content)
            elif isinstance(m, CoreAssistantMessage):
                role = "assistant"
                if isinstance(m.content, str):
                    content = m.content
                else:
                    content = str(m.content)
            
            converted.append({"role": role, "content": content})
        return converted

    def generate_text(
        self,
        *,
        prompt: str | None = None,
        system: str | None = None,
        messages: list[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Synchronous-style generation (not used by stream_text usually)."""
        raise NotImplementedError("This custom provider is optimized for streaming.")

    async def stream_text(
        self,
        *,
        prompt: str | None = None,
        system: str | None = None,
        messages: list[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Yield text deltas from native Ollama stream."""
        
        # Merge system prompt if provided separately
        final_messages = self._convert_messages(messages or [])
        if system:
            final_messages.insert(0, {"role": "system", "content": system})
        if prompt:
            final_messages.append({"role": "user", "content": prompt})

        import logging
        logger = logging.getLogger("chat_api")
        
        try:
            # Call native Ollama API
            async for chunk in await self.client.chat(
                model=self.model,
                messages=final_messages,
                stream=True,
            ):
                if "message" in chunk and "content" in chunk["message"]:
                    token = chunk["message"]["content"]
                    logger.info(f"Ollama Chunk: {token}")
                    yield token
        except Exception as e:
            logger.error(f"CustomOllamaProvider Error: {e}", exc_info=True)
            raise e
