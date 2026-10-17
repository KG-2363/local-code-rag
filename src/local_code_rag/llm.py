"""Grounded chat with a local Ollama model."""

from __future__ import annotations

from collections.abc import Iterator

SYSTEM_PROMPT = """You are a senior engineer helping a developer understand the codebase "{name}".{description}

You are given code retrieved from that codebase. Rules:
- Answer only from the provided context. If it isn't there, say you couldn't find it; never invent
  file names, functions, tables or signatures.
- Cite every claim with file:line exactly as it appears in the context headers.
- For "where is X defined" questions, give the location first, then a short explanation.
- Prefer short answers with code excerpts over long prose.
"""

MAX_HISTORY_MESSAGES = 6


def build_messages(question: str, context: str, project: str, description: str = "",
                   history: list[dict] | None = None) -> list[dict]:
    desc = f"\nProject description: {description.strip()}" if description.strip() else ""
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(name=project, description=desc)}]
    messages += (history or [])[-MAX_HISTORY_MESSAGES:]
    user = question if not context else f"Context from the codebase:\n\n{context}\n\n---\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user})
    return messages


class OllamaChat:
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        import ollama

        self.model = model
        self._client = ollama.Client(host=base_url)

    def stream(self, messages: list[dict]) -> Iterator[str]:
        for chunk in self._client.chat(model=self.model, messages=messages, stream=True,
                                       options={"temperature": 0.1}):
            if chunk.message.content:
                yield chunk.message.content

    def available_models(self) -> list[str]:
        try:
            return [m.model for m in self._client.list().models]
        except Exception:
            return []
