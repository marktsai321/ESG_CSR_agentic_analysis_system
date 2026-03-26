from __future__ import annotations

"""Unified LLM chat completion — works with both OpenAI and Anthropic keys."""

from esg_csr_agent.config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL_NAME,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL_NAME,
    GEMINI_API_KEY,
    REVISION_MODEL_NAME,
)


def chat_completion(prompt: str, temperature: float = 0.2, max_tokens: int = 2000) -> str:
    """Send a single user message and return the assistant's text response."""
    if LLM_PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=ANTHROPIC_MODEL_NAME,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    else:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content


def gemini_json_completion(prompt: str, system_instruction: str = "") -> dict:
    """Call Gemini for JSON output, primarily for revision stage."""
    import json
    import google.generativeai as genai

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY 未設定，無法執行 Gemini 修訂。")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=REVISION_MODEL_NAME,
        system_instruction=system_instruction,
        generation_config={"response_mime_type": "application/json"},
    )
    response = model.generate_content(prompt)
    text = (response.text or "").strip()
    if not text:
        return {}
    return json.loads(text)
