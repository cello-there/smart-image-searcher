from __future__ import annotations
import json
import os
from typing import Optional, Dict, Any

import requests

# Optional providers (commented libs remain optional if you enable them)
try:
    from anthropic import Anthropic  # type: ignore
except Exception:
    Anthropic = None

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None


def _cfg_get(cfg: Optional[dict], key: str, default=None):
    if cfg and key in cfg:
        return cfg[key]
    return os.getenv(key.upper(), default)


def _pick_provider(cfg: Optional[dict] = None):
    # Prefer explicit config; fall back to env vars
    prov = (cfg or {}).get("llm_provider") or os.getenv("LLM_PROVIDER")
    model = (cfg or {}).get("llm_model") or os.getenv("LLM_MODEL")
    if prov:
        return prov.lower(), model
    # Auto-detect based on available keys/libs
    if os.getenv("ANTHROPIC_API_KEY") and Anthropic is not None:
        return "anthropic", os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
    if os.getenv("OPENAI_API_KEY") and OpenAI is not None:
        return "openai", os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # Default to local ollama
    return "ollama", (model or "mistral")


def _ollama_chat(model: str, prompt: str, *, system: Optional[str] = None, temperature: float = 0.2, max_tokens: int = 512, host: Optional[str] = None) -> str:
    host = host or os.getenv("OLLAMA_HOST", (cfg_host := ("http://localhost:11434")))
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [],
        "stream": False,
        "format": "json",  # enforce JSON mode
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if system:
        payload["messages"].append({"role": "system", "content": system})
    payload["messages"].append({"role": "user", "content": prompt})
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return (data.get("message") or {}).get("content", "")


def complete_json(prompt: str, schema: Dict[str, Any], temperature: float = 0.2, max_tokens: int = 512, system: Optional[str] = None, provider: Optional[str] = None, model: Optional[str] = None, cfg: Optional[dict] = None) -> Optional[Dict[str, Any]]:
    """Call the configured LLM and parse a JSON object. Returns None on failure.

    Provider priority:
      1) Explicit args `provider`/`model`
      2) `config.json` keys (llm_provider, llm_model)
      3) Env vars / available SDKs
      4) Default: Ollama + `mistral`
    """
    prov = (provider or (cfg or {}).get("llm_provider") or os.getenv("LLM_PROVIDER") or "").lower()
    mdl = model or (cfg or {}).get("llm_model") or os.getenv("LLM_MODEL")

    if not prov:
        prov, mdl = _pick_provider(cfg)

    try:
        if prov == "ollama":
            text = _ollama_chat(
                mdl or "mistral",
                prompt,
                system=system or "You are a careful assistant. Output only valid JSON that matches the requested schema.",
                temperature=temperature,
                max_tokens=max_tokens,
                host=(cfg or {}).get("ollama_host") or os.getenv("OLLAMA_HOST"),
            )
            return _safe_json(text, schema)
        elif prov == "anthropic" and Anthropic is not None and os.getenv("ANTHROPIC_API_KEY"):
            client = Anthropic()
            msg = client.messages.create(
                model=mdl or os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest"),
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "You are a careful assistant. Output only valid JSON that matches the requested schema.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join([b.text for b in msg.content if hasattr(b, "text")])
            return _safe_json(text, schema)
        elif prov == "openai" and OpenAI is not None and os.getenv("OPENAI_API_KEY"):
            client = OpenAI()
            resp = client.chat.completions.create(
                model=mdl or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                response_format={"type": "json_object"},
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system or "You are a careful assistant. Output only valid JSON that matches the requested schema."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content
            return _safe_json(text, schema)
        else:
            return None
    except Exception:
        return None


def _safe_json(text: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(text)
    except Exception:
        # try to extract JSON substring
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start:end+1])
        except Exception:
            return None
    # Optional: schema validation if jsonschema is installed
    try:
        import jsonschema  # type: ignore
        jsonschema.validate(data, schema)
    except Exception:
        pass
    return data