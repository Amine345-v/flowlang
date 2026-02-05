from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Optional

import time
import logging

from .types import TypedValue, ValueTag

# Optional third-party SDK imports guarded to avoid hard deps
try:
    from openai import OpenAI as _OpenAIClient  # type: ignore
except Exception:  # pragma: no cover
    _OpenAIClient = None

try:
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None

try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None

try:
    from mistralai.client import MistralClient  # type: ignore
    from mistralai.models.chat_completion import ChatMessage as _MistralMsg  # type: ignore
except Exception:  # pragma: no cover
    MistralClient = None
    _MistralMsg = None

try:
    import cohere as _cohere  # type: ignore
except Exception:  # pragma: no cover
    _cohere = None

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None


class ProviderError(Exception):
    pass


def _get_timeout_s(kwargs: Dict[str, Any], default: int = 60) -> int:
    try:
        v = kwargs.get("timeout_s", default)
        return int(v)
    except Exception:
        return default


def _get_retries(kwargs: Dict[str, Any], default: int = 2) -> int:
    try:
        v = kwargs.get("retries", default)
        return int(v)
    except Exception:
        return default


def _with_retries(fn, retries: int = 2, base_delay: float = 0.5):
    last_exc = None
    for i in range(retries + 1):
        try:
            return fn()
        except Exception as e:  # pragma: no cover - network variability
            last_exc = e
            if i == retries:
                break
            time.sleep(base_delay * (2 ** i))
    raise last_exc  # type: ignore


def _system_prompt(verb: str, maestro_path: Optional[str] = None) -> str:
    path_ctx = f"\n[Maestro Context: Binary Path {maestro_path}]" if maestro_path else ""
    if verb == "ask":
        return (
            "You are a helpful assistant. Respond with JSON: {\n"
            "  \"text\": string,\n  \"history\": array\n}" + path_ctx
        )
    if verb == "search":
        return (
            "You are an information retrieval agent. Respond with JSON: {\n"
            "  \"hits\": array of strings\n}" + path_ctx
        )
    if verb == "try":
        return (
            "Execute a task and report results as JSON: {\n"
            "  \"output\": string,\n  \"metrics\": {\"time\": number}\n}" + path_ctx
        )
    if verb == "judge":
        return (
            "Evaluate and respond with JSON: {\n"
            "  \"score\": number,\n  \"confidence\": number,\n  \"pass\": boolean\n}" + path_ctx
        )
    return "You are an AI that executes commands. Prefer JSON outputs." + path_ctx


def _build_user_payload(team: str, verb: str, args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if verb == "ask":
        prompt = args[0] if args else kwargs.get("prompt", "")
        if hasattr(prompt, "payload"):
            prompt = prompt.payload
        history = kwargs.get("history", [])
        return {
            "verb": verb,
            "team": team,
            "prompt": str(prompt),
            "history": list(history) if isinstance(history, list) else [],
            "options": {k: v for k, v in kwargs.items() if k not in ("prompt", "history")},
        }
    if verb == "search":
        query = args[0] if args else kwargs.get("query", "")
        if hasattr(query, "payload"):
            query = query.payload
        return {"verb": verb, "team": team, "query": str(query), "options": kwargs}
    if verb == "try":
        task = args[0] if args else kwargs.get("task", "")
        if hasattr(task, "payload"):
            task = task.payload
        return {"verb": verb, "team": team, "task": str(task), "options": kwargs}
    if verb == "judge":
        target = args[0] if args else kwargs.get("target", "")
        if hasattr(target, "payload"):
            target = target.payload
        criteria = args[1] if len(args) > 1 else kwargs.get("criteria", "score")
        if hasattr(criteria, "payload"):
            criteria = criteria.payload
        return {"verb": verb, "team": team, "target": target, "criteria": criteria, "options": kwargs}
    return {"verb": verb, "team": team, "args": args, "options": kwargs}


def _map_to_typed_value(verb: str, content: str, parsed: Optional[Dict[str, Any]], kwargs: Dict[str, Any]) -> TypedValue:
    """Map AI response to TypedValue with strict schema validation.
    
    Raises SchemaValidationError if the response doesn't match the expected schema.
    """
    from .schemas import validate_response, JudgeResult, SearchResult, TryResult, CommunicateResult
    
    # Build data dict for validation
    if parsed is None:
        # If we couldn't parse JSON, construct minimal data from raw content
        if verb == "ask":
            data = {"text": content, "history": kwargs.get("history", [])}
        elif verb == "search":
            data = {"hits": [content] if content else []}
        elif verb == "try":
            data = {"output": content, "metrics": {}}
        elif verb == "judge":
            # Raw content for judge is a validation failure - we need structured data
            from .errors import SchemaValidationError
            raise SchemaValidationError(
                f"AI response for 'judge' must be valid JSON with score/confidence/pass fields.\n"
                f"Received raw content: {content[:200]}..."
            )
        else:
            data = {"output": content, "metrics": {}}
    else:
        data = parsed
    
    # Validate and get typed model
    validated = validate_response(verb, data)
    
    # Convert to TypedValue
    if verb == "ask":
        return TypedValue(
            tag=ValueTag.CommunicateResult,
            meta={"text": validated.text, "history": validated.history}
        )
    if verb == "search":
        return TypedValue(
            tag=ValueTag.SearchResult,
            meta={"hits": validated.hits}
        )
    if verb == "try":
        return TypedValue(
            tag=ValueTag.TryResult,
            meta={"output": validated.output, "metrics": validated.metrics}
        )
    if verb == "judge":
        return TypedValue(
            tag=ValueTag.JudgeResult,
            meta={"score": validated.score, "confidence": validated.confidence, "pass": validated.pass_result}
        )
    
    # Unknown verb - use TryResult as fallback
    return TypedValue(tag=ValueTag.Unknown, meta={"text": content})


class AIProvider:
    name: str = "base"

    def execute(self, team: str, verb: str, args: List[Any], kwargs: Dict[str, Any]) -> TypedValue:  # pragma: no cover - abstract
        raise NotImplementedError


class OpenAIProvider(AIProvider):
    name = "openai"

    def __init__(self) -> None:
        if not _OpenAIClient or not os.getenv("OPENAI_API_KEY"):
            raise ProviderError("OpenAI not available: missing client or OPENAI_API_KEY")
        self.client = _OpenAIClient()

    def execute(self, team: str, verb: str, args: List[Any], kwargs: Dict[str, Any]) -> TypedValue:
        self.ai_default_model = os.getenv("FLOWLANG_AI_MODEL", "gpt-4o")
        model = (
            os.getenv(f"FLOWLANG_AI_MODEL_{verb.upper()}")
            or self.ai_default_model
        )
        user_payload = _build_user_payload(team, verb, args, kwargs)
        timeout_s = _get_timeout_s(kwargs, 60)
        network_retries = _get_retries(kwargs, 2)
        schema_retries = 2

        maestro_path = kwargs.get("maestro_path")
        # Initial messages
        messages = [
            {"role": "system", "content": _system_prompt(verb, maestro_path)},
            {"role": "user", "content": json.dumps(user_payload)},
        ]

        last_error = None
        content = ""

        # Schema correction loop
        for attempt in range(schema_retries + 1):
            try:
                def _call():
                    if kwargs.get("stream"):
                        content_buf: List[str] = []
                        stream = self.client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=float(kwargs.get("temperature", 0.7)),
                            max_tokens=int(kwargs.get("max_tokens", 1000)) if str(kwargs.get("max_tokens", "")).isdigit() else 1000,
                            stream=True,
                            timeout=timeout_s,
                        )
                        for ev in stream:
                            try:
                                delta = ev.choices[0].delta  # type: ignore[attr-defined]
                                if hasattr(delta, "content") and delta.content:
                                    content_buf.append(delta.content)
                            except Exception:
                                pass
                        return "".join(content_buf)
                    else:
                        resp = self.client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=float(kwargs.get("temperature", 0.7)),
                            max_tokens=int(kwargs.get("max_tokens", 1000)) if str(kwargs.get("max_tokens", "")).isdigit() else 1000,
                            timeout=timeout_s,
                        )
                        return resp.choices[0].message.content if resp and resp.choices else ""

                content = _with_retries(_call, retries=network_retries)
                
                # Parse and validate
                try:
                    parsed = json.loads(content) if content else {}
                except Exception:
                    parsed = None
                
                return _map_to_typed_value(verb, content, parsed, kwargs)

            except SchemaValidationError as e:
                last_error = e
                # Don't retry if it was the last attempt
                if attempt == schema_retries:
                    break
                
                # Append error to messages for correction
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "system", 
                    "content": f"ERROR: Your response failed validation: {str(e)}\n"
                               f"Please CORRECT your JSON output to match the required schema."
                })
            except Exception as e:
                return TypedValue(tag=ValueTag.Unknown, meta={"error": str(e), "provider": self.name})

        # Final failure after retries
        return TypedValue(
            tag=ValueTag.Unknown, 
            meta={"error": f"Schema validation failed after {schema_retries} retries. Last error: {last_error}", "content": content}
        )


class AnthropicProvider(AIProvider):
    name = "anthropic"

    def __init__(self) -> None:
        if not anthropic or not os.getenv("ANTHROPIC_API_KEY"):
            raise ProviderError("Anthropic not available")
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def execute(self, team: str, verb: str, args: List[Any], kwargs: Dict[str, Any]) -> TypedValue:
        model = (
            os.getenv(f"FLOWLANG_ANTHROPIC_MODEL_{verb.upper()}")
            or os.getenv("FLOWLANG_ANTHROPIC_MODEL")
            or "claude-sonnet-5"
        )
        user_payload = _build_user_payload(team, verb, args, kwargs)
        timeout_s = _get_timeout_s(kwargs, 60)
        retries = _get_retries(kwargs, 2)
        try:
            def _call():
                if kwargs.get("stream"):
                    content_buf: List[str] = []
                    stream = self.client.messages.create(
                        model=model,
                        max_tokens=int(kwargs.get("max_tokens", 1000)) if str(kwargs.get("max_tokens", "")).isdigit() else 1000,
                        temperature=float(kwargs.get("temperature", 0.7)),
                        system=_system_prompt(verb),
                        messages=[{"role": "user", "content": json.dumps(user_payload)}],
                        stream=True,
                        timeout=timeout_s,
                    )
                    for ev in stream:
                        try:
                            if hasattr(ev, "delta") and hasattr(ev.delta, "text"):
                                content_buf.append(ev.delta.text)
                        except Exception:
                            pass
                    return "".join(content_buf)
                else:
                    msg = self.client.messages.create(
                        model=model,
                        max_tokens=int(kwargs.get("max_tokens", 1000)) if str(kwargs.get("max_tokens", "")).isdigit() else 1000,
                        temperature=float(kwargs.get("temperature", 0.7)),
                        system=_system_prompt(verb),
                        messages=[{"role": "user", "content": json.dumps(user_payload)}],
                        timeout=timeout_s,
                    )
                    return "".join(part.text for part in msg.content if getattr(part, "type", "") == "text")

            content = _with_retries(_call, retries=retries)
        except Exception as e:
            return TypedValue(tag=ValueTag.Unknown, meta={"error": str(e), "provider": self.name})
        try:
            parsed = json.loads(content) if content else {}
        except Exception:
            parsed = None
        return _map_to_typed_value(verb, content, parsed, kwargs)


class GeminiProvider(AIProvider):
    name = "gemini"

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not genai or not api_key:
            raise ProviderError("Gemini not available")
        genai.configure(api_key=api_key)

    def execute(self, team: str, verb: str, args: List[Any], kwargs: Dict[str, Any]) -> TypedValue:
        model_name = (
            os.getenv(f"FLOWLANG_GEMINI_MODEL_{verb.upper()}")
            or os.getenv("FLOWLANG_GEMINI_MODEL")
            or "gemini-3-flash"
        )
        user_payload = _build_user_payload(team, verb, args, kwargs)
        timeout_s = _get_timeout_s(kwargs, 60)
        retries = _get_retries(kwargs, 2)
        try:
            def _call():
                model = genai.GenerativeModel(model_name)
                system = _system_prompt(verb)
                prompt = f"SYSTEM:\n{system}\n\nUSER:\n{json.dumps(user_payload)}"
                if kwargs.get("stream"):
                    stream = model.generate_content(prompt, stream=True)
                    content_buf: List[str] = []
                    for ev in stream:
                        try:
                            if hasattr(ev, "text") and ev.text:
                                content_buf.append(ev.text)
                        except Exception:
                            pass
                    return "".join(content_buf)
                else:
                    resp = model.generate_content(prompt, request_options={"timeout": timeout_s})
                    return resp.text or ""

            content = _with_retries(_call, retries=retries)
        except Exception as e:
            return TypedValue(tag=ValueTag.Unknown, meta={"error": str(e), "provider": self.name})
        try:
            parsed = json.loads(content) if content else {}
        except Exception:
            parsed = None
        return _map_to_typed_value(verb, content, parsed, kwargs)


class MistralProvider(AIProvider):
    name = "mistral"

    def __init__(self) -> None:
        if not MistralClient or not os.getenv("MISTRAL_API_KEY"):
            raise ProviderError("Mistral not available")
        self.client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

    def execute(self, team: str, verb: str, args: List[Any], kwargs: Dict[str, Any]) -> TypedValue:
        model = (
            os.getenv(f"FLOWLANG_MISTRAL_MODEL_{verb.upper()}")
            or os.getenv("FLOWLANG_MISTRAL_MODEL")
            or "mistral-large-latest"
        )
        user_payload = _build_user_payload(team, verb, args, kwargs)
        timeout_s = _get_timeout_s(kwargs, 60)
        retries = _get_retries(kwargs, 2)
        try:
            def _call():
                messages = [
                    _MistralMsg(role="system", content=_system_prompt(verb)),
                    _MistralMsg(role="user", content=json.dumps(user_payload)),
                ]
                if kwargs.get("stream"):
                    content_buf: List[str] = []
                    for ev in self.client.chat_stream(model=model, messages=messages, temperature=float(kwargs.get("temperature", 0.7))):
                        try:
                            if hasattr(ev, "data") and hasattr(ev.data, "content") and ev.data.content:
                                content_buf.append(ev.data.content)
                        except Exception:
                            pass
                    return "".join(content_buf)
                else:
                    resp = self.client.chat(model=model, messages=messages, temperature=float(kwargs.get("temperature", 0.7)))
                    return resp.choices[0].message.content if resp and resp.choices else ""

            content = _with_retries(_call, retries=retries)
        except Exception as e:
            return TypedValue(tag=ValueTag.Unknown, meta={"error": str(e), "provider": self.name})
        try:
            parsed = json.loads(content) if content else {}
        except Exception:
            parsed = None
        return _map_to_typed_value(verb, content, parsed, kwargs)


class CohereProvider(AIProvider):
    name = "cohere"

    def __init__(self) -> None:
        if not _cohere or not os.getenv("COHERE_API_KEY"):
            raise ProviderError("Cohere not available")
        self.client = _cohere.Client(os.getenv("COHERE_API_KEY"))

    def execute(self, team: str, verb: str, args: List[Any], kwargs: Dict[str, Any]) -> TypedValue:
        model = (
            os.getenv(f"FLOWLANG_COHERE_MODEL_{verb.upper()}")
            or os.getenv("FLOWLANG_COHERE_MODEL")
            or "command-r-plus"
        )
        user_payload = _build_user_payload(team, verb, args, kwargs)
        timeout_s = _get_timeout_s(kwargs, 60)
        retries = _get_retries(kwargs, 2)
        try:
            def _call():
                system = _system_prompt(verb)
                if kwargs.get("stream"):
                    content_buf: List[str] = []
                    for ev in self.client.chat_stream(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user_payload)}]):
                        try:
                            if getattr(ev, "event", "") == "text-generation" and hasattr(ev, "text"):
                                content_buf.append(ev.text)
                        except Exception:
                            pass
                    return "".join(content_buf)
                else:
                    resp = self.client.chat(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user_payload)}])
                    return resp.text or ""

            content = _with_retries(_call, retries=retries)
        except Exception as e:
            return TypedValue(tag=ValueTag.Unknown, meta={"error": str(e), "provider": self.name})
        try:
            parsed = json.loads(content) if content else {}
        except Exception:
            parsed = None
        return _map_to_typed_value(verb, content, parsed, kwargs)


class AzureOpenAIProvider(AIProvider):
    name = "azure"

    def __init__(self) -> None:
        if not requests or not (os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_DEPLOYMENT")):
            raise ProviderError("Azure OpenAI not available")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip("/")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    def execute(self, team: str, verb: str, args: List[Any], kwargs: Dict[str, Any]) -> TypedValue:
        user_payload = _build_user_payload(team, verb, args, kwargs)
        url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
        body = {
            "messages": [
                {"role": "system", "content": _system_prompt(verb)},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            "temperature": float(kwargs.get("temperature", 0.7)),
            "max_tokens": int(kwargs.get("max_tokens", 1000)) if str(kwargs.get("max_tokens", "")).isdigit() else 1000,
        }
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        timeout_s = _get_timeout_s(kwargs, 60)
        retries = _get_retries(kwargs, 2)
        try:
            def _call():
                r = requests.post(url, headers=headers, json=body, timeout=timeout_s)
                r.raise_for_status()
                data = r.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")

            content = _with_retries(_call, retries=retries)
        except Exception as e:
            return TypedValue(tag=ValueTag.Unknown, meta={"error": str(e), "provider": self.name})
        try:
            parsed = json.loads(content) if content else {}
        except Exception:
            parsed = None
        return _map_to_typed_value(verb, content, parsed, kwargs)


class OpenRouterProvider(AIProvider):
    name = "openrouter"

    def __init__(self) -> None:
        if not requests or not os.getenv("OPENROUTER_API_KEY"):
            raise ProviderError("OpenRouter not available")
        self.api_key = os.getenv("OPENROUTER_API_KEY")

    def execute(self, team: str, verb: str, args: List[Any], kwargs: Dict[str, Any]) -> TypedValue:
        model = (
            os.getenv(f"FLOWLANG_OPENROUTER_MODEL_{verb.upper()}")
            or os.getenv("FLOWLANG_OPENROUTER_MODEL")
            or "openrouter/openai/gpt-4o"
        )
        user_payload = _build_user_payload(team, verb, args, kwargs)
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": _system_prompt(verb)},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        }
        timeout_s = _get_timeout_s(kwargs, 60)
        retries = _get_retries(kwargs, 2)
        try:
            def _call():
                r = requests.post(url, headers=headers, json=body, timeout=timeout_s)
                r.raise_for_status()
                data = r.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")

            content = _with_retries(_call, retries=retries)
        except Exception as e:
            return TypedValue(tag=ValueTag.Unknown, meta={"error": str(e), "provider": self.name})
        try:
            parsed = json.loads(content) if content else {}
        except Exception:
            parsed = None
        return _map_to_typed_value(verb, content, parsed, kwargs)


class OllamaProvider(AIProvider):
    name = "ollama"

    def __init__(self) -> None:
        if not requests or not os.getenv("OLLAMA_HOST"):
            raise ProviderError("Ollama not available")
        self.base = os.getenv("OLLAMA_HOST").rstrip("/")

    def execute(self, team: str, verb: str, args: List[Any], kwargs: Dict[str, Any]) -> TypedValue:
        model = (
            os.getenv(f"FLOWLANG_OLLAMA_MODEL_{verb.upper()}")
            or os.getenv("FLOWLANG_OLLAMA_MODEL")
            or "llama3.3"
        )
        user_payload = _build_user_payload(team, verb, args, kwargs)
        url = f"{self.base}/api/chat"
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": _system_prompt(verb)},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            "stream": False,
        }
        timeout_s = _get_timeout_s(kwargs, 120)
        retries = _get_retries(kwargs, 2)
        try:
            def _call():
                # Ollama streaming API differs; we aggregate when stream=True
                if kwargs.get("stream"):
                    body_stream = dict(body)
                    body_stream["stream"] = True
                    content_buf: List[str] = []
                    with requests.post(url, json=body_stream, timeout=timeout_s, stream=True) as r:
                        r.raise_for_status()
                        for line in r.iter_lines():
                            if not line:
                                continue
                            try:
                                obj = json.loads(line.decode("utf-8"))
                                msg = obj.get("message") or {}
                                chunk = msg.get("content", "")
                                if chunk:
                                    content_buf.append(chunk)
                            except Exception:
                                pass
                    return "".join(content_buf)
                else:
                    r = requests.post(url, json=body, timeout=timeout_s)
                    r.raise_for_status()
                    data = r.json()
                    content = ""
                    if isinstance(data, dict):
                        msg = data.get("message") or {}
                        content = msg.get("content", "")
                    return content

            content = _with_retries(_call, retries=retries)
        except Exception as e:
            return TypedValue(tag=ValueTag.Unknown, meta={"error": str(e), "provider": self.name})
        try:
            parsed = json.loads(content) if content else {}
        except Exception:
            parsed = None
        return _map_to_typed_value(verb, content, parsed, kwargs)


def _env_present(keys: List[str]) -> bool:
    return all(os.getenv(k) for k in keys)


def select_provider() -> Optional[AIProvider]:
    """Select an AI provider based on FLOWLANG_AI_PROVIDER or precedence list.

    Precedence: OpenAI → Anthropic → Gemini → Mistral → Cohere → Azure → OpenRouter → Ollama
    """
    forced = (os.getenv("FLOWLANG_AI_PROVIDER") or "").strip().lower()

    candidates: List[str] = [
        "openai",
        "anthropic",
        "gemini",
        "mistral",
        "cohere",
        "azure",
        "openrouter",
        "ollama",
    ]

    def _instantiate(name: str) -> Optional[AIProvider]:
        try:
            if name == "openai" and _env_present(["OPENAI_API_KEY"]) and _OpenAIClient:
                return OpenAIProvider()
            if name == "anthropic" and _env_present(["ANTHROPIC_API_KEY"]) and anthropic:
                return AnthropicProvider()
            if name == "gemini" and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")) and genai:
                return GeminiProvider()
            if name == "mistral" and _env_present(["MISTRAL_API_KEY"]) and MistralClient:
                return MistralProvider()
            if name == "cohere" and _env_present(["COHERE_API_KEY"]) and _cohere:
                return CohereProvider()
            if name == "azure" and _env_present(["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]) and requests:
                return AzureOpenAIProvider()
            if name == "openrouter" and _env_present(["OPENROUTER_API_KEY"]) and requests:
                return OpenRouterProvider()
            if name == "ollama" and os.getenv("OLLAMA_HOST") and requests:
                return OllamaProvider()
        except Exception:
            return None
        return None

    if forced:
        return _instantiate(forced)

    for name in candidates:
        prov = _instantiate(name)
        if prov:
            return prov

    return None
