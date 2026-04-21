"""
Production guardrails:
- Input: block jailbreaks, prompt injections, PII in queries
- Context: warn on injected instructions inside retrieved chunks
- Output: detect refusals / hallucination red flags
"""

import re
import logging
import os
from dataclasses import dataclass
from threading import Lock

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_load_lock = Lock()
_llama_guard_model = None
_llama_guard_tokenizer = None
_llama_guard_load_attempted = False


def _strip_wrapping_quotes(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
        return value[1:-1]
    return value


def _read_hf_token_from_env_file() -> str | None:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return None

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, raw_val = stripped.split("=", 1)
                key = key.strip().lower()
                if key in {"hf_token", "huggingface_hub_token", "huggingface_token"}:
                    return _strip_wrapping_quotes(raw_val)
    except Exception as exc:
        logger.warning("Failed to read .env token fallback: %s", exc)

    return None


def _resolve_hf_token() -> str | None:
    # 1) Settings value loaded by pydantic
    token = _strip_wrapping_quotes(settings.hf_token)
    if token:
        return token

    # 2) Process environment (common HF variable names)
    token = _strip_wrapping_quotes(os.getenv("HF_TOKEN"))
    if token:
        return token

    token = _strip_wrapping_quotes(os.getenv("HUGGINGFACE_HUB_TOKEN"))
    if token:
        return token

    token = _strip_wrapping_quotes(os.getenv("HUGGINGFACE_TOKEN"))
    if token:
        return token

    # 3) Direct read from rag_system/.env for late updates
    return _read_hf_token_from_env_file()

# Injection patterns
_INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|above)\s+instructions?",
    r"forget\s+(everything|what\s+you|all\s+(of\s+)?your\s+instructions|your\s+instructions)",
    r"you\s+are\s+now\s+(a|an|the)\s+\w+",
    r"act\s+as\s+(a|an)\s+\w+",
    r"jailbreak",
    r"dan\s+mode",
    r"<\|im_start\|>",
    r"</?(system|user|assistant)>",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS),re.IGNORECASE)

# PII patterns (basic)
_PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"(\+?\d[\d\-\s().]{7,}\d)"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}

@dataclass
class GuardrailResult:
    allowed: bool
    reason: str = ""
    sanitized_text: str = ""


def _load_llama_guard() -> bool:
    """Lazy-load Llama Guard once. Returns True when model is ready."""
    global _llama_guard_model, _llama_guard_tokenizer, _llama_guard_load_attempted

    if _llama_guard_model is not None and _llama_guard_tokenizer is not None:
        return True
    if _llama_guard_load_attempted:
        return False

    with _load_lock:
        if _llama_guard_model is not None and _llama_guard_tokenizer is not None:
            return True
        if _llama_guard_load_attempted:
            return False

        _llama_guard_load_attempted = True

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_source = settings.guardrails_local_model_path or settings.guardrails_model_id
            local_exists = False
            if settings.guardrails_local_model_path:
                local_exists = os.path.exists(model_source) and os.path.exists(
                    os.path.join(model_source, "config.json")
                )

            # Optional one-time local download for gated model (requires accepted license + HF token).
            if not local_exists and settings.guardrails_download_if_missing:
                from huggingface_hub import snapshot_download

                token = _resolve_hf_token()
                if not token:
                    logger.warning(
                        "HF_TOKEN is not set. Gated model download may fail for %s",
                        settings.guardrails_model_id,
                    )
                logger.info(
                    "Llama Guard model not found locally. Downloading from %s",
                    settings.guardrails_model_id,
                )

                download_kwargs = {
                    "repo_id": settings.guardrails_model_id,
                    "token": token,
                }
                if settings.guardrails_local_model_path:
                    download_kwargs["local_dir"] = settings.guardrails_local_model_path

                snapshot_download(**download_kwargs)

                if settings.guardrails_local_model_path:
                    logger.info(
                        "Llama Guard model downloaded to explicit path: %s",
                    model_source,
                    )
                    local_exists = os.path.exists(os.path.join(model_source, "config.json"))
                else:
                    logger.info("Llama Guard model downloaded into Hugging Face shared cache")
                    local_exists = True

            # If local files are unavailable and offline-only mode is disabled, fall back to repo id.
            if not local_exists:
                model_source = settings.guardrails_model_id

            load_kwargs = {}
            if torch.cuda.is_available():
                load_kwargs["torch_dtype"] = torch.bfloat16
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["torch_dtype"] = torch.float32

            # Enforce offline load when explicit local model path is being used.
            if settings.guardrails_local_model_path and model_source == settings.guardrails_local_model_path:
                load_kwargs["local_files_only"] = True
            else:
                load_kwargs["local_files_only"] = settings.guardrails_local_files_only

            _llama_guard_model = AutoModelForCausalLM.from_pretrained(
                model_source,
                **load_kwargs,
            )
            _llama_guard_tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                local_files_only=load_kwargs["local_files_only"],
            )
            logger.info("Llama Guard loaded successfully from: %s", model_source)
            return True
        except Exception as exc:
            logger.warning(
                "Llama Guard unavailable, falling back to regex guardrails: %s | "
                "Tip: accept model license on Hugging Face, set HF_TOKEN, and/or "
                "download model into Hugging Face cache",
                exc,
            )
            _llama_guard_model = None
            _llama_guard_tokenizer = None
            return False


def _check_query_llama_guard(query: str) -> GuardrailResult:
    """Run model-based safety classification using Llama Guard."""
    if not _load_llama_guard():
        return GuardrailResult(allowed=True, sanitized_text=query)

    try:
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": query}],
            }
        ]

        input_ids = _llama_guard_tokenizer.apply_chat_template(
            conversation,
            return_tensors="pt",
        ).to(_llama_guard_model.device)

        prompt_len = input_ids.shape[1]
        output = _llama_guard_model.generate(
            input_ids,
            max_new_tokens=settings.guardrails_max_new_tokens,
            pad_token_id=_llama_guard_tokenizer.eos_token_id or 0,
        )
        generated_tokens = output[:, prompt_len:]
        verdict = _llama_guard_tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True,
        ).strip()

        verdict_lower = verdict.lower()
        if "unsafe" in verdict_lower:
            reason = f"Llama Guard blocked query: {verdict}"
            logger.warning("Blocked by Llama Guard | query='%s' | verdict='%s'", query, verdict)
            return GuardrailResult(allowed=False, reason=reason)

        logger.info("Llama Guard passed query | verdict='%s'", verdict)
        return GuardrailResult(allowed=True, sanitized_text=query)
    except Exception as exc:
        logger.warning("Llama Guard runtime check failed; falling back to regex checks: %s", exc)
        return GuardrailResult(allowed=True, sanitized_text=query)

def check_query(query: str) -> GuardrailResult:
    """Validate incoming user query"""
    if settings.guardrails_use_llama_guard:
        llama_result = _check_query_llama_guard(query)
        if not llama_result.allowed:
            return llama_result

    if _INJECTION_RE.search(query):
        logger.warning("Blocked by regex guardrails | reason='Potential Prompt Injection detected' | query='%s'", query)
        return GuardrailResult(allowed=False,reason="Potential Prompt Injection detected")
    
    #Warn on PII (here we won't block, just log it)
    found_pii = [pii_type for pii_type,pat in _PII_PATTERNS.items() if pat.search(query)]
    if found_pii:
        logger.warning(f"PII detected in query: {found_pii}")
    
    return GuardrailResult(allowed=True, sanitized_text=query)

def check_context(text: str) -> bool:
    """
    Scan retrieved context for embedded instructions
    Returns True if suspicious (log + apply defensive prompt)
    """
    if _INJECTION_RE.search(text):
        logger.warning("Potential prompt injection found in retrieved context!")
        return True
    return False

def redact_pii(text: str) -> str:
    """Replace detected PII in a string with placeholder tokens"""
    for label, pat in _PII_PATTERNS.items():
        text = pat.sub(f"[{label.upper()}_REDACTED]",text)
    return text

_REFUSAL_PHRASES = [
    "i cannot answer",
    "i don't have information",
    "i'm not able to",
    "as an ai",
    "i cannot provide",
]

def is_refusal(answer: str) -> bool:
    lower = answer.lower()
    return any(p in lower for p in _REFUSAL_PHRASES)

print("[guardrails] Module ready")