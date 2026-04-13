"""
Production guardrails:
- Input: block jailbreaks, prompt injections, PII in queries
- Context: warn on injected instructions inside retrieved chunks
- Output: detect refusals / hallucination red flags
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Injection patterns
_INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|above)\s+instructions?",
    r"forget\s+(everything|what\s+you|your\s+instructions)",
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

def check_query(query: str) -> GuardrailResult:
    """Validate incoming user query"""
    if _INJECTION_RE.search(query):
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