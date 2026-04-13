# memory.py
"""
Statelesss-server-friendly conversation memory.
History is passed from the client on each request (no server-side session state)
Compression kicks in when history exceeds the token budget
"""

import logging
import tiktoken
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from config import get_settings
from prompt import STANDALONE_QUESTION_PROMPT

logger = logging.getLogger(__name__)
settings = get_settings()

_enc = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text: str) -> int:
    return len(_enc.encode(text))

def build_lc_messages(
    history: list[dict],
    system_prompt: str,
) -> list:
    """Convert raw history dicts to Langchain message objects"""
    messages = [SystemMessage(content=system_prompt)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    return messages

def trim_history_to_budget(
    history: list[dict],
    max_tokens: int = None,
) -> list[dict]:
    """
    Sliding window: keep the MOST recent turns that fit the token budget.
    Always keeps at minimum the last 2 turns (1 exchange)
    """
    budget = max_tokens or settings.context_window_tokens // 3 # 1/3 of budget for history
    trimmed: list[dict] = []
    total = 0

    for turn in reversed(history[-settings.max_history_turns * 2:]): # multiplied by 2 to take both user and AIResponse
        tokens = count_tokens(turn["content"])
        if total + tokens > budget and len(trimmed) >= 2: # checks if token exceeds budget limit or trimmed is more than 2, to avoid only user or ai going with sys prompt
            break
        trimmed.insert(0,turn)
        total += tokens
    
    return trimmed

async def resolve_standalone_question(
    question: str,
    history: list[dict],
    llm: ChatOpenAI
) -> str:
    """
    If conversation history exists, use LLM to rewrite the followup question
    as a self-contained query (critical for multi-turn retrieval accuracy)
    """
    if not history:
        return question
    
    history_str = "\n".join(
        f"{t['role'].capitalize()}: {t['content']}" for t in history[-6:]
    )

    prompt = STANDALONE_QUESTION_PROMPT.format(
        history=history_str,
        question=question
    )

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    standalone = response.content.strip()
    logger.debug(f"Standalone question: '{standalone}")
    return standalone

print("[memory] Module ready")