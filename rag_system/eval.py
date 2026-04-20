"""
Lightweight RAGAS style evaluation without the heavy RAGAS dependency.
Implements:
- Faithfulness: are all claims in the answer supported by the context?
- Answer Relevance: does the answer address the question?
- Context Precision: are the retrieved docs actually relevant to the answer?

Each metric uses an LLM judge (GPT-4o) + optional embedding similarity.
"""

import json
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .config import get_settings
from .models import EvalRequest, EvalResponse

logger = logging.getLogger(__name__)
settings = get_settings()

_eval_llm = ChatOpenAI(
    model = "gpt-4o",
    temperature=0.0,
    openai_api_key=settings.openai_api_key
)

# Faithfulness
_FAITHFULNESS_PROMPT = """\
You are an evaluation judge. Given the CONTEXT and an ANSWER, assess whether every \
factual claim in the answer is explicitly supported by the context.

Context:
{context}

Answer:
{answer}

Score the faithfulness from 0.0 (completely unsupported) to 1.0 (Fully supported).
Respond ONLY with a JSON object: {{"faithfulness": <float>, "reasoning":"<brief>"}}
"""

async def score_faithfulness(answer: str, contexts: list[str]) -> float:
    context_str = "\n---\n".join(contexts)
    prompt = _FAITHFULNESS_PROMPT.format(context=context_str, answer=answer)
    response = await _eval_llm.ainvoke([HumanMessage(content=prompt)])
    try:
        data = json.loads(response.content)
        return float(data["faithfulness"])
    except Exception:
        logger.warning("Faithfulness parse error")
        return 0.0

# Answer Relevance 
_RELEVANCE_PROMPT = """\
You are an evaluation judge. Give a Question and an ANSWER, score how well \
the answer addresses the question.

Question: {question}
Answer: {answer}

Score from 0.0 (completely irrelevant) to 1.0 (perfectly answers the question).
Respond ONLY with a JSON object: {{"relevance": <float>, "reasoning": "<brief>"}}
"""

async def score_answer_relevance(question: str, answer: str) -> float:
    prompt = _RELEVANCE_PROMPT.format(question=question,answer=answer)
    response = await _eval_llm.ainvoke([HumanMessage(content=prompt)])
    try:
        data = json.loads(response.content)
        return float(data["relevance"])
    except Exception:
        logger.warning("Relevance parse error")
        return 0.0

# Context Precision
_PRECISION_PROMPT = """\
You are an evaluation judge. For each retrieved context below, determine whether \
it was USEFUL for answering the question.

Question: {question}
Answer: {answer}
Contexts:
{contexts}

Response with a JSON object:
{{"useful":[trur/false, ...], "precision": <float 0-1>}}
where useful[i] = whether context i contributed to the answer.
"""

async def score_context_precision(
    question: str, answer: str, contexts: list[str]
) -> float:
    ctx_str = "\n".join(f"[{i+1}] {c[:300]}" for i,c in enumerate(contexts))
    prompt = _PRECISION_PROMPT.format(question=question,answer=answer,contexts=ctx_str)
    response = await _eval_llm.invoke([HumanMessage(content=prompt)])
    try:
        data = json.loads(response.content)
        return float(data["precision"])
    except Exception:
        return 0.0

# Composite evaluator
async def evaluate(req: EvalRequest) -> EvalResponse:
    faithfulness = await score_faithfulness(req.answer, req.contexts)
    relevance = await score_answer_relevance(req.question, req.answer)
    precision = await score_context_precision(req.question, req.answer, req.contexts)

    passed = (
        faithfulness >= settings.faithfullness_threshold
        and relevance >= settings.answer_relevance_threshold
    )

    return EvalResponse(
        faithfulness=round(faithfulness,3),
        answer_relevance=round(relevance,3),
        context_precision=round(precision,3),
        passed=passed
    )

print("[eval] Module ready")