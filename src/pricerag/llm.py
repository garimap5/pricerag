"""Optional LLM estimator grounded on retrieved neighbours (the 'R' and 'G' in RAG).

Kept separate from the headline numbers: it needs an API key and costs money per call, so
`scripts/run_experiments.py --llm N` evaluates it on N test items only if ANTHROPIC_API_KEY is set.
The client is injected (`complete: Callable[[str], str]`) so the logic is unit-testable offline.
"""
from __future__ import annotations

import os
import re
from typing import Callable

PROMPT = """You estimate the retail price (USD) of a product from its description.
Use ONLY the reference products below as pricing evidence. If they disagree, weigh the most
similar ones more and say so. End your answer with a line exactly of the form: PRICE: <number>

Reference products (most similar first):
{refs}

Product to price:
{query}
"""


def build_prompt(query: str, neighbor_texts, neighbor_prices) -> str:
    refs = "\n".join(f"{i + 1}. {t} -> ${p:,.2f}" for i, (t, p) in enumerate(zip(neighbor_texts, neighbor_prices)))
    return PROMPT.format(refs=refs, query=query)


_PRICE = re.compile(r"PRICE:\s*\$?\s*([0-9][0-9,]*\.?[0-9]*)", re.I)


def parse_price(text: str) -> float | None:
    m = _PRICE.findall(text or "")
    return float(m[-1].replace(",", "")) if m else None


def anthropic_complete(model: str = "claude-haiku-5-5") -> Callable[[str], str]:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    import anthropic
    client = anthropic.Anthropic()

    def _complete(prompt: str) -> str:
        r = client.messages.create(model=model, max_tokens=300,
                                   messages=[{"role": "user", "content": prompt}])
        return r.content[0].text
    return _complete


def llm_estimate(model, text: str, complete: Callable[[str], str], k: int = 5) -> float | None:
    sims, idx = model.neighbors([text], k)
    prompt = build_prompt(text, [model.train_texts[i] for i in idx[0]], model.train_prices[idx[0]])
    return parse_price(complete(prompt))
