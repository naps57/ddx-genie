"""All LLM calls go through here (LiteLLM), so the model is one config value.

Each call is recorded on a Meter so every trace carries its own token and dollar cost.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import litellm
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "demo"

load_dotenv(REPO_ROOT / ".env")
litellm.drop_params = True  # let non-OpenAI models ignore params they don't support
litellm.suppress_debug_info = True


def load_config(path: Path | str = DEMO_DIR / "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["mediq_dir"] = str((REPO_ROOT / cfg["mediq_dir"]).resolve())
    return cfg


@dataclass
class Meter:
    """Accumulates usage across all calls made for one case."""
    calls: list = field(default_factory=list)

    def record(self, tag: str, model: str, resp) -> None:
        u = getattr(resp, "usage", None)
        try:
            cost = float(litellm.completion_cost(completion_response=resp))
        except Exception:
            cost = 0.0  # unknown model pricing; tokens are still recorded
        self.calls.append({
            "tag": tag,
            "model": model,
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
            "cost_usd": cost,
        })

    def summary(self) -> dict:
        by_tag: dict[str, int] = {}
        for c in self.calls:
            by_tag[c["tag"]] = by_tag.get(c["tag"], 0) + 1
        return {
            "num_calls": len(self.calls),
            "calls_by_tag": by_tag,
            "prompt_tokens": sum(c["prompt_tokens"] for c in self.calls),
            "completion_tokens": sum(c["completion_tokens"] for c in self.calls),
            "cost_usd": round(sum(c["cost_usd"] for c in self.calls), 6),
        }


def chat(messages: list[dict], *, model: str, temperature: float | None = None,
         max_tokens: int | None = None, json_mode: bool = False,
         meter: Meter | None = None, tag: str = "") -> str:
    kwargs = {"model": model, "messages": messages}
    if temperature is not None:  # some models (current Claude) reject non-default values
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = litellm.completion(**kwargs)
    if meter is not None:
        meter.record(tag, model, resp)
    return (resp.choices[0].message.content or "").strip()


def parse_json(text: str) -> dict | None:
    """Lenient JSON parse: accepts fenced blocks or leading/trailing prose."""
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None
