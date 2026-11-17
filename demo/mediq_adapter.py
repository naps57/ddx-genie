"""Thin wrapper around an unmodified MediQ checkout.

We load MediQ's own `src/patient.py` and run its `FactSelectPatient` class as-is.
The only thing we replace is the module it imports its LLM call from (`helper`):
before importing patient.py we register a stand-in `helper` module that routes
`get_response` through LiteLLM. This sidesteps the bugs in MediQ's helper.py
(top-level `import torch`, missing `return` on the OpenAI path, pre-1.0 OpenAI
client usage) without editing any MediQ file. See demo/NOTES.md.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from llm import Meter, chat

SOURCES = {
    "medqa": "all_dev_good.jsonl",
    "craftmd": "all_craft_md.jsonl",
}

# ---------------------------------------------------------------------------
# Stand-in for MediQ's helper.get_response
# ---------------------------------------------------------------------------
_shim_state = {"meter": None, "max_tokens": 256, "temperature": 0.0}


def _shim_get_response(messages, model_name, use_vllm=False, use_api=None, **kwargs):
    # Stock MediQ ignores the patient's `max_length` (it reads `max_tokens`, default 256).
    # We use config.patient_max_tokens instead (2048: thinking tokens count against it). See NOTES.md.
    text = chat(messages, model=model_name, temperature=_shim_state["temperature"],
                max_tokens=_shim_state["max_tokens"], meter=_shim_state["meter"],
                tag="patient")
    return text, None, None  # MediQ callers unpack (text, logprobs, usage)


def _load_mediq_patient_module(mediq_dir: str):
    shim = types.ModuleType("helper")
    shim.get_response = _shim_get_response
    sys.modules["helper"] = shim  # patient.py does `from helper import get_response`
    path = Path(mediq_dir) / "src" / "patient.py"
    spec = importlib.util.spec_from_file_location("mediq_patient", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_patient_mod = None


def make_patient(sample: dict, cfg: dict, meter: Meter):
    """Return MediQ's FactSelectPatient for this case, using the shipped facts."""
    global _patient_mod
    if _patient_mod is None:
        _patient_mod = _load_mediq_patient_module(cfg["mediq_dir"])
    _shim_state.update(meter=meter, max_tokens=cfg["patient_max_tokens"],
                       temperature=cfg["temperature"])
    s = dict(sample)
    # MediQ's data file calls it `facts`; patient.py looks for `atomic_facts` and
    # otherwise regenerates facts with the LLM. Map it so the shipped facts are used.
    s["atomic_facts"] = sample["facts"]
    args = SimpleNamespace(patient_model=cfg["patient_model"], use_vllm=False, use_api=None)
    return _patient_mod.FactSelectPatient(args, s)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_cases(mediq_dir: str, source: str) -> list[dict]:
    sources = list(SOURCES) if source == "both" else [source]
    cases = []
    for src in sources:
        with open(Path(mediq_dir) / "data" / SOURCES[src]) as f:
            for line in f:
                rec = json.loads(line)
                rec["source"] = src
                rec["case_id"] = f"{src}-{rec['id']}"
                cases.append(rec)
    return cases


def initial_info(sample: dict) -> str:
    """Same rule MediQ's Patient uses: the first context sentence."""
    ctx = sample["context"]
    return ctx[0] if isinstance(ctx, list) else ctx.split(". ")[0]


def full_context(sample: dict) -> str:
    ctx = sample["context"]
    return " ".join(ctx) if isinstance(ctx, list) else ctx
