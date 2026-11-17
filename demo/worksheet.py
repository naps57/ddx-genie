"""ClinicalHistory as a Genie Worksheet, built from demo/chart_sections.yaml.

Structure (Genie nested worksheets):
    ClinicalHistory
        HPI: HPI          -> GenieWorksheet with str fields onset, duration, ...
        PMH: PMH          -> ...
        ...

We use Genie's data model (GenieWorksheet / GenieField, nested worksheets,
`get_genie_fields_from_ws`, `is_complete`) to hold and query state. A field is
unfilled iff its Genie value is None. Deciding *which* fields a patient answer
fills is done by our LiteLLM tagger in agent.py, not by Genie's own semantic
parser (which is wired to LangChain/Azure and would bypass LiteLLM).
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from loguru import logger

# Genie logs every field lookup at DEBUG; keep the console readable.
logger.remove()
logger.add(sys.stderr, level="WARNING")

from worksheets.core.fields import GenieField  # noqa: E402
from worksheets.core.worksheet import GenieWorksheet  # noqa: E402
from worksheets.utils.field import get_genie_fields_from_ws  # noqa: E402

DEFAULT_SCHEMA = Path(__file__).resolve().parent / "chart_sections.yaml"


def load_schema(path: Path | str = DEFAULT_SCHEMA) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_classes(schema: dict):
    """Create one GenieWorksheet subclass per section plus the ClinicalHistory root."""
    section_classes = {}
    root_attrs = {}
    for sec in schema["sections"]:
        attrs = {
            f["name"]: GenieField(str, f["name"], description=f["description"])
            for f in sec["fields"]
        }
        cls = type(sec["name"], (GenieWorksheet,), attrs)
        section_classes[sec["name"]] = cls
        root_attrs[sec["name"]] = GenieField(
            cls, sec["name"], description=sec.get("title", sec["name"]),
            question=sec.get("opening_question", ""))
    root_cls = type(schema.get("worksheet", "ClinicalHistory"), (GenieWorksheet,), root_attrs)
    return root_cls, section_classes


class ChartWorksheet:
    """Convenience wrapper around a live ClinicalHistory Genie worksheet instance."""

    def __init__(self, schema: dict | None = None):
        self.schema = schema or load_schema()
        self.root_cls, self.section_classes = build_classes(self.schema)
        # Pre-instantiate every section so all leaf fields exist and can be read.
        self.ws = self.root_cls(**{name: cls() for name, cls in self.section_classes.items()})
        self.sections = {s["name"]: s for s in self.schema["sections"]}

    # --- Genie access -----------------------------------------------------
    def _section_ws(self, section: str) -> GenieWorksheet:
        return getattr(self.ws, section).value

    def _leaf(self, section: str, field: str) -> GenieField:
        return getattr(self._section_ws(section), field)

    def keys(self) -> list[str]:
        out = []
        for sec_field in get_genie_fields_from_ws(self.ws):
            for leaf in get_genie_fields_from_ws(sec_field.value):
                out.append(f"{sec_field.name}.{leaf.name}")
        return out

    def get(self, key: str):
        section, field = key.split(".", 1)
        return self._leaf(section, field).value

    def set(self, key: str, value: str) -> None:
        section, field = key.split(".", 1)
        setattr(self._section_ws(section), field, value)

    def is_filled(self, key: str) -> bool:
        return self.get(key) is not None

    def unfilled(self) -> list[str]:
        return [k for k in self.keys() if not self.is_filled(k)]

    def is_complete(self) -> bool:
        # No predicates are used, so Genie's completeness check needs no bot/context.
        return bool(self.ws.is_complete(None, None))

    def snapshot(self) -> dict:
        return {k: self.get(k) for k in self.keys()}

    # --- prompt helpers ---------------------------------------------------
    def describe(self, key: str) -> str:
        section, field = key.split(".", 1)
        for f in self.sections[section]["fields"]:
            if f["name"] == field:
                return f["description"]
        return ""

    def field_catalog(self, keys: list[str] | None = None) -> str:
        keys = keys if keys is not None else self.keys()
        return "\n".join(f"- {k}: {self.describe(k)}" for k in keys)
