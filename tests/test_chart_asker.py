"""Tests for the chart-section asker: schema, Genie spec, parsing, policies, the loop.

No LLM calls: a scripted fake client stands in for ``LLMClient``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from ddx_genie.chart_asker import (
    GENIE_HEADER,
    SPEC_CSV_PATH,
    ChartField,
    ChartWorksheet,
    apply_tags,
    choose_checklist,
    estimate_prior,
    genie_spec_rows,
    load_prior,
    load_schema,
    mcnemar_exact,
    parse_fact_tags,
    parse_scores,
    parse_tags,
    pick_field,
    run_chart,
    summarize,
    write_genie_spec,
)

FIELDS = [
    ChartField("HPI", "onset", "When it began.", "When did it start?"),
    ChartField("HPI", "location", "Where it is.", "Where is it?"),
    ChartField("ExamLabs", "lab_results", "Test results.", "What do the tests show?"),
]
KEYS = [f.key for f in FIELDS]


@pytest.fixture
def spec(tmp_path: Path) -> Path:
    path = tmp_path / "spec.csv"
    write_genie_spec(FIELDS, path)
    return path


# ------------------------------------------------------------------ schema and spec


def test_repo_schema_loads_and_spec_is_current() -> None:
    fields = load_schema()
    assert len(fields) == len({f.key for f in fields}) >= 20
    assert all(f.question.endswith("?") and f.description for f in fields)
    rows = list(csv.reader(SPEC_CSV_PATH.open()))
    assert rows == [[c for c in r] for r in genie_spec_rows(fields)], "rerun --write-spec"


def test_spec_rows_have_14_columns_and_one_form_per_section() -> None:
    rows = genie_spec_rows(FIELDS)
    assert rows[0] == GENIE_HEADER and len(GENIE_HEADER) == 14
    assert all(len(r) == 14 for r in rows)
    assert [r[1] for r in rows[1:] if r[4] == "worksheet"] == ["HPI", "ExamLabs"]


def test_genie_builds_independent_worksheets(spec: Path) -> None:
    a, b = ChartWorksheet(FIELDS, spec), ChartWorksheet(FIELDS, spec)
    assert type(a.sections["HPI"]).__mro__[1].__name__ == "GenieWorksheet"
    assert a.unfilled() == KEYS
    a.set("HPI.onset", "2 days")
    assert a.get("HPI.onset") == "2 days" and a.unfilled() == KEYS[1:]
    assert b.unfilled() == KEYS  # no state shared through the classes


def test_stale_spec_is_rejected(spec: Path) -> None:
    extra = FIELDS + [ChartField("Family", "family_history", "d", "q?")]
    with pytest.raises(ValueError, match="write-spec"):
        ChartWorksheet(extra, spec)


# ------------------------------------------------------------------ parsing


def test_parse_tags_keeps_only_known_nonempty_fields() -> None:
    text = '```json\n{"fields": {"HPI.onset": "2 days", "Bogus.x": "y", "HPI.location": ""}}\n```'
    assert parse_tags(text, set(KEYS)) == {"HPI.onset": "2 days"}
    assert parse_tags("nope", set(KEYS)) == {}
    assert parse_tags('{"fields": []}', set(KEYS)) == {}


def test_apply_tags_appends_new_information(spec: Path) -> None:
    ws = ChartWorksheet(FIELDS, spec)
    assert apply_tags(ws, {"HPI.onset": "2 days"}) == ["HPI.onset"]
    apply_tags(ws, {"HPI.onset": "sudden"})
    apply_tags(ws, {"HPI.onset": "sudden"})
    assert ws.get("HPI.onset") == "2 days; sudden"


def test_parse_scores_clamps_and_drops_junk() -> None:
    text = json.dumps({"scores": {"HPI.onset": 12, "HPI.location": "x", "ExamLabs.lab_results": -1,
                                  "Other.y": 5}})
    assert parse_scores(text, KEYS) == {"HPI.onset": 10.0, "ExamLabs.lab_results": 0.0}
    assert parse_scores("garbage", KEYS) == {}


def test_pick_field_prior_weighting_and_ties() -> None:
    scores = {"HPI.onset": 6.0, "HPI.location": 6.0, "ExamLabs.lab_results": 8.0}
    assert pick_field(scores, KEYS, None)[0] == "ExamLabs.lab_results"
    best, weighted = pick_field(scores, KEYS, {"ExamLabs.lab_results": 0.5})
    assert best == "HPI.onset"  # 6 == 6 > 4: tie goes to schema order
    assert weighted["ExamLabs.lab_results"] == 4.0
    assert pick_field({}, KEYS, None)[0] is None


# ------------------------------------------------------------------ the loop


@dataclass
class _Completion:
    text: str


class FakeClient:
    """Answers by the tag suffix; the patient reveals fact 3 for lab questions only."""

    def __init__(self, scores: str = "") -> None:
        self.tags: list[str] = []
        self.scores = scores

    def complete(self, messages, *, tag="", **_):
        self.tags.append(tag)
        user = messages[-1]["content"].lower()
        kind = tag.rsplit(":", 1)[-1]
        if kind == "patient":
            # Hidden pool is facts [1, 2], numbered 1..2 in the prompt; "2" is the x-ray.
            hit = "tests show" in user and "chest x-ray" in user
            return _Completion(json.dumps({"facts": [2] if hit else []}))
        if kind == "tag":
            filled = {"ExamLabs.lab_results": "x-ray"} if "x-ray" in user.split("answer:")[-1] else {}
            return _Completion(json.dumps({"fields": filled}))
        if kind == "score":
            return _Completion(self.scores)
        if kind == "diagnose":
            found = "chest x-ray" in user
            ddx = ["Pneumonia", "Bronchitis"] if found else ["Bronchitis", "Pneumonia"]
            return _Completion(json.dumps({"differential": [
                {"diagnosis": ddx[0], "probability": 0.7}, {"diagnosis": ddx[1], "probability": 0.3}]}))
        if kind == "judge":
            first = user.split("candidates:\n1. ")[1].split("\n")[0]
            rel = ["same", "different"] if first == "pneumonia" else ["different", "same"]
            return _Completion(json.dumps({"verdicts": [
                {"candidate": i, "relation": r} for i, r in enumerate(rel, 1)]}))
        raise AssertionError(f"unexpected call {tag}")


CASE = {"case_id": "c1", "source_id": 1, "answer": "Pneumonia", "n_initial": 1,
        "facts": ["A 40-year-old man has a cough.", "He smokes.", "Chest x-ray shows consolidation."]}


def test_choose_checklist_round_robins_sections() -> None:
    keys = ["A.a1", "A.a2", "A.a3", "B.b1", "C.c1", "C.c2"]

    def next_key(asked: list[str]) -> str:
        return choose_checklist([k for k in keys if k not in asked], asked).key

    order: list[str] = []
    for _ in keys:
        order.append(next_key(order))
    assert order == ["A.a1", "B.b1", "C.c1", "A.a2", "C.c2", "A.a3"]
    # A field filled by tagging (not asked) leaves its section's count unchanged.
    assert choose_checklist(["A.a2", "B.b1"], []).key == "A.a2"


def test_checklist_round_robin_and_never_repeats(spec: Path) -> None:
    client = FakeClient()
    res = run_chart(CASE, client, "checklist", FIELDS, k_max=5, spec_path=spec)
    asked = [s["field"] for s in res["steps"][1:]]
    assert asked == ["onset", "lab_results", "location"]  # then nothing askable is left
    assert [s["wasted"] for s in res["steps"][1:]] == [True, False, True]
    assert res["steps"][2]["answered_with"] == [2] and res["steps"][2]["target_filled"]
    assert res["steps"][1]["scored"]["match_rank"] == 2
    assert res["steps"][2]["scored"]["match_rank"] == 1
    assert not any(t.endswith(":score") for t in client.tags)
    assert "chart_checklist:patient" in client.tags


def test_differential_follows_scores_and_logs_them(spec: Path) -> None:
    scores = json.dumps({"scores": {"HPI.onset": 2, "HPI.location": 1, "ExamLabs.lab_results": 9}})
    res = run_chart(CASE, FakeClient(scores), "differential", FIELDS, k_max=1, spec_path=spec)
    step = res["steps"][1]
    assert step["field"] == "lab_results" and not step["policy_fallback"]
    assert step["policy_scores"]["ExamLabs.lab_results"] == 9.0
    assert step["scored"]["match_rank"] == 1


def test_differential_falls_back_to_checklist_on_bad_output(spec: Path) -> None:
    res = run_chart(CASE, FakeClient("not json"), "differential", FIELDS, k_max=1, spec_path=spec)
    assert res["steps"][1]["field"] == "onset" and res["steps"][1]["policy_fallback"]


def test_summarize(spec: Path) -> None:
    res = run_chart(CASE, FakeClient(), "checklist", FIELDS, k_max=5, spec_path=spec)
    s = summarize([res])
    assert s["by_k"][0]["top1"] == 0.0 and s["by_k"][3]["top1"] == 1.0
    assert s["behaviour"]["questions"] == 3
    assert s["behaviour"]["hit_rate"] == pytest.approx(1 / 3)
    assert s["behaviour"]["questions_by_section"] == {"HPI": 2, "ExamLabs": 1}


def test_summarize_hits_by_field(spec: Path) -> None:
    res = run_chart(CASE, FakeClient(), "checklist", FIELDS, k_max=5, spec_path=spec)
    hits = summarize([res])["behaviour"]["hits_by_field"]
    assert hits == {"ExamLabs.lab_results": [1, 1], "HPI.location": [0, 1], "HPI.onset": [0, 1]}


# ------------------------------------------------------------------ reveal prior and stats


def test_parse_fact_tags_maps_numbers_to_indices() -> None:
    idx = [2, 5, 7]  # hidden facts, numbered 1..3 in the prompt
    text = '{"fields": {"HPI.onset": [1, 3, 3, 9, "x"], "Bogus.y": [1], "HPI.location": []}}'
    assert parse_fact_tags(text, set(KEYS), idx) == {"HPI.onset": [2, 7]}
    assert parse_fact_tags("junk", set(KEYS), idx) == {}


def test_estimate_prior_is_smoothed_per_field_and_section() -> None:
    tagged = [{"HPI.onset": [1]}, {"HPI.onset": [2], "HPI.location": [3]}, {}]
    prior = estimate_prior(tagged, FIELDS)
    assert prior["n_cases"] == 3
    assert prior["field_rate"]["HPI.onset"] == pytest.approx(3 / 5)
    assert prior["field_rate"]["ExamLabs.lab_results"] == pytest.approx(1 / 5)
    assert prior["section_rate"] == {"HPI": pytest.approx(3 / 5), "ExamLabs": pytest.approx(1 / 5)}


def test_load_prior_accepts_both_shapes(tmp_path: Path) -> None:
    full, flat = tmp_path / "a.json", tmp_path / "b.json"
    full.write_text(json.dumps({"n_cases": 2, "field_rate": {"HPI.onset": 0.5}}))
    flat.write_text(json.dumps({"HPI.onset": 0.25}))
    assert load_prior(full) == {"HPI.onset": 0.5}
    assert load_prior(flat) == {"HPI.onset": 0.25}
    assert load_prior(None) is None


def test_mcnemar_exact() -> None:
    # 6 discordant pairs all favouring a: p = 2 * (1/2)**6
    a = [True] * 6 + [True, False] * 2
    b = [False] * 6 + [True, False] * 2
    r = mcnemar_exact(a, b)
    assert (r["only_a"], r["only_b"]) == (6, 0) and r["p"] == pytest.approx(2 / 64)
    assert mcnemar_exact([True, False], [True, False])["p"] == 1.0
    with pytest.raises(ValueError):
        mcnemar_exact([True], [])
