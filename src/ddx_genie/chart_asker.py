"""Chart-section asker: a Genie-worksheet-guided arm for the asking baselines.

Same loop and scoring as ``baselines.run_asking`` (start from the complaint-only anchor,
k = 1..K_ASK questions, the same fact-select ``patient_answer``, the same open-ended
``score_facts`` diagnose + judge after every step). The difference is how each question
is chosen: every question targets one unfilled field of a ``ClinicalHistory`` Genie
worksheet and is that field's fixed question from ``specs/chart_sections.yaml``.

Two policies pick the field; everything else is shared, so they differ ONLY in choice:

- ``checklist``: fixed order. The first unfilled, not-yet-asked field in schema order.
- ``differential``: one call scores every unfilled field 0-10 by how much its answer would
  change the current top-5 differential; ask the highest (optionally weighted by a
  per-field reveal-rate prior, see ``--prior``). All scores are logged.

Genie's role: ``specs/chart_sections.yaml`` is written out as a 14-column Genie spec CSV
(``specs/clinical_history.csv``) and the live worksheet is built from that CSV by Genie's
own ``csv_to_classes``. Genie holds the chart state (a field is unfilled iff its Genie value
is None). Genie's dialogue loop is NOT used: its LLM calls bypass ``llm.py`` (no cache, no
cost log) and its policy only asks in declaration order.

Calls per step: [differential scoring] + patient + [tagger, when facts came back] +
diagnose + judge. Every call goes through ``llm.py``.

Results: ``runs/<run_id>/<policy>/<case_id>.json`` (resumable) and ``summary.json``.

Run:
  uv run python -m ddx_genie.chart_asker --write-spec
  uv run python -m ddx_genie.chart_asker --run-id chart_dev_smoke --policy checklist --limit 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from ddx_genie.baselines import (
    CANNOT_ANSWER,
    K_ASK,
    MAX_UNCONFIRMED_CASES,
    anchor,
    patient_answer,
    score_facts,
)
from ddx_genie.diagnosis import _load_json, format_facts
from ddx_genie.llm import RUNS_DIR, LLMClient
from ddx_genie.mediq import DEV_PATH
from ddx_genie.mediq import OUT_PATH as TEST_PATH
from ddx_genie.partial_eval import load_jsonl

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "specs" / "chart_sections.yaml"
SPEC_CSV_PATH = REPO_ROOT / "specs" / "clinical_history.csv"

POLICIES = ("checklist", "differential")
SPLITS = {"dev": DEV_PATH, "test": TEST_PATH}

#: Genie spec header, all 14 columns (see CLAUDE.md: ``Validation`` is the 14th).
GENIE_HEADER = [
    "Form Predicate", "Form Name", "Predicate", "Kind", "Type", "Name", "Enum Values",
    "Description", "Don't Ask", "Required", "Confirm", "Actions", "Form Actions", "Validation",
]

TAG_SYSTEM = """You maintain a patient's clinical chart. Given a doctor's question and the \
patient's answer, list every chart field that the ANSWER provides information for, with a \
short value copied or paraphrased from the answer. Only use what the answer actually \
states; do not infer. A negative finding counts (e.g. "denies fever" fills \
ROS.ros_negatives). Use only field keys from the list.

Reply with JSON only: {"fields": {"Section.field": "value"}}. Use {"fields": {}} if the \
answer fills nothing."""

SCORE_SYSTEM = """You are a physician deciding what to ask next. You see what is known, \
your current differential diagnosis, and the chart fields not yet documented.

For EACH listed field, rate from 0 to 10 how much learning it would change your \
differential: 10 = likely to confirm, rule out, or reorder the leading diagnoses; \
0 = would not change it. Rate every field.

Reply with JSON only: {"scores": {"Section.field": <0-10>}}"""


# ------------------------------------------------------------------------- schema


@dataclass(frozen=True)
class ChartField:
    section: str
    name: str
    description: str
    question: str

    @property
    def key(self) -> str:
        return f"{self.section}.{self.name}"


def load_schema(path: Path = SCHEMA_PATH) -> list[ChartField]:
    """Fields in schema order (the checklist order)."""
    raw = yaml.safe_load(path.read_text())
    return [
        ChartField(sec["name"], f["name"], f["description"].strip(), f["question"].strip())
        for sec in raw["sections"]
        for f in sec["fields"]
    ]


def genie_spec_rows(fields: list[ChartField]) -> list[list[str]]:
    """One Genie worksheet per chart section; every field an optional ``str`` input."""
    blank = [""] * len(GENIE_HEADER)
    rows = [list(GENIE_HEADER)]
    section = None
    for f in fields:
        if f.section != section:
            section = f.section
            form = list(blank)
            form[1], form[4] = section, "worksheet"
            rows.append(form)
        row = list(blank)
        row[3], row[4], row[5], row[7] = "input", "str", f.name, f.description
        rows.append(row)
    return rows


def write_genie_spec(fields: list[ChartField], path: Path = SPEC_CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        csv.writer(fh).writerows(genie_spec_rows(fields))


def _quiet_genie() -> None:
    """Genie logs every field lookup at DEBUG through loguru; keep warnings only."""
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")


def load_genie_worksheets(path: Path = SPEC_CSV_PATH) -> dict[str, Any]:
    """Instantiate every worksheet class Genie builds from the spec CSV, by name."""
    _quiet_genie()
    from worksheets.specification.from_spreadsheet import csv_to_classes

    return {cls.__name__: cls() for kind, cls in csv_to_classes(str(path)) if kind == "worksheet"}


class ChartWorksheet:
    """The live ClinicalHistory chart: one Genie worksheet instance per section."""

    def __init__(self, fields: list[ChartField], spec_path: Path = SPEC_CSV_PATH) -> None:
        self.fields = fields
        self.by_key = {f.key: f for f in fields}
        self.sections = load_genie_worksheets(spec_path)
        for f in fields:
            if f.section not in self.sections or not hasattr(self.sections[f.section], f.name):
                raise ValueError(f"{spec_path} lacks {f.key}; rerun with --write-spec")

    @property
    def keys(self) -> list[str]:
        return [f.key for f in self.fields]

    def get(self, key: str) -> str | None:
        f = self.by_key[key]
        return getattr(self.sections[f.section], f.name).value

    def set(self, key: str, value: str) -> None:
        f = self.by_key[key]
        setattr(self.sections[f.section], f.name, value)

    def unfilled(self) -> list[str]:
        return [k for k in self.keys if self.get(k) is None]

    def snapshot(self) -> dict[str, str | None]:
        return {k: self.get(k) for k in self.keys}

    def catalog(self, keys: list[str]) -> str:
        return "\n".join(f"- {k}: {self.by_key[k].description}" for k in keys)


# ------------------------------------------------------------------------- tagging


def parse_tags(text: str, valid: set[str]) -> dict[str, str]:
    parsed = _load_json(text)
    fields = parsed.get("fields") if isinstance(parsed, dict) else None
    if not isinstance(fields, dict):
        return {}
    return {k: str(v).strip() for k, v in fields.items() if k in valid and str(v).strip()}


def apply_tags(ws: ChartWorksheet, tags: dict[str, str]) -> list[str]:
    """Write tags into the worksheet; appends to an already-filled field. Returns keys."""
    for key, value in tags.items():
        prev = ws.get(key)
        if prev is None:
            ws.set(key, value)
        elif value not in prev:
            ws.set(key, f"{prev}; {value}")
    return list(tags)


def tag_answer(ws: ChartWorksheet, question: str, answer: str, client: LLMClient,
               tag: str) -> tuple[dict[str, str], str]:
    user = (f"Chart fields:\n{ws.catalog(ws.keys)}\n\n"
            f"Doctor's question: {question}\nPatient's answer: {answer}")
    completion = client.complete(
        [{"role": "system", "content": TAG_SYSTEM}, {"role": "user", "content": user}],
        tag=f"{tag}:tag", max_tokens=400,
    )
    return parse_tags(completion.text, set(ws.keys)), completion.text


# ------------------------------------------------------------------------- policies


@dataclass
class Choice:
    key: str
    #: differential only: raw 0-10 scores and prior-weighted scores, per askable field.
    scores: dict[str, float] | None = None
    weighted: dict[str, float] | None = None
    #: True when the differential call gave no usable score and checklist order was used.
    fallback: bool = False
    raw: str = ""


def choose_checklist(askable: list[str]) -> Choice:
    return Choice(key=askable[0])


def parse_scores(text: str, askable: list[str]) -> dict[str, float]:
    """0-10 scores for askable fields; out-of-range values are clamped, junk dropped."""
    parsed = _load_json(text)
    raw = parsed.get("scores") if isinstance(parsed, dict) else None
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key in askable:
        try:
            out[key] = min(10.0, max(0.0, float(raw[key])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def pick_field(scores: dict[str, float], askable: list[str],
               prior: dict[str, float] | None) -> tuple[str | None, dict[str, float]]:
    """argmax of score x prior (prior defaults to 1); ties go to schema order."""
    weighted = {k: s * (prior.get(k, 1.0) if prior else 1.0) for k, s in scores.items()}
    best = None
    for key in askable:  # schema order, so the first maximum wins ties
        if key in weighted and (best is None or weighted[key] > weighted[best]):
            best = key
    return best, weighted


def choose_differential(case: dict[str, Any], known: list[int], differential: list[str],
                        ws: ChartWorksheet, askable: list[str], client: LLMClient, tag: str,
                        prior: dict[str, float] | None = None) -> Choice:
    ddx = "\n".join(f"{i}. {d}" for i, d in enumerate(differential, 1)) or "None yet."
    user = (f"Known so far:\n{format_facts([case['facts'][i] for i in sorted(known)])}\n\n"
            f"Current differential (most likely first):\n{ddx}\n\n"
            f"Fields not yet documented:\n{ws.catalog(askable)}")
    completion = client.complete(
        [{"role": "system", "content": SCORE_SYSTEM}, {"role": "user", "content": user}],
        tag=f"{tag}:score", max_tokens=800,
    )
    scores = parse_scores(completion.text, askable)
    best, weighted = pick_field(scores, askable, prior)
    if best is None:
        return Choice(key=askable[0], scores=scores, weighted=weighted, fallback=True,
                      raw=completion.text)
    return Choice(key=best, scores=scores, weighted=weighted, raw=completion.text)


# ------------------------------------------------------------------------- one case


def run_chart(case: dict[str, Any], client: LLMClient, policy: str,
              fields: list[ChartField], *, prior: dict[str, float] | None = None,
              k_max: int = K_ASK, spec_path: Path = SPEC_CSV_PATH) -> dict[str, Any]:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    tag = f"chart_{policy}"
    ws = ChartWorksheet(fields, spec_path)

    known = anchor(case)
    initial = " ".join(case["facts"][i] for i in known)
    init_tags, init_raw = tag_answer(ws, "(initial presentation)", initial, client, tag)
    current = score_facts(case, known, client, tag)
    steps: list[dict[str, Any]] = [{
        "k": 0, "fields_filled": apply_tags(ws, init_tags), "tagger_raw": init_raw,
        "worksheet": ws.snapshot(), "scored": asdict(current),
    }]

    asked: set[str] = set()
    for k in range(1, k_max + 1):
        askable = [key for key in ws.unfilled() if key not in asked]
        if not askable:
            break
        if policy == "checklist":
            choice = choose_checklist(askable)
        else:
            choice = choose_differential(case, known, current.differential, ws, askable,
                                         client, tag, prior)
        asked.add(choice.key)
        target = ws.by_key[choice.key]

        got = patient_answer(case, target.question, known, client, tag=f"{tag}:patient")
        answer = " ".join(case["facts"][i] for i in got) if got else CANNOT_ANSWER
        tags, tag_raw = tag_answer(ws, target.question, answer, client, tag) if got else ({}, "")
        filled = apply_tags(ws, tags)
        known = known + got
        current = score_facts(case, known, client, tag)
        steps.append({
            "k": k,
            "section": target.section,
            "field": target.name,
            "question": target.question,
            "answer": answer,
            "answered_with": got,
            "wasted": not got,
            "target_filled": target.key in filled,
            "fields_filled": filled,
            "tagger_raw": tag_raw,
            "policy_scores": choice.scores,
            "policy_weighted": choice.weighted,
            "policy_fallback": choice.fallback,
            "policy_raw": choice.raw,
            "worksheet": ws.snapshot(),
            "scored": asdict(current),
        })
    return {"case_id": case["case_id"], "source_id": case["source_id"], "policy": policy,
            "answer": case["answer"], "n_initial": case["n_initial"],
            "n_facts": len(case["facts"]), "steps": steps}


# ------------------------------------------------------------------------- reveal prior
#
# The grounded part of the differential policy (TODO step 4). MediQ has no population
# statistics for P(finding | diagnosis), but it does show how often a chart field is
# documented at all. File every non-anchor fact of each DEV case under chart fields
# (one call per case), then
#     reveal_rate[field] = (cases where >= 1 hidden fact documents the field + 1) / (cases + 2)
# (Laplace-smoothed). A question about a field the record rarely documents is likely to
# come back "cannot answer", so the policy weights its LLM impact score by this rate.

FACT_TAG_SYSTEM = """You file each numbered fact from a patient's case into a clinical \
chart. For each chart field, list the numbers of the facts that document it. A fact may \
go under several fields; leave out fields no fact documents.

Reply with JSON only: {"fields": {"Section.field": [<fact numbers>]}}"""


def parse_fact_tags(text: str, valid: set[str], idx: list[int]) -> dict[str, list[int]]:
    """Map 1-based fact numbers back to fact indices, per known field."""
    parsed = _load_json(text)
    fields = parsed.get("fields") if isinstance(parsed, dict) else None
    if not isinstance(fields, dict):
        return {}
    out: dict[str, list[int]] = {}
    for key, nums in fields.items():
        if key not in valid or not isinstance(nums, list):
            continue
        got = []
        for n in nums:
            try:
                n = int(n)
            except (TypeError, ValueError):
                continue
            if 1 <= n <= len(idx) and idx[n - 1] not in got:
                got.append(idx[n - 1])
        if got:
            out[key] = got
    return out


def tag_case_facts(case: dict[str, Any], fields: list[ChartField], client: LLMClient,
                   tag: str = "prior") -> dict[str, list[int]]:
    idx = [i for i in range(len(case["facts"])) if i not in set(anchor(case))]
    if not idx:
        return {}
    numbered = "\n".join(f"{n}. {case['facts'][i]}" for n, i in enumerate(idx, 1))
    catalog = "\n".join(f"- {f.key}: {f.description}" for f in fields)
    completion = client.complete(
        [{"role": "system", "content": FACT_TAG_SYSTEM},
         {"role": "user", "content": f"Chart fields:\n{catalog}\n\nFacts:\n{numbered}"}],
        tag=f"{tag}:facts", max_tokens=800,
    )
    return parse_fact_tags(completion.text, {f.key for f in fields}, idx)


def estimate_prior(tagged: list[dict[str, list[int]]],
                   fields: list[ChartField]) -> dict[str, Any]:
    """Smoothed per-field and per-section documentation rates over cases."""
    n = len(tagged)
    sections = list(dict.fromkeys(f.section for f in fields))

    def rate(hits: int) -> float:
        return (hits + 1) / (n + 2)

    field_hits = {f.key: sum(bool(t.get(f.key)) for t in tagged) for f in fields}
    section_hits = {s: sum(any(t.get(f.key) for f in fields if f.section == s) for t in tagged)
                    for s in sections}
    return {
        "n_cases": n,
        "smoothing": "laplace (hits + 1) / (n + 2)",
        "field_rate": {k: rate(h) for k, h in field_hits.items()},
        "field_hits": field_hits,
        "section_rate": {s: rate(h) for s, h in section_hits.items()},
    }


# ------------------------------------------------------------------------- summary


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict[str, float]:
    """Exact two-sided McNemar test on paired outcomes (e.g. top-1 per case)."""
    if len(a) != len(b):
        raise ValueError("paired outcomes must have equal length")
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n if n else 1.0
    return {"only_a": only_a, "only_b": only_b, "p": min(1.0, 2 * tail)}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-k top-1 / top-5 / P(gold) (k=0 is the anchor) and asking behaviour."""
    cells: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for res in results:
        for step in res["steps"]:
            cells[step["k"]].append(step["scored"])
    by_k = {k: {"n": len(v),
                "top1": _mean([float(s["match_rank"] == 1) for s in v]),
                "top5": _mean([float(s["match_rank"] is not None) for s in v]),
                "p_gold": _mean([s["p_gold"] for s in v])}
            for k, v in sorted(cells.items())}
    asks = [s for r in results for s in r["steps"] if s["k"] > 0]
    by_field: dict[str, list[float]] = defaultdict(list)
    for s in asks:
        by_field[f"{s['section']}.{s['field']}"].append(float(not s["wasted"]))
    return {
        "by_k": by_k,
        "behaviour": {
            "questions": len(asks),
            "hit_rate": _mean([float(not s["wasted"]) for s in asks]),
            "facts_per_question": _mean([float(len(s["answered_with"])) for s in asks]),
            "target_filled_share": _mean([float(s["target_filled"]) for s in asks]),
            "fallback_share": _mean([float(s["policy_fallback"]) for s in asks]),
            "questions_by_section": dict(Counter(s["section"] for s in asks)),
            # P(asking a field reveals a fact), observed: {field: [hits, asks]}.
            "hits_by_field": {k: [int(sum(v)), len(v)] for k, v in sorted(by_field.items())},
        },
    }


# ------------------------------------------------------------------------- driver


PRIOR_PATH = REPO_ROOT / "data" / "processed" / "mediq" / "reveal_prior_dev.json"


def load_prior(path: Path | None) -> dict[str, float] | None:
    """Accepts the ``--estimate-prior`` output or a flat {field_key: rate} map."""
    if not path:
        return None
    raw = json.loads(path.read_text())
    return raw.get("field_rate", raw)


def run_prior(client: LLMClient, run_id: str, cases: list[dict[str, Any]],
              fields: list[ChartField]) -> dict[str, Any]:
    """Tag every dev case's hidden facts (resumable per case) and estimate the prior."""
    out_dir = RUNS_DIR / run_id / "prior"
    tagged = []
    for n, case in enumerate(cases, 1):
        path = out_dir / f"{case['case_id']}.json"
        if path.exists():
            tagged.append(json.loads(path.read_text())["fields"])
            continue
        client.budget.check()
        fields_found = tag_case_facts(case, fields, client)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"case_id": case["case_id"], "fields": fields_found}) + "\n")
        tagged.append(fields_found)
        print(f"[{n}/{len(cases)}] {case['case_id']}: {len(fields_found)} fields "
              f"| {client.summary()}", flush=True)
    return estimate_prior(tagged, fields)


def run(client: LLMClient, run_id: str, policy: str, cases: list[dict[str, Any]],
        fields: list[ChartField], prior: dict[str, float] | None) -> list[dict[str, Any]]:
    out_dir = RUNS_DIR / run_id / policy
    results = []
    for n, case in enumerate(cases, 1):
        path = out_dir / f"{case['case_id']}.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
            continue
        client.budget.check()
        res = run_chart(case, client, policy, fields, prior=prior)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=1) + "\n")
        results.append(res)
        print(f"[{n}/{len(cases)}] {case['case_id']}: "
              f"top-1 k0={res['steps'][0]['scored']['match_rank'] == 1} "
              f"k{res['steps'][-1]['k']}={res['steps'][-1]['scored']['match_rank'] == 1} "
              f"| {client.summary()}", flush=True)
        if n == 1 and len(cases) > 1:
            per_case = client.cost_log.total_usd
            projected = per_case * len(cases)
            print(f"projected run total ${projected:.4f} (cap ${client.budget.max_run_usd:.2f})")
            if projected > client.budget.max_run_usd:
                raise SystemExit("projected spend exceeds MAX_RUN_USD; aborting the batch")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-spec", action="store_true",
                        help="regenerate specs/clinical_history.csv from the YAML and exit")
    parser.add_argument("--run-id")
    parser.add_argument("--policy", choices=POLICIES, default="checklist")
    parser.add_argument("--split", choices=tuple(SPLITS), default="dev")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--ids", nargs="*", help="case_ids to run instead of the first --limit")
    parser.add_argument("--prior", type=Path,
                        help=f"reveal prior for --policy differential, e.g. {PRIOR_PATH.name}")
    parser.add_argument("--estimate-prior", action="store_true",
                        help=f"tag dev facts and write {PRIOR_PATH.name} (dev split only)")
    parser.add_argument("--confirmed", action="store_true",
                        help=f"required for > {MAX_UNCONFIRMED_CASES} cases or --split test")
    args = parser.parse_args()

    fields = load_schema()
    if args.write_spec:
        write_genie_spec(fields)
        print(f"wrote {SPEC_CSV_PATH.relative_to(REPO_ROOT)} ({len(fields)} fields)")
        return 0
    if not args.run_id:
        parser.error("--run-id is required")

    cases = load_jsonl(SPLITS[args.split])
    if args.ids:
        wanted = set(args.ids)
        cases = [c for c in cases if c["case_id"] in wanted]
        if len(cases) != len(wanted):
            parser.error(f"not in {args.split}: {sorted(wanted - {c['case_id'] for c in cases})}")
    else:
        cases = cases[:args.limit]
    if (len(cases) > MAX_UNCONFIRMED_CASES or args.split == "test") and not args.confirmed:
        parser.error(f"{len(cases)} {args.split} cases needs --confirmed")

    if args.estimate_prior and args.split != "dev":
        parser.error("the reveal prior is estimated on the dev split only")

    client = LLMClient(args.run_id)
    if args.estimate_prior:
        prior = run_prior(client, args.run_id, cases, fields)
        PRIOR_PATH.write_text(json.dumps(prior, indent=2) + "\n")
        print(json.dumps(prior["section_rate"], indent=2))
        print(f"\n{client.summary()}\nwrote {PRIOR_PATH.relative_to(REPO_ROOT)}")
        return 0
    results = run(client, args.run_id, args.policy, cases, fields, load_prior(args.prior))
    summary = summarize(results) | {"policy": args.policy, "split": args.split,
                                    "n_cases": len(results), "usd": client.cost_log.total_usd}
    out = RUNS_DIR / args.run_id / args.policy / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\n{client.summary()}\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
