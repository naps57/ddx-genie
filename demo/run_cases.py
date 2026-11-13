"""Generate replayable traces: floor / agent / ceiling for N MediQ dev cases.

    python demo/run_cases.py --n 10 --max-questions 10 --seed 0
    python demo/run_cases.py --ids medqa-0 medqa-17     # specific cases

Floor   = option ranking from the initial info only (zero questions). This is the
          agent's own turn-0 ranking, so floor and agent share one prompt.
Ceiling = the same ranking prompt given the full patient record up front.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from agent import ChartSectionAgent, rank_options
from llm import DEMO_DIR, Meter, load_config
from mediq_adapter import full_context, initial_info, load_cases, make_patient
from worksheet import load_schema


def run_one(case: dict, cfg: dict, schema: dict, max_questions: int) -> dict:
    meter = Meter()
    info = initial_info(case)
    patient = make_patient(case, cfg, meter)
    agent = ChartSectionAgent(cfg, meter, schema)
    t0 = time.time()
    res = agent.run_case(case, patient, info, max_questions)

    ceiling_diff, ceiling_raw = rank_options(full_context(case), [], case["question"],
                                             case["options"], cfg=cfg, meter=meter,
                                             tag="ceiling_ranker")
    gold = case["answer_idx"]
    floor_answer = res["initial_differential"][0]["option"]
    trace = {
        "case_id": case["case_id"],
        "source": case["source"],
        "model": cfg["model"],
        "patient_model": cfg["patient_model"],
        "initial_info": info,
        "question": case["question"],
        "options": case["options"],
        "gold_answer": gold,
        "initial_fields_filled": res["initial_fields_filled"],
        "initial_tagger_raw": res["initial_tagger_raw"],
        "initial_worksheet_snapshot": res["initial_worksheet_snapshot"],
        "initial_differential": res["initial_differential"],
        "turns": res["turns"],
        "final_answer": res["final_answer"],
        "correct": res["final_answer"] == gold,
        "num_questions": res["num_questions"],
        "stop_reason": res["stop_reason"],
        "floor_answer": floor_answer,
        "floor_correct": floor_answer == gold,
        "ceiling_answer": ceiling_diff[0]["option"],
        "ceiling_correct": ceiling_diff[0]["option"] == gold,
        "ceiling_differential": ceiling_diff,
        "ceiling_ranker_raw": ceiling_raw,
        "full_context": full_context(case),
        "patient_facts": case["facts"],
        "run": {
            "max_questions": max_questions,
            "answer_confidence_threshold": cfg["answer_confidence_threshold"],
            "temperature": cfg["temperature"],
            "seconds": round(time.time() - t0, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "usage": meter.summary(),
        },
    }
    return trace


def pick_cases(cases: list[dict], n: int, seed: int) -> list[dict]:
    # Skip the few MediQ records with < 3 shipped facts: with empty facts, MediQ's
    # patient silently regenerates them with the LLM, which we want to avoid.
    eligible = [c for c in cases if len(c.get("facts") or []) >= 3]
    return random.Random(seed).sample(eligible, min(n, len(eligible)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-questions", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", choices=["medqa", "craftmd", "both"], default="medqa")
    ap.add_argument("--ids", nargs="*", help="explicit case ids, e.g. medqa-12")
    ap.add_argument("--out", default=str(DEMO_DIR / "traces"))
    ap.add_argument("--config", default=str(DEMO_DIR / "config.yaml"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    schema = load_schema()
    all_cases = load_cases(cfg["mediq_dir"], "both" if args.ids else args.source)
    if args.ids:
        by_id = {c["case_id"]: c for c in all_cases}
        missing = [i for i in args.ids if i not in by_id]
        if missing:
            sys.exit(f"unknown case ids: {missing}")
        chosen = [by_id[i] for i in args.ids]
    else:
        chosen = pick_cases(all_cases, args.n, args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"model={cfg['model']} patient={cfg['patient_model']} cases={len(chosen)} "
          f"max_questions={args.max_questions}")

    traces, failures = [], []
    for i, case in enumerate(chosen, 1):
        try:
            tr = run_one(case, cfg, schema, args.max_questions)
        except Exception as e:  # keep going; report at the end
            failures.append((case["case_id"], repr(e)))
            print(f"[{i}/{len(chosen)}] {case['case_id']}: FAILED {e!r}")
            continue
        (out_dir / f"{tr['case_id']}.json").write_text(json.dumps(tr, indent=2))
        traces.append(tr)
        u = tr["run"]["usage"]
        print(f"[{i}/{len(chosen)}] {tr['case_id']}: gold={tr['gold_answer']} "
              f"floor={tr['floor_answer']} agent={tr['final_answer']} ceiling={tr['ceiling_answer']} "
              f"q={tr['num_questions']} ({tr['stop_reason']}) calls={u['num_calls']} "
              f"${u['cost_usd']:.4f}")

    if traces:
        n = len(traces)
        acc = lambda k: sum(t[k] for t in traces) / n  # noqa: E731
        cost = sum(t["run"]["usage"]["cost_usd"] for t in traces)
        print("\n| metric | value |\n|---|---|")
        print(f"| cases | {n} |")
        print(f"| floor accuracy (chief complaint only) | {acc('floor_correct'):.0%} |")
        print(f"| agent accuracy | {acc('correct'):.0%} |")
        print(f"| ceiling accuracy (full record) | {acc('ceiling_correct'):.0%} |")
        print(f"| avg questions asked | {sum(t['num_questions'] for t in traces) / n:.1f} |")
        print(f"| total cost | ${cost:.4f} |")
    if failures:
        print("\nFailures:")
        for cid, err in failures:
            print(f"  {cid}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
