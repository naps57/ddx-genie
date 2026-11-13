"""Minimal chart-section Expert agent (placeholder baseline).

This is a stand-in until Arda's baseline is available; everything the trace
generator needs from an agent goes through `ChartSectionAgent.run_case`, so this
file is the single swap point.

Loop per case:
    0. Tag the initial info into the worksheet; rank options (= floor answer).
    repeat:
    1. Question writer picks an unfilled chart field and writes one question.
    2. MediQ FactSelectPatient answers.
    3. Field tagger (LiteLLM) says which worksheet fields the answer fills.
    4. Option ranker produces the differential.
    5. Decide: answer if top confidence >= threshold, or out of questions/fields.

Calls per turn: question writer, patient, tagger (skipped when the patient
cannot answer), ranker.
"""
from __future__ import annotations

import json

from llm import Meter, chat, parse_json
from worksheet import ChartWorksheet

CANNOT_ANSWER = "cannot answer"

EXPERT_SYSTEM = (
    "You are an experienced physician interviewing a patient to answer a clinical "
    "multiple-choice question. You only know what is written below."
)


def _options_block(options: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in options.items())


def _conversation_block(history: list[dict]) -> str:
    if not history:
        return "(no questions asked yet)"
    return "\n".join(f"Doctor: {h['question']}\nPatient: {h['answer']}" for h in history)


# ---------------------------------------------------------------------------
# Option ranker.
# DEMO INSTRUMENTATION: MediQ's experts produce only a single letter per turn,
# never a ranked differential. In this placeholder agent the ranking is also
# the stopping rule; if Arda's baseline replaces the agent, keep this call
# purely to draw the differential chart.
# ---------------------------------------------------------------------------
def rank_options(patient_info: str, history: list[dict], question: str, options: dict,
                 *, cfg: dict, meter: Meter, tag: str = "ranker") -> tuple[list[dict], str]:
    letters = list(options)
    json_shape = "{" + ", ".join(f'"{l}": 0.0' for l in letters) + "}"
    prompt = (
        f"PATIENT INFORMATION:\n{patient_info}\n\n"
        f"CONVERSATION LOG:\n{_conversation_block(history)}\n\n"
        f"QUESTION: {question}\n\nOPTIONS:\n{_options_block(options)}\n\n"
        "Using only the information above, estimate the probability that each option "
        "is the correct answer. Probabilities must sum to 1. Respond with JSON only, "
        f"in the form {json_shape}."
    )
    raw = chat([{"role": "system", "content": EXPERT_SYSTEM},
                {"role": "user", "content": prompt}],
               model=cfg["model"], temperature=cfg["temperature"], json_mode=True,
               meter=meter, tag=tag)
    parsed = parse_json(raw) or {}
    scores = {}
    for l in letters:
        try:
            scores[l] = max(0.0, float(parsed.get(l, 0.0)))
        except (TypeError, ValueError):
            scores[l] = 0.0
    total = sum(scores.values())
    if total <= 0:  # unparseable -> uniform, flagged in trace via raw output
        scores = {l: 1.0 / len(letters) for l in letters}
        total = 1.0
    diff = [{"option": l, "label": options[l], "confidence": round(scores[l] / total, 4)}
            for l in letters]
    diff.sort(key=lambda d: (-d["confidence"], d["option"]))
    return diff, raw


class ChartSectionAgent:
    def __init__(self, cfg: dict, meter: Meter, schema: dict | None = None):
        self.cfg = cfg
        self.meter = meter
        self.schema = schema

    # --- field tagger ------------------------------------------------------
    def tag_fields(self, ws: ChartWorksheet, question: str, answer: str) -> tuple[dict, str]:
        """Return ({field_key: extracted value}, raw tagger output)."""
        prompt = (
            "You maintain a patient's clinical chart. Given a doctor's question and the "
            "patient's answer, list every chart field that the ANSWER provides information "
            "for, with a short value copied or paraphrased from the answer. Only use what "
            "the answer actually states; do not infer. A field may be filled by a negative "
            "finding (e.g. 'denies fever' fills ROS.ros_negatives). Use only field keys from "
            "the list.\n\n"
            f"CHART FIELDS:\n{ws.field_catalog()}\n\n"
            f"DOCTOR QUESTION: {question}\n"
            f"PATIENT ANSWER: {answer}\n\n"
            'Respond with JSON only: {"fields": {"Section.field": "value", ...}}. '
            'Use {"fields": {}} if the answer fills nothing.'
        )
        raw = chat([{"role": "user", "content": prompt}], model=self.cfg["model"],
                   temperature=self.cfg["temperature"], json_mode=True,
                   meter=self.meter, tag="tagger")
        parsed = parse_json(raw) or {}
        fields = parsed.get("fields", {}) if isinstance(parsed, dict) else {}
        valid = set(ws.keys())
        return {k: str(v) for k, v in fields.items() if k in valid and v}, raw

    @staticmethod
    def apply_tags(ws: ChartWorksheet, tags: dict) -> list[str]:
        for key, value in tags.items():
            prev = ws.get(key)
            ws.set(key, value if prev is None or value in prev else f"{prev}; {value}")
        return list(tags)

    # --- question writer ---------------------------------------------------
    def next_question(self, ws: ChartWorksheet, info: str, history: list[dict],
                      differential: list[dict], question: str, options: dict,
                      dead_ends: set[str]) -> dict | None:
        askable = [k for k in ws.unfilled() if k not in dead_ends]
        if not askable:
            return None
        asked_sections = {h["chart_section"] for h in history}
        hints = []
        for s in ws.schema["sections"]:
            if any(k.startswith(s["name"] + ".") for k in askable):
                tag = "already asked" if s["name"] in asked_sections else "not yet asked"
                hints.append(f"- {s['name']} ({tag}): opening question \"{s['opening_question']}\"; "
                             f"follow-ups: {', '.join(s['follow_ups'])}")
        filled = {k: v for k, v in ws.snapshot().items() if v is not None}
        top = ", ".join(f"{d['option']} ({d['confidence']:.2f})" for d in differential[:3])
        prompt = (
            f"PATIENT INFORMATION:\n{info}\n\n"
            f"CONVERSATION LOG:\n{_conversation_block(history)}\n\n"
            f"QUESTION: {question}\n\nOPTIONS:\n{_options_block(options)}\n\n"
            f"CURRENT LEADING OPTIONS: {top}\n\n"
            f"CHART FIELDS ALREADY KNOWN:\n{json.dumps(filled, indent=1) if filled else '(none)'}\n\n"
            f"UNFILLED CHART FIELDS YOU MAY ASK ABOUT:\n{ws.field_catalog(askable)}\n\n"
            f"SECTION QUESTION GUIDE:\n" + "\n".join(hints) + "\n\n"
            "Choose the ONE unfilled field whose answer would best help decide between the "
            "options, and write one short, specific question to the patient about it. For a "
            "section not yet asked, you may adapt its opening question. Respond with JSON only: "
            '{"chart_section": "...", "target_field": "...", "question": "..."}'
        )
        raw = chat([{"role": "system", "content": EXPERT_SYSTEM},
                    {"role": "user", "content": prompt}],
                   model=self.cfg["model"], temperature=self.cfg["temperature"],
                   json_mode=True, meter=self.meter, tag="question_writer")
        p = parse_json(raw) or {}
        key = f"{p.get('chart_section')}.{p.get('target_field')}"
        q = (p.get("question") or "").strip()
        fallback = False
        if key not in askable or not q:
            # Fall back to the first askable field in schema order, with its opening question.
            key = askable[0]
            sec = ws.sections[key.split(".")[0]]
            q = sec["opening_question"]
            fallback = True
        section, field = key.split(".", 1)
        return {"chart_section": section, "target_field": field, "question": q,
                "question_writer_raw": raw, "question_writer_fallback": fallback}

    # --- main loop ----------------------------------------------------------
    def run_case(self, case: dict, patient, info: str, max_questions: int) -> dict:
        question, options = case["question"], case["options"]
        ws = ChartWorksheet(self.schema)
        history: list[dict] = []
        dead_ends: set[str] = set()  # fields asked about that the patient could not answer

        init_tags, init_raw = self.tag_fields(ws, "(initial presentation)", info)
        init_filled = self.apply_tags(ws, init_tags)
        differential, rank_raw = rank_options(info, history, question, options,
                                              cfg=self.cfg, meter=self.meter)
        out = {
            "initial_fields_filled": init_filled,
            "initial_tagger_raw": init_raw,
            "initial_worksheet_snapshot": ws.snapshot(),
            "initial_differential": differential,
            "initial_ranker_raw": rank_raw,
            "turns": [],
        }

        stop_reason = None
        if differential[0]["confidence"] >= self.cfg["answer_confidence_threshold"]:
            stop_reason = "confident"
        elif max_questions <= 0:
            stop_reason = "max_questions"

        while stop_reason is None:
            nq = self.next_question(ws, info, history, differential, question, options, dead_ends)
            if nq is None:
                stop_reason = "no_fields_left"
                if out["turns"]:
                    out["turns"][-1]["decision"] = "answer"
                break
            answer = patient.respond(nq["question"])
            target = f"{nq['chart_section']}.{nq['target_field']}"
            if CANNOT_ANSWER in answer.lower():
                tags, tag_raw = {}, "(skipped: patient could not answer)"
                dead_ends.add(target)
            else:
                tags, tag_raw = self.tag_fields(ws, nq["question"], answer)
            filled = self.apply_tags(ws, tags)
            if target not in filled and CANNOT_ANSWER not in answer.lower():
                dead_ends.add(target)  # answered, but not the field we asked about
            history.append({"question": nq["question"], "answer": answer,
                            "chart_section": nq["chart_section"]})
            differential, rank_raw = rank_options(info, history, question, options,
                                                  cfg=self.cfg, meter=self.meter)
            n = len(history)
            if differential[0]["confidence"] >= self.cfg["answer_confidence_threshold"]:
                stop_reason = "confident"
            elif n >= max_questions:
                stop_reason = "max_questions"
            out["turns"].append({
                "turn": n,
                "chart_section": nq["chart_section"],
                "target_field": nq["target_field"],
                "question": nq["question"],
                "patient_answer": answer,
                "fields_filled": filled,
                "worksheet_snapshot": ws.snapshot(),
                "differential": differential,
                "decision": "answer" if stop_reason else "ask",
                "tagger_raw": tag_raw,
                "ranker_raw": rank_raw,
                "question_writer_raw": nq["question_writer_raw"],
                "question_writer_fallback": nq["question_writer_fallback"],
            })

        out["final_answer"] = differential[0]["option"]
        out["final_differential"] = differential
        out["stop_reason"] = stop_reason
        out["num_questions"] = len(history)
        return out
