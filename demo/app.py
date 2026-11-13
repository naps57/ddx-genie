"""Replay MediQ chart-section traces turn by turn. Reads demo/traces/*.json only; no LLM calls.

    streamlit run demo/app.py
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

TRACE_DIR = Path(os.environ.get("DEMO_TRACE_DIR", Path(__file__).resolve().parent / "traces"))

# Colors: one neutral series hue for options; gold gets an accent AND a text label.
OPTION_COLOR = "#2a78d6"
GOLD_COLOR = "#eb6834"

st.set_page_config(page_title="MediQ chart-section demo", layout="wide")
st.markdown("""
<style>
html, body, [class*="css"] { font-size: 18px; }
.ws-section { font-weight: 700; margin: 0.6rem 0 0.2rem 0; font-size: 1.05rem; }
.ws-field { padding: 2px 6px; margin: 1px 0; border-radius: 4px; line-height: 1.35; }
.ws-filled { color: #0b0b0b; }
.ws-empty { color: #9a9993; }
.ws-new { background: #fff1c2; border-left: 4px solid #eda100; }
.tag { display: inline-block; background: #e7eefa; color: #1b4f91; border-radius: 4px;
       padding: 0 6px; font-size: 0.85rem; font-weight: 600; margin-right: 6px; }
.turn { border-bottom: 1px solid #e3e2de; padding: 6px 0 8px 0; }
.turn-current { background: #fffaf0; }
.q { font-weight: 600; }
.a { color: #33322f; }
.cannot { color: #9a9993; font-style: italic; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_traces(dir_str: str, mtime_key: float) -> dict[str, dict]:
    out = {}
    for p in sorted(Path(dir_str).glob("*.json")):
        try:
            t = json.loads(p.read_text())
            out[t["case_id"]] = t
        except Exception as e:  # show, don't crash
            st.sidebar.warning(f"Could not read {p.name}: {e}")
    return out


def dir_mtime(d: Path) -> float:
    return max((p.stat().st_mtime for p in d.glob("*.json")), default=0.0)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def state_at(trace: dict, step: int) -> dict:
    """Step 0 = initial presentation; step k = after turn k."""
    if step == 0:
        return {"snapshot": trace["initial_worksheet_snapshot"],
                "new": set(trace.get("initial_fields_filled", [])),
                "differential": trace["initial_differential"],
                "decision": "answer" if not trace["turns"] else "ask"}
    t = trace["turns"][step - 1]
    return {"snapshot": t["worksheet_snapshot"], "new": set(t["fields_filled"]),
            "differential": t["differential"], "decision": t["decision"]}


def render_worksheet(snapshot: dict, new: set[str]):
    sections: dict[str, list] = {}
    for key, val in snapshot.items():
        sec, field = key.split(".", 1)
        sections.setdefault(sec, []).append((key, field, val))
    parts = []
    for sec, fields in sections.items():
        n_filled = sum(v is not None for _, _, v in fields)
        parts.append(f'<div class="ws-section">{html.escape(sec)} '
                     f'<span style="font-weight:400;color:#6b6a65">({n_filled}/{len(fields)})</span></div>')
        for key, field, val in fields:
            if val is None:
                parts.append(f'<div class="ws-field ws-empty">{html.escape(field)}: unknown</div>')
            else:
                cls = "ws-field ws-filled" + (" ws-new" if key in new else "")
                parts.append(f'<div class="{cls}"><b>{html.escape(field)}</b>: {html.escape(str(val))}</div>')
    st.markdown("\n".join(parts), unsafe_allow_html=True)


def render_conversation(trace: dict, step: int):
    st.markdown(f'<div class="turn"><span class="tag">Initial info</span>'
                f'{html.escape(trace["initial_info"])}</div>', unsafe_allow_html=True)
    for t in trace["turns"][:step]:
        cur = " turn-current" if t["turn"] == step else ""
        ans = t["patient_answer"]
        ans_cls = "cannot" if "cannot answer" in ans.lower() else "a"
        st.markdown(
            f'<div class="turn{cur}"><span class="tag">{t["turn"]}. {html.escape(t["chart_section"])}'
            f' &rarr; {html.escape(t["target_field"])}</span><br>'
            f'<span class="q">Doctor: {html.escape(t["question"])}</span><br>'
            f'<span class="{ans_cls}">Patient: {html.escape(ans)}</span></div>',
            unsafe_allow_html=True)
    if step == 0 and trace["turns"]:
        st.caption("Press **Next turn** to see the first question.")


def differential_chart(differential: list[dict], gold: str):
    rows = []
    for d in differential:
        is_gold = d["option"] == gold
        rows.append({
            "label": f'{d["option"]}: {d["label"]}' + ("  ★ gold" if is_gold else ""),
            "confidence": d["confidence"],
            "pct": f'{d["confidence"]:.0%}',
            "kind": "gold answer" if is_gold else "other option",
        })
    df = pd.DataFrame(rows)
    order = df["label"].tolist()  # already ranked high -> low
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort=order, title=None,
                axis=alt.Axis(labelLimit=320, labelFontSize=15)),
        x=alt.X("confidence:Q", scale=alt.Scale(domain=[0, 1]), title="confidence",
                axis=alt.Axis(format="%", grid=True, gridColor="#ecebe7", labelFontSize=13)),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
        color=alt.Color("kind:N", scale=alt.Scale(domain=["other option", "gold answer"],
                                                   range=[OPTION_COLOR, GOLD_COLOR]),
                        legend=alt.Legend(orient="bottom", title=None, labelFontSize=13)),
        tooltip=[alt.Tooltip("label:N", title="option"), alt.Tooltip("pct:N", title="confidence")],
    )
    text = base.mark_text(align="left", dx=4, fontSize=14, color="#33322f").encode(text="pct:N")
    return (bars + text).properties(height=max(180, 56 * len(df)))


def result_panel(trace: dict):
    gold = trace["gold_answer"]
    opts = trace["options"]

    def fmt(letter):
        mark = "✅" if letter == gold else "❌"
        return f"{mark} {letter}: {opts.get(letter, '?')}"

    st.subheader("Result")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**Gold**  \n{gold}: {opts[gold]}")
    c2.markdown(f"**Floor (chief complaint only)**  \n{fmt(trace['floor_answer'])}")
    c3.markdown(f"**Agent after {trace['num_questions']} questions**  \n{fmt(trace['final_answer'])}")
    c4.markdown(f"**Ceiling (full record)**  \n{fmt(trace['ceiling_answer'])}")
    reason = {"confident": "top option reached the confidence threshold",
              "max_questions": "hit the question limit",
              "no_fields_left": "no askable chart fields left"}.get(trace.get("stop_reason"), "")
    if reason:
        st.caption(f"Stopped because: {reason}.")
    with st.expander("Full patient record (hidden from the agent)"):
        st.write(trace.get("full_context", ""))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def replay_page(traces: dict[str, dict]):
    ids = list(traces)

    def label(cid):
        return f'{"✅" if traces[cid]["correct"] else "❌"}  {cid}'

    cid = st.sidebar.selectbox("Case", ids, format_func=label)
    trace = traces[cid]
    n_steps = len(trace["turns"])
    st.sidebar.markdown(f"**Model:** `{trace['model']}`  \n**Patient:** `{trace.get('patient_model', '')}`")

    key = f"step::{cid}"
    st.session_state.setdefault(key, 0)
    b1, b2, b3 = st.sidebar.columns(3)
    if b1.button("◀ Prev", width="stretch"):
        st.session_state[key] = max(0, st.session_state[key] - 1)
    if b2.button("Next ▶", width="stretch"):
        st.session_state[key] = min(n_steps, st.session_state[key] + 1)
    if b3.button("Reset", width="stretch"):
        st.session_state[key] = 0
    step = st.session_state[key]
    st.sidebar.progress(step / n_steps if n_steps else 1.0,
                        text=f"Turn {step} of {n_steps}" if n_steps else "No questions asked")

    # Top: case header
    st.markdown(f"### {cid}  ·  {trace['source']}")
    st.markdown(f"**Initial info given to the agent:** {trace['initial_info']}")
    st.markdown(f"**Question:** {trace['question']}")
    st.markdown("  \n".join(f"**{k}.** {v}" + ("  ★ *gold*" if k == trace["gold_answer"] else "")
                            for k, v in trace["options"].items()))
    st.divider()

    s = state_at(trace, step)
    left, mid, right = st.columns([1.05, 1.35, 1.2], gap="large")
    with left:
        st.subheader("Worksheet: ClinicalHistory")
        st.caption("Highlighted = filled on this turn; grey = still unknown.")
        render_worksheet(s["snapshot"], s["new"])
    with mid:
        st.subheader("Conversation")
        render_conversation(trace, step)
        if step > 0:
            t = trace["turns"][step - 1]
            with st.expander("Field tagger raw output (this turn)"):
                st.code(t.get("tagger_raw", ""), language="json")
        elif trace.get("initial_tagger_raw"):
            with st.expander("Field tagger raw output (initial info)"):
                st.code(trace["initial_tagger_raw"], language="json")
    with right:
        st.subheader("Differential")
        st.altair_chart(differential_chart(s["differential"], trace["gold_answer"]),
                        width="stretch")
        verb = "**answer**" if s["decision"] == "answer" else "**ask** another question"
        st.markdown(f"Decision after this step: {verb}")

    if step == n_steps:
        st.divider()
        result_panel(trace)


def summary_page(traces: dict[str, dict]):
    st.markdown("### Summary across loaded traces")
    ts = list(traces.values())
    n = len(ts)
    acc = lambda k: sum(t[k] for t in ts) / n  # noqa: E731
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Floor accuracy", f"{acc('floor_correct'):.0%}", help="Chief complaint only, zero questions")
    c2.metric("Agent accuracy", f"{acc('correct'):.0%}")
    c3.metric("Ceiling accuracy", f"{acc('ceiling_correct'):.0%}", help="Full patient record up front")
    c4.metric("Avg questions asked", f"{sum(t['num_questions'] for t in ts) / n:.1f}")
    st.caption(f"{n} cases · model `{ts[0]['model']}`")

    yes = lambda b: "✅" if b else "❌"  # noqa: E731
    rows = [{
        "case": t["case_id"], "gold": t["gold_answer"],
        "floor": f"{yes(t['floor_correct'])} {t['floor_answer']}",
        "agent": f"{yes(t['correct'])} {t['final_answer']}",
        "ceiling": f"{yes(t['ceiling_correct'])} {t['ceiling_answer']}",
        "questions": t["num_questions"], "stopped": t.get("stop_reason", ""),
        "cost ($)": round(t.get("run", {}).get("usage", {}).get("cost_usd", 0.0), 4),
    } for t in ts]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def main():
    traces = load_traces(str(TRACE_DIR), dir_mtime(TRACE_DIR))
    st.sidebar.title("MediQ chart-section demo")
    if not traces:
        st.info(f"No traces found in {TRACE_DIR}. Run `python demo/run_cases.py --n 10` first.")
        return
    page = st.sidebar.radio("View", ["Replay", "Summary"], horizontal=True)
    st.sidebar.divider()
    if page == "Replay":
        replay_page(traces)
    else:
        summary_page(traces)


main()
