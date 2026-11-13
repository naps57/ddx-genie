"""Bake demo/traces/*.json into one self-contained HTML replay page (no server needed).

    python demo/export_html.py                 # -> demo/site/index.html
    python demo/export_html.py --out some.html

The page mirrors the Streamlit app: case picker, step through turns, worksheet /
conversation / differential columns, result panel, and a summary view.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

DEMO = Path(__file__).resolve().parent

KEEP_TURN = ["turn", "chart_section", "target_field", "question", "patient_answer",
             "fields_filled", "worksheet_snapshot", "differential", "decision", "tagger_raw"]
KEEP_TRACE = ["case_id", "source", "model", "patient_model", "initial_info", "question", "options",
              "gold_answer", "initial_fields_filled", "initial_tagger_raw",
              "initial_worksheet_snapshot", "initial_differential", "final_answer", "correct",
              "num_questions", "stop_reason", "floor_answer", "floor_correct", "ceiling_answer",
              "ceiling_correct", "ceiling_differential", "full_context"]


def slim(t: dict) -> dict:
    out = {k: t.get(k) for k in KEEP_TRACE}
    out["turns"] = [{k: x.get(k) for k in KEEP_TURN} for x in t["turns"]]
    out["cost_usd"] = t.get("run", {}).get("usage", {}).get("cost_usd")
    out["max_questions"] = t.get("run", {}).get("max_questions")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=str(DEMO / "traces"))
    ap.add_argument("--out", default=str(DEMO / "site" / "index.html"))
    args = ap.parse_args(argv)

    traces = [slim(json.loads(p.read_text())) for p in sorted(Path(args.traces).glob("*.json"))]
    if not traces:
        raise SystemExit(f"no traces in {args.traces}")
    schema = yaml.safe_load((DEMO / "chart_sections.yaml").read_text())
    sections = {s["name"]: s.get("title", s["name"]) for s in schema["sections"]}
    payload = json.dumps({"traces": traces, "sections": sections}, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")  # safe inside <script>

    html = TEMPLATE.replace("__DATA__", payload)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({len(traces)} cases, {len(html) / 1024:.0f} KB)")


TEMPLATE = r"""<title>MediQ Chart Replay</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: clinical chart replay. Sticky control bar, case header, three working columns
   (chart / interview / differential), results strip. Stacks to one column on phones. */
:root {
  --bg: #f6f7f8;
  --panel: #ffffff;
  --ink: #15191d;
  --ink-2: #4a5560;
  --ink-3: #8a949e;
  --rule: #dfe3e7;
  --accent: #1f6f8b;          /* chart-section teal: interface accent */
  --accent-soft: #e3f0f4;
  --bar: #2a78d6;             /* option bars */
  --gold: #d9622b;            /* gold answer: color + star label */
  --new: #fff3c9;             /* field filled on this turn */
  --new-edge: #d9a300;
  --good: #1f8a4c;
  --bad: #b3361f;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111417; --panel: #191d21; --ink: #e9edf0; --ink-2: #b4bec7; --ink-3: #7c8791;
  --rule: #2c3238; --accent: #5bb3d0; --accent-soft: #17323c; --bar: #3987e5; --gold: #ef7a45;
  --new: #3a3215; --new-edge: #d9a300; --good: #4cc27f; --bad: #ef7461; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #111417; --panel: #191d21; --ink: #e9edf0; --ink-2: #b4bec7; --ink-3: #7c8791;
  --rule: #2c3238; --accent: #5bb3d0; --accent-soft: #17323c; --bar: #3987e5; --gold: #ef7a45;
  --new: #3a3215; --new-edge: #d9a300; --good: #4cc27f; --bad: #ef7461; color-scheme: dark; }

* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: var(--font-body); font-size: 16px; line-height: 1.45; }
body::before { content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10; background: var(--panel); }
.wrap { max-width: 1400px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
header.bar { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--panel); border-bottom: 1px solid var(--rule); }
header.bar .wrap { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 18px; padding-block: 10px; }
.brand { font-weight: 700; font-size: 1.05rem; letter-spacing: -0.01em; }
.brand small { font-family: var(--font-mono); font-weight: 400; color: var(--ink-3); font-size: 0.78rem; margin-left: 6px; }
nav.tabs { display: flex; gap: 4px; }
nav.tabs a { text-decoration: none; color: var(--ink-2); padding: 6px 12px; border-radius: 6px; font-weight: 500; }
nav.tabs a[aria-current="page"] { background: var(--accent-soft); color: var(--accent); }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-left: auto; }
select, button { font: inherit; color: var(--ink); background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 6px 10px; }
select { max-width: 100%; }
button { cursor: pointer; font-weight: 500; }
button:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
button:disabled { opacity: 0.45; cursor: default; }
button.primary { background: var(--accent); border-color: var(--accent); color: var(--panel); }
button.primary:hover:not(:disabled) { color: var(--panel); filter: brightness(1.08); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.step { font-family: var(--font-mono); font-size: 0.85rem; color: var(--ink-2); font-variant-numeric: tabular-nums; min-width: 9ch; }
.progress { height: 3px; background: var(--rule); }
.progress > i { display: block; height: 100%; background: var(--accent); transition: width 0.2s; }

main { padding-block: 20px 48px; display: grid; gap: 20px; }
.eyebrow { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink-3); font-weight: 600; }
h1, h2, h3 { margin: 0; text-wrap: balance; }
h2 { font-size: 1rem; font-weight: 600; }
.case-head { display: grid; gap: 10px; }
.case-head h1 { font-size: 1.35rem; font-weight: 600; max-width: 70ch; }
.presenting { font-size: 1.05rem; max-width: 80ch; }
.presenting b { font-weight: 600; }
.options { list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 6px 18px; }
.options li { display: flex; gap: 8px; min-width: 0; }
.options .L { font-family: var(--font-mono); font-weight: 500; color: var(--ink-2); }
.gold-tag { color: var(--gold); font-weight: 600; white-space: nowrap; }

.cols { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.35fr) minmax(0, 1.1fr); gap: 16px; align-items: start; }
@media (max-width: 1000px) { .cols { grid-template-columns: minmax(0, 1fr); } }
.panel { background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 14px 16px; display: grid; gap: 10px; min-width: 0; }
.panel > header { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
.hint { font-size: 0.8rem; color: var(--ink-3); }

.ws-sec { display: grid; gap: 2px; }
.ws-sec h3 { font-size: 0.85rem; font-weight: 600; display: flex; justify-content: space-between; color: var(--ink); padding-top: 6px; border-top: 1px solid var(--rule); }
.ws-sec:first-of-type h3 { border-top: 0; padding-top: 0; }
.ws-sec h3 span { font-family: var(--font-mono); font-weight: 400; color: var(--ink-3); font-variant-numeric: tabular-nums; }
.f { font-size: 0.88rem; padding: 2px 6px; border-radius: 4px; border-left: 3px solid transparent; overflow-wrap: anywhere; }
.f .k { font-family: var(--font-mono); font-size: 0.8rem; color: var(--ink-2); }
.f.empty { color: var(--ink-3); }
.f.empty .k { color: var(--ink-3); }
.f.new { background: var(--new); border-left-color: var(--new-edge); }

.convo { display: grid; gap: 0; }
.turn { padding: 10px 0; border-top: 1px solid var(--rule); display: grid; gap: 4px; }
.turn:first-child { border-top: 0; padding-top: 0; }
.turn.current { background: linear-gradient(90deg, var(--accent-soft), transparent 70%); margin-inline: -16px; padding-inline: 16px; }
.chip { justify-self: start; font-family: var(--font-mono); font-size: 0.75rem; background: var(--accent-soft); color: var(--accent); border-radius: 4px; padding: 1px 7px; }
.q { font-weight: 600; }
.a { color: var(--ink-2); white-space: pre-line; overflow-wrap: anywhere; }
.a.cannot { font-style: italic; color: var(--ink-3); }
details { font-size: 0.85rem; }
summary { cursor: pointer; color: var(--accent); }
pre { font-family: var(--font-mono); font-size: 0.78rem; background: var(--bg); border: 1px solid var(--rule); border-radius: 6px; padding: 8px; overflow-x: auto; white-space: pre-wrap; overflow-wrap: anywhere; margin: 6px 0 0; }

.dd { display: grid; gap: 10px; }
.dd-row { display: grid; gap: 3px; }
.dd-label { display: flex; gap: 8px; font-size: 0.9rem; min-width: 0; }
.dd-label .L { font-family: var(--font-mono); font-weight: 500; }
.dd-label .t { min-width: 0; overflow-wrap: anywhere; }
.dd-track { display: grid; grid-template-columns: 1fr 4.5ch; align-items: center; gap: 8px; }
.dd-bg { height: 16px; background: var(--bg); border-radius: 0 4px 4px 0; position: relative; }
.dd-bar { position: absolute; inset: 0 auto 0 0; background: var(--bar); border-radius: 0 4px 4px 0; transition: width 0.25s; min-width: 2px; }
.dd-row.gold .dd-bar { background: var(--gold); }
.dd-val { font-family: var(--font-mono); font-size: 0.85rem; text-align: right; font-variant-numeric: tabular-nums; }
.legend { display: flex; gap: 14px; font-size: 0.8rem; color: var(--ink-2); flex-wrap: wrap; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.decision { font-size: 0.9rem; color: var(--ink-2); }
.decision b { color: var(--ink); }

.result { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }
.res { background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 12px 14px; display: grid; gap: 4px; min-width: 0; }
.res .v { font-size: 0.95rem; overflow-wrap: anywhere; }
.ok { color: var(--good); font-weight: 600; }
.no { color: var(--bad); font-weight: 600; }

.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.tile { background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 14px 16px; display: grid; gap: 2px; }
.tile .n { font-size: 2rem; font-weight: 600; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
.tile .s { font-size: 0.82rem; color: var(--ink-3); }
.tablebox { overflow-x: auto; background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--ink-3); font-weight: 600; }
td a { color: var(--accent); font-family: var(--font-mono); }
tr:last-child td { border-bottom: 0; }
.note { font-size: 0.85rem; color: var(--ink-2); max-width: 80ch; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<header class="bar">
  <div class="wrap">
    <div class="brand">MediQ Chart Replay<small id="model"></small></div>
    <nav class="tabs"><a href="#" id="tab-replay">Replay</a><a href="#summary" id="tab-summary">Summary</a></nav>
    <div class="controls" id="controls">
      <label for="case" class="hint">Case</label>
      <select id="case"></select>
      <button id="prev" type="button">&#9664; Prev</button>
      <button id="next" type="button" class="primary">Next &#9654;</button>
      <button id="reset" type="button">Reset</button>
      <span class="step" id="step"></span>
    </div>
  </div>
  <div class="progress"><i id="prog"></i></div>
</header>

<main class="wrap" id="main"></main>

<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const T = DATA.traces;
const byId = Object.fromEntries(T.map(t => [t.case_id, t]));
const steps = {};          // case_id -> current step (0 = initial presentation)
let current = T[0].case_id;

const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pct = x => Math.round(x * 100) + "%";
const mark = ok => ok ? '<span class="ok">&#10003; correct</span>' : '<span class="no">&#10007; wrong</span>';

$("model").textContent = T[0].model.replace(/^anthropic\//, "");
$("case").innerHTML = T.map(t => `<option value="${esc(t.case_id)}">${t.correct ? "✓" : "✗"}  ${esc(t.case_id)}</option>`).join("");

function stateAt(t, k) {
  if (k === 0) return { snap: t.initial_worksheet_snapshot, fresh: new Set(t.initial_fields_filled || []),
                        diff: t.initial_differential, decision: t.turns.length ? "ask" : "answer" };
  const x = t.turns[k - 1];
  return { snap: x.worksheet_snapshot, fresh: new Set(x.fields_filled), diff: x.differential, decision: x.decision };
}

function worksheet(snap, fresh) {
  const groups = {};
  for (const [key, val] of Object.entries(snap)) {
    const [sec, field] = key.split(".");
    (groups[sec] ||= []).push([key, field, val]);
  }
  return Object.entries(groups).map(([sec, fs]) => {
    const n = fs.filter(f => f[2] !== null).length;
    return `<div class="ws-sec"><h3>${esc(DATA.sections[sec] || sec)}<span>${n}/${fs.length}</span></h3>` +
      fs.map(([key, field, val]) => val === null
        ? `<div class="f empty"><span class="k">${esc(field)}</span> &middot; unknown</div>`
        : `<div class="f${fresh.has(key) ? " new" : ""}"><span class="k">${esc(field)}</span> ${esc(val)}</div>`).join("") +
      `</div>`;
  }).join("");
}

function conversation(t, k) {
  let h = `<div class="turn${k === 0 ? " current" : ""}"><span class="chip">initial info</span><div>${esc(t.initial_info)}</div></div>`;
  for (const x of t.turns.slice(0, k)) {
    const cannot = /cannot answer/i.test(x.patient_answer);
    h += `<div class="turn${x.turn === k ? " current" : ""}">
      <span class="chip">${x.turn} &middot; ${esc(x.chart_section)} &rarr; ${esc(x.target_field)}</span>
      <div class="q">Doctor: ${esc(x.question)}</div>
      <div class="a${cannot ? " cannot" : ""}">Patient: ${esc(x.patient_answer.trim())}</div></div>`;
  }
  const raw = k === 0 ? t.initial_tagger_raw : t.turns[k - 1].tagger_raw;
  if (raw) h += `<details><summary>Field tagger raw output (${k === 0 ? "initial info" : "turn " + k})</summary><pre>${esc(raw)}</pre></details>`;
  if (k === 0 && t.turns.length) h += `<p class="hint">Press Next (or &rarr;) to see the first question.</p>`;
  return h;
}

function differential(diff, gold) {
  return `<div class="dd">` + diff.map(d => `
    <div class="dd-row${d.option === gold ? " gold" : ""}" title="${esc(d.option)}: ${pct(d.confidence)}">
      <div class="dd-label"><span class="L">${esc(d.option)}</span><span class="t">${esc(d.label)}${d.option === gold ? ' <span class="gold-tag">&#9733; gold</span>' : ""}</span></div>
      <div class="dd-track"><div class="dd-bg"><div class="dd-bar" style="width:${(d.confidence * 100).toFixed(1)}%"></div></div><span class="dd-val">${pct(d.confidence)}</span></div>
    </div>`).join("") + `</div>
    <div class="legend"><span><i style="background:var(--bar)"></i>other option</span><span><i style="background:var(--gold)"></i>gold answer</span></div>`;
}

function result(t) {
  const o = l => `${esc(l)}: ${esc(t.options[l] ?? "?")}`;
  const stop = { max_questions: "used its full question budget", confident: "reached the confidence threshold",
                 no_fields_left: "ran out of chart fields to ask about" }[t.stop_reason] || "";
  return `<section style="display:grid;gap:10px">
    <div class="eyebrow">Result</div>
    <div class="result">
      <div class="res"><div class="eyebrow">Gold answer</div><div class="v">${o(t.gold_answer)}</div></div>
      <div class="res"><div class="eyebrow">Floor &middot; chief complaint only</div><div class="v">${o(t.floor_answer)}</div>${mark(t.floor_correct)}</div>
      <div class="res"><div class="eyebrow">Agent &middot; after ${t.num_questions} questions</div><div class="v">${o(t.final_answer)}</div>${mark(t.correct)}</div>
      <div class="res"><div class="eyebrow">Ceiling &middot; full record</div><div class="v">${o(t.ceiling_answer)}</div>${mark(t.ceiling_correct)}</div>
    </div>
    ${stop ? `<p class="note">The agent stopped because it ${stop}.</p>` : ""}
    <details><summary>Full patient record (hidden from the agent)</summary><p class="note">${esc(t.full_context)}</p></details>
  </section>`;
}

function renderReplay() {
  const t = byId[current], n = t.turns.length, k = steps[current] ?? 0, s = stateAt(t, k);
  $("case").value = current;
  $("prev").disabled = k === 0; $("next").disabled = k === n; $("reset").disabled = k === 0;
  $("step").textContent = n ? `Turn ${k} of ${n}` : "No questions";
  $("prog").style.width = (n ? (k / n) * 100 : 100) + "%";
  $("main").innerHTML = `
    <section class="case-head">
      <div class="eyebrow">${esc(t.case_id)} &middot; ${esc(t.source)}</div>
      <p class="presenting"><b>Given to the agent:</b> ${esc(t.initial_info)}</p>
      <h1>${esc(t.question)}</h1>
      <ul class="options">${Object.entries(t.options).map(([l, v]) =>
        `<li><span class="L">${esc(l)}</span><span>${esc(v)}${l === t.gold_answer ? ' <span class="gold-tag">&#9733; gold</span>' : ""}</span></li>`).join("")}</ul>
    </section>
    <div class="cols">
      <section class="panel"><header><h2>Worksheet &middot; ClinicalHistory</h2></header>
        <div class="hint">Highlighted = filled this turn &middot; grey = still unknown</div>${worksheet(s.snap, s.fresh)}</section>
      <section class="panel"><header><h2>Interview</h2></header><div class="convo">${conversation(t, k)}</div></section>
      <section class="panel"><header><h2>Differential</h2><span class="hint">${k === 0 ? "before any questions (floor)" : "after turn " + k}</span></header>
        ${differential(s.diff, t.gold_answer)}
        <div class="decision">Next step: <b>${s.decision === "answer" ? "answer" : "ask another question"}</b></div></section>
    </div>
    ${k === n ? result(t) : ""}`;
}

function renderSummary() {
  const n = T.length, acc = key => T.filter(t => t[key]).length;
  const avgQ = (T.reduce((a, t) => a + t.num_questions, 0) / n).toFixed(1);
  const cost = T.reduce((a, t) => a + (t.cost_usd || 0), 0);
  const tile = (label, k, sub) => `<div class="tile"><div class="eyebrow">${label}</div><div class="n">${Math.round(100 * k / n)}%</div><div class="s">${k} of ${n} &middot; ${sub}</div></div>`;
  const yn = (ok, l) => `<span class="${ok ? "ok" : "no"}">${ok ? "&#10003;" : "&#10007;"} ${esc(l)}</span>`;
  $("main").innerHTML = `
    <section class="case-head"><div class="eyebrow">Summary &middot; ${n} MediQ cases &middot; ${esc(T[0].model.replace(/^anthropic\//, ""))}</div>
      <h1>How much of the floor-to-ceiling gap does the agent close?</h1>
      <p class="note">Floor: the agent's ranking from age, sex and chief complaint only. Ceiling: the same ranking prompt given the full patient record. Agent: after interviewing the MediQ patient by chart section.</p></section>
    <div class="tiles">
      ${tile("Floor", acc("floor_correct"), "chief complaint only")}
      ${tile("Agent", acc("correct"), "after interviewing")}
      ${tile("Ceiling", acc("ceiling_correct"), "full record up front")}
      <div class="tile"><div class="eyebrow">Questions asked</div><div class="n">${avgQ}</div><div class="s">average per case &middot; total cost $${cost.toFixed(2)}</div></div>
    </div>
    <div class="tablebox"><table>
      <thead><tr><th>Case</th><th>Gold</th><th>Floor</th><th>Agent</th><th>Ceiling</th><th>Questions</th><th>Cost</th></tr></thead>
      <tbody>${T.map(t => `<tr><td><a href="#${esc(t.case_id)}">${esc(t.case_id)}</a></td><td>${esc(t.gold_answer)}</td>
        <td>${yn(t.floor_correct, t.floor_answer)}</td><td>${yn(t.correct, t.final_answer)}</td><td>${yn(t.ceiling_correct, t.ceiling_answer)}</td>
        <td>${t.num_questions}</td><td>$${(t.cost_usd || 0).toFixed(4)}</td></tr>`).join("")}</tbody>
    </table></div>`;
}

function route() {
  const h = decodeURIComponent(location.hash.slice(1));
  const summary = h === "summary";
  if (!summary && byId[h]) current = h;
  $("controls").hidden = summary;
  $("prog").parentElement.hidden = summary;
  $("tab-summary").toggleAttribute("aria-current", summary);
  $("tab-replay").toggleAttribute("aria-current", !summary);
  if (summary) { $("tab-summary").setAttribute("aria-current", "page"); renderSummary(); }
  else { $("tab-replay").setAttribute("aria-current", "page"); renderReplay(); }
}
function go(delta) {
  const n = byId[current].turns.length;
  steps[current] = Math.max(0, Math.min(n, (steps[current] ?? 0) + delta));
  renderReplay();
}
$("tab-replay").addEventListener("click", e => { e.preventDefault(); location.hash = current; });
$("case").addEventListener("change", e => { location.hash = e.target.value; });
$("prev").addEventListener("click", () => go(-1));
$("next").addEventListener("click", () => go(1));
$("reset").addEventListener("click", () => { steps[current] = 0; renderReplay(); });
document.addEventListener("keydown", e => {
  if (location.hash === "#summary" || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowRight") go(1);
  if (e.key === "ArrowLeft") go(-1);
});
window.addEventListener("hashchange", route);
route();
</script>
"""

if __name__ == "__main__":
    main()
