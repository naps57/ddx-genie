# John's TODO: chart-section asker as a baseline arm

Context for the agent: read `CLAUDE.md`, `README.md`, `baseline_scores.md`, `demo/README.md` and
`demo/NOTES.md` first. Follow every hard rule in `CLAUDE.md`: all LLM calls through
`src/ddx_genie/llm.py`, cheapest model (`gemini-3.1-flash-lite`), budget guard, cost logged to
`runs/<run_id>/cost.jsonl`, develop on N=1 to 5 cases, and **never launch more than 5 cases without
asking me first**. Do not edit the genie-worksheets source or the frozen numbers in `baseline_scores.md`.
Stop and show me a plan before step 2.

> **Status (2026-11-16):** steps 1–5 are built and unit-tested offline (103 tests pass, no LLM
> calls made). Step 6 is blocked on a Gemini key in `.env`. Steps 7–8 need real runs.

## 0. Admin (me, not the agent)
- [ ] Send the weekly update (was due Mon Nov 16, 12:00 PM PT): demo merged into `demo/`, next step is
      porting it into the harness as an arm on the frozen 50.
- [ ] Confirm with Arda: is he building the differential-driven Genie arm, or am I building both policies?
      Agree on a branch name.
- [ ] **New:** get a Gemini key into `.env` (copy `.env.example`), or have Arda run step 6.
- [ ] **New:** ask Arda for the per-case `runs/baselines_v1/{asking,random,bestk}/*.json` and
      `runs/mediq_open_v1/*.json` (and ideally `.llm_cache/`), zipped, not committed. Without them the
      McNemar, equal-facts, gap-closed and key-fact-recall comparisons print `n/a`.

## 1. Verify the held-out split (no LLM calls) ✅
- [x] Confirm `demo/`'s case id (`rec['id']` in `demo/mediq_adapter.py`) and `source_id` in
      `data/processed/mediq/subset.jsonl` use the same numbering, then check the 10 cases in
      `demo/traces/` don't overlap the frozen 50.
      → Same numbering (MediQ record `id` in `all_dev_good.jsonl`, both at MediQ `faa2ce6`; the
      frozen subset regenerates byte-identical). No overlap. Demo cases 982 and 1001 are in dev.
- [x] Build a dev set: the remaining ~59 diagnosis cases from the same filter as `mediq.py`, minus the 50.
      Save as `data/processed/mediq/dev.jsonl`. All design and debugging of fields/prompts uses dev only.
      → 59 cases, `mediq.dev_split()`, tested disjoint and complete.

## 2. Port the asker into the harness as an arm ✅ (offline)
- [x] Show me a plan: drive questions through Genie's agent loop vs. use the Genie worksheet only as a
      state object (what `demo/` does now). Include the tradeoff and which is less work by Dec 2.
      → Approved: worksheet as state object, built from a Genie spec CSV by Genie's own loader.
- [x] Move the `ClinicalHistory` worksheet into the package: `src/ddx_genie/chart_asker.py`, schema in
      `specs/chart_sections.yaml` (26 fields, 8 sections, one fixed question per field), 14-column
      Genie spec at `specs/clinical_history.csv` (`--write-spec`).
- [x] Implement the same loop as `baselines.run_asking`: complaint-only start, k = 1..5, same
      `patient_answer`, same `score_facts` (open top-5 + judge) after every step. Each question
      targets one unfilled worksheet field.
- [x] Fill worksheet fields from each patient answer with one tagging call (demo tagger, ported).
      Fixed a demo bug on the way: re-tagging a value already present overwrote the field.

## 3. Two policies ✅ (offline)
- [x] `checklist` (fixed order, my arm): fixed section order, ask the first unfilled field.
- [x] `differential` (info-gain style): one call scores every unfilled field 0–10 by how much it would
      change the current top-5; ask the highest. All scores logged; falls back to checklist order
      (and logs it) if the call returns nothing usable.
- [x] Both selectable by `--policy`; same everything else, including the question wording.
- [ ] **Check on dev:** with 8 HPI fields first, `checklist` will likely spend all 5 questions in HPI
      and never reach exam/labs. That is a faithful fixed order but maybe a weak one. Decide after
      the 5-case dev run whether the order should be section round-robin instead.

## 4. Grounded probabilities ✅ code, ⏳ run
- [x] Code: `--estimate-prior` tags every dev case's hidden facts into fields (1 call per case) and
      writes `data/processed/mediq/reveal_prior_dev.json` (Laplace-smoothed per-field and per-section
      rates). The `differential` policy uses expected value = reveal rate × LLM impact via `--prior`.
      Observed P(asking a field reveals a fact) is in each run's summary (`hits_by_field`).
- [x] Run it on all 59 dev cases → `reveal_prior_dev.json` (59 calls, $0.021). Section rates:
      ExamLabs 0.93, HPI 0.87, PMH 0.59, ROS 0.44, Medications 0.39, Social 0.39, Family 0.10,
      Allergies 0.02.
- [x] Paragraph for the paper: in README.md, "Why a reveal-rate prior".

## 5. Metrics ✅ code
- [x] Per k = 0..5: top-1, top-5, P(gold), Wilson / bootstrap CIs (`baseline_report.py --chart-run`).
- [x] Gap closed on P(gold) at k=3, vs prompted asking (27%) and random facts (40%).
- [x] Question hit rate (vs 43%) and key-fact recall (vs 0.42).
- [x] Per-question and per-fact (equal facts obtained vs random) reported separately.
- [x] Paired exact McNemar on top-1 vs prompted asking, per k.
      → The last four need Arda's per-case baseline files (see step 0).

## 6. Runs (ask me before each one over 5 cases)
- [x] Smoke test: 1 case per policy on dev ($0.0065). Run as
      `PYTHONPATH=src:../genie-worksheets/src .venv/bin/python -m ddx_genie.chart_asker ...`
      (macOS keeps re-hiding the venv's `.pth` files under iCloud `~/Documents`).
- [x] 5 dev cases per policy (`runs/chart_dev5`, `runs/chart_dev5_noprior`, $0.06 total).
      Top-1 out of 5 at k=0..5: checklist 1,1,1,2,3,3; differential+prior 1,3,5,4,4,4;
      differential without prior 1,2,4,5,4,4. Hit rate 15/25, 18/25, 16/25. Checklist asked
      23/25 questions in HPI; the prior moved the differential policy toward ExamLabs (12/25).
      n=5 is anecdotal; do not read effect sizes from it.
- [ ] Decide the checklist order (HPI-first vs a section round-robin) on dev before the 50.
- [ ] Look at why accuracy dips after it peaks (differential+prior 5/5 at k=2, 4/5 at k=5).
- [ ] **Ask me**, then the frozen 50, new run ids (e.g. `runs/chart_checklist_v1`,
      `runs/chart_differential_v1`). Report via `scripts/baseline_report.py --chart-run`.

## 7. Per-step traces and demo
- [x] Per-case JSON per step: section, field, question, patient answer, facts gained, worksheet
      snapshot, top-5 differential, policy scores. (Judge relation labels are not stored, only the
      match rank, the same as Arda's `Scored`.)
- [ ] Point `demo/app.py` and `demo/export_html.py` at these traces so the demo replays the real arm
      (open-ended top-5 instead of 4 options). Keep the old Haiku traces only as an archive.

## 8. Before Dec 2
- [ ] Hand-check 30 judge decisions via `runs/<run>/judge_review.csv`.
- [ ] One stronger-model run on the best policy as a robustness check (ask me first: cost).
- [ ] Results table + 2 or 3 replay cases picked for the presentation.
