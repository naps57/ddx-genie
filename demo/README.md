# MediQ chart-section demo

This is a replayable demo of a clinician-facing agent that interviews a simulated MediQ patient chart section by chart section. The agent tracks what it has learned in a Genie Worksheet (`ClinicalHistory`) and then commits to a multiple-choice answer.

The demo has two parts:

- `run_cases.py` generates traces. It is the only part that calls LLMs.
- `app.py` is a Streamlit app that replays those traces turn by turn, with no LLM calls.

> The agent in `agent.py` is a minimal placeholder baseline. `ChartSectionAgent.run_case` is the single swap point for Arda's baseline. For how we wrap MediQ, the MediQ bugs we found, and how this setup differs from stock MediQ, see [NOTES.md](NOTES.md).

## Setup

You need Python 3.11 or newer (we use 3.13) and no GPU.

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install uv && uv pip install -r demo/requirements.txt   # plain pip can stall resolving Genie deps

# MediQ, unmodified, pinned (data + Patient code)
git clone https://github.com/stellalisy/mediQ.git external/mediQ
git -C external/mediQ checkout faa2ce62fef0423e35af4c31d7537aad973173eb

cp .env.example .env   # then set ANTHROPIC_API_KEY (or the key for whichever model you configure)
```

`demo/config.yaml` holds the settings:

| Setting | Meaning |
|---|---|
| `model` | Expert model, as a LiteLLM model string |
| `patient_model` | Model behind MediQ's patient |
| `stop_rule` | `fixed` (default): always ask `--max-questions`. `confidence`: also stop at `answer_confidence_threshold` |
| `patient_max_tokens` | Token cap for each patient answer |

Change `model` to swap the whole pipeline to another provider.

`demo/chart_sections.yaml` defines the chart sections and fields, along with the opening and follow-up questions. The field descriptions are what the tagger uses to decide whether an answer fills a field. You can edit this file without touching any code.

## Generate traces

```bash
python demo/run_cases.py --n 10 --max-questions 10 --seed 0
python demo/run_cases.py --ids medqa-12 medqa-40      # rerun specific cases
python demo/run_cases.py --n 5 --source craftmd       # Craft-MD instead of MedQA
```

For each case the script runs three things:

- **Floor:** the agent's option ranking from the initial info only (age, sex, chief complaint), with zero questions.
- **Agent:** the interactive run.
- **Ceiling:** the same ranking prompt, given the full patient record.

It writes `demo/traces/<case_id>.json` and prints floor, agent and ceiling accuracy, average questions asked, and the total cost.

With the default model (Claude Haiku 5.5, about $0.01 per case at 10 questions) each case makes about 4 LLM calls per question:

- question writer
- patient
- field tagger
- option ranker

## Replay app

```bash
streamlit run demo/app.py
```

The sidebar has a case picker (✅/❌ shows whether the agent was correct), the model name, and Prev / Next / Reset buttons. Step 0 shows the initial presentation, and each Next adds one question/answer turn. The main view has three columns:

- **Worksheet:** fields grouped by chart section. Fields filled on the current turn are highlighted and unknown fields are grey.
- **Conversation:** each question is tagged with the chart section and field it targets. An expander shows the tagger's raw output.
- **Differential:** confidence for each option. The gold answer is marked with ★ and a separate color.

After the last turn, the app shows the agent's answer next to gold, the floor and ceiling answers, and why the agent stopped. The **Summary** view shows floor, agent and ceiling accuracy across all loaded traces.

## Trace format

Each trace contains the fields from the project spec:

`case_id, source, model, initial_info, question, options, gold_answer, turns[], final_answer, correct, num_questions, floor_answer, ceiling_answer`

Each turn has:

`turn, chart_section, target_field, question, patient_answer, fields_filled, worksheet_snapshot, differential, decision`

The traces also carry extra fields for inspection:

- **Initial state:** `initial_worksheet_snapshot`, `initial_differential`, `initial_tagger_raw`.
- **Raw model outputs per turn:** `tagger_raw`, `ranker_raw`, `question_writer_raw`.
- **Outcome:** `stop_reason`, `floor_correct`, `ceiling_correct`, `ceiling_differential`.
- **Inputs:** `full_context`, `patient_facts`.
- **Run settings:** `run`, which records usage and cost per case.
