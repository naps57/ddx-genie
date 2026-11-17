# Clinical diagnosis under partial information (CS224V)

Can an agent that notices what is missing and asks for it diagnose better than a one-shot LLM
that answers from whatever it was given? Cases come from a fixed 50-case diagnosis subset of
**MediQ** (MedQA-derived, CC BY 4.0). Each case is shown as partial views revealing a fraction of
its atomic facts.

```
MediQ subset → partial views (reveal p of the facts) → diagnosis (one-shot or agent) → open top-5 → LLM judge vs gold
```

- `src/ddx_genie/mediq.py` builds `data/processed/mediq/subset.jsonl`.
- `src/ddx_genie/reveal.py` builds `data/processed/mediq/views.jsonl`.
- `src/ddx_genie/partial_eval.py` runs the one-shot evaluation; `baselines.py` runs the other baselines.
- `scripts/baseline_report.py` prints the frozen numbers in `baseline_scores.md`.

See `CLAUDE.md` for the design contract.

### Chart-section asker (Genie worksheet arm)

`src/ddx_genie/chart_asker.py` is an asking arm built on the same loop as prompted asking
(`baselines.run_asking`): complaint-only start, k = 1..5 questions, the same fact-select patient,
the same open-ended diagnose + judge after every step. Each question targets one unfilled field of a
`ClinicalHistory` Genie worksheet and is that field's fixed question. Two policies differ only in
which field they pick:

- `checklist`: the first unfilled field in schema order.
- `differential`: one call scores every unfilled field 0–10 by how much it would change the current
  top-5; optionally weighted by a reveal-rate prior estimated on dev (`--prior`).

`specs/chart_sections.yaml` is the schema (sections, fields, descriptions, questions). It is written
out as a 14-column Genie spec, `specs/clinical_history.csv`, and the live worksheet is built from that
CSV by Genie's own `csv_to_classes`. Genie's dialogue loop is not used: its LLM calls would bypass
`llm.py`, and its policy only asks in declaration order.

`data/processed/mediq/dev.jsonl` holds the 59 eligible diagnosis cases not in the frozen 50
(written by `ddx_genie.mediq`). Design and tune on dev only. The CLI refuses `--split test` or more
than 5 cases without `--confirmed`.

```bash
uv run python -m ddx_genie.chart_asker --write-spec                    # after editing the YAML
uv run python -m ddx_genie.chart_asker --run-id chart_dev_smoke --policy checklist --limit 1
uv run python -m ddx_genie.chart_asker --run-id chart_prior_dev --estimate-prior --limit 59 --confirmed
uv run python -m ddx_genie.chart_asker --run-id chart_dev --policy differential \
    --prior data/processed/mediq/reveal_prior_dev.json --limit 5
uv run python scripts/baseline_report.py --chart-run chart_dev            # chart arms only
```

The report's comparisons against prompted asking, random facts and best-k (McNemar, equal-facts,
gap closed, key-fact recall) need the per-case baseline JSONs in `runs/baselines_v1/`. Only
`summary.json` is in git, so these print `n/a` until those files are present.

**Why a reveal-rate prior.** MediQ has no population statistics for P(finding | diagnosis), so the
conditional (how much a finding would move the differential) has to come from the LLM. What the data
*does* ground is how often each chart field is documented at all: we file every non-anchor fact of
each dev case under chart fields and take the Laplace-smoothed share of cases in which each field is
documented. The differential policy ranks fields by expected value = reveal rate × LLM-scored impact,
so it stops spending questions on fields the record rarely contains. The rate is estimated on dev only
and never on the 50 test cases.

### `demo/`: the original prototype

`demo/` is the first, self-contained prototype of the asker, with a turn-by-turn replay app
(`streamlit run demo/app.py`) and a static replay page (`demo/site/index.html`). It has its own setup;
see [demo/README.md](demo/README.md) and [demo/NOTES.md](demo/NOTES.md). Its numbers are **not**
comparable to `baseline_scores.md`: it scores multiple-choice option ranking on 10 different cases
with Claude Haiku. Its traces are kept as an archive.

## Setup

This repo expects the Genie Worksheets clone as a **sibling directory**:

```
224V Project/
  ddx-genie/          # this repo
  genie-worksheets/   # git clone https://github.com/stanford-oval/genie-worksheets.git
```

```bash
cd ddx-genie
uv sync                       # installs genie-worksheets as an editable dependency
cp .env.example .env          # then fill in GEMINI_API_KEY and the price-per-token values
```

Authenticate to Google Cloud (Application Default Credentials — Vertex uses short-lived OAuth
tokens, not a static API key):

```bash
gcloud auth application-default login
gcloud config set project <your-project-id>
gcloud services enable aiplatform.googleapis.com
```

Verify:

```bash
uv run python -c "import worksheets, ddx_genie; print('ok')"
uv run pytest
uv run python scripts/smoke_llm.py   # one real LLM call, well under a cent
```

All LLM traffic goes through `src/ddx_genie/llm.py`. It caches to `.llm_cache/`,
appends a line per call to `runs/<run_id>/cost.jsonl`, and raises `BudgetExceeded`
rather than crossing `MAX_RUN_USD`. `configure_genie_env()` points Genie's own
LangChain client at the same backend and must be called **after** `import worksheets`.

### Optional: free response caching

Genie's own LLM client caches to Redis on `localhost:6379` when `temperature == 0.0`, and silently
disables caching if Redis is absent. Running it locally makes repeated development turns free:

```bash
brew install redis && redis-server --daemonize yes
```

## Notes on the dependencies

- **`psycopg2` override.** Importing `worksheets` pulls in `suql` → `psycopg2` even though we use no
  knowledge base. On macOS the plain `psycopg2` sdist fails at import with
  `Library not loaded: @rpath/libpq.5.dylib`. `pyproject.toml` overrides it out with an impossible
  version marker so only `psycopg2-binary` (which bundles its own libpq) is installed.
- **Stale `VIRTUAL_ENV`.** If another venv is active, `uv pip install` silently targets *that* venv.
  Run `env -u VIRTUAL_ENV uv …` if installs appear to vanish.
- **`ModuleNotFoundError: ddx_genie` on macOS.** Python ≥ 3.13 skips `.pth` files that carry the
  macOS `hidden` flag, and the editable-install `.pth` can pick it up after a re-sync. Fix:
  `chflags nohidden .venv/lib/python3.*/site-packages/*.pth`.

## Corrections to the upstream docs

Verified against the cloned source, not the README or the published docs:

| Claim | Reality |
| --- | --- |
| Repo is `stanford-oval/worksheets` | 404s; it is `stanford-oval/genie-worksheets` |
| `Config(prompt_dir=…)` / `prompt_dir:` in `config.yaml` | **No such field.** `Config` is a pydantic model with `extra="ignore"`, so a `prompt_dir` key is accepted and silently discarded. Custom prompts must be registered with `init_llm(prompt_dir=…, dotenv_path=…)` after importing `worksheets`. Getting this wrong means the agent runs on Genie's bundled prompts with no error. |
| Spec CSV has 13 columns | 14. A `Validation` column follows `Form Actions` (`FIELD_VALIDATION = 13`). Short rows are padded by `fill_all_empty`, so 13 columns happens to work, but write the header in full. |
| `conversation_loop(agent, "state.json", debug=True)` | Signature is `conversation_loop(agent, quit_commands=None, debug=False)`. The per-turn call is `await agent.generate_next_turn(utterance)`; history is `agent.dlg_history`; serialize with `worksheets.utils.interface.convert_to_json(dialogue, session_id)`. |
| Specs require a Google Sheet + service account | `AgentBuilder.with_csv_specification(path)` and `.with_json_specification(path)` both work. |

Genie's LLM client hardcodes `timeout=10` and `max_retries=3` (`src/worksheets/llm/llm.py`). If calls
time out under Vertex latency, monkeypatch from our code — never edit the package source.

## Licence / data

MediQ is CC BY 4.0. Raw downloads live in `data/raw/`, which is gitignored and never committed.
