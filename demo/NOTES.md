# Notes: MediQ issues and how our wrapper differs from stock MediQ

Everything below refers to MediQ at commit `faa2ce62fef0423e35af4c31d7537aad973173eb`
(stellalisy/mediQ, "add vllm support", 2025-01-26), cloned unmodified into `external/mediQ`.
Line numbers are from that commit.

## 1. Bugs and pitfalls in stock MediQ

### Affect the Patient system (the part we use)

| # | Location | Problem | Effect |
|---|---|---|---|
| 1 | `src/helper.py:1` | `import torch` at module top level; torch is never used on the API path. | Every Patient and Expert imports `helper`, so nothing runs without PyTorch installed, even with an API model. |
| 2 | `src/helper.py:65` | `if self.use_api == "openai": self.openai_generate(messages)` has no `return`. | `generate()` returns `None`; every caller's `response, log_probs, num_tokens = ...` raises `TypeError`. The OpenAI path cannot work. |
| 3 | `src/helper.py:141-144` | Pre-1.0 OpenAI client usage: `response["usage"]["prompt_tokens"]`, `response.choices[0].text`, `.logprobs.top_logprobs`. The pinned environment ships `openai=1.57.4`, where the response is a pydantic object and chat choices have `.message.content`. | Even with bug 2 fixed, the OpenAI path crashes (`TypeError` / `AttributeError`). |
| 4 | `src/patient.py:27` vs data files | Patient reads `sample['atomic_facts']`, but the shipped JSONL key is `facts`. | `self.facts` is always `None`, so `FactSelectPatient` ignores the shipped facts and re-decomposes the context into facts with the patient LLM on the first question (`patient.py:90-97`). |
| 5 | `src/patient.py:29,52-54` vs `src/helper.py:61` | Patient sets `max_length = 50` (and 1000 for fact decomposition) and passes `max_length=`; `ModelCache` only reads `max_tokens` (default 256). | Patient length limits are silently ignored; every patient call uses 256 tokens. The 1000-token fact decomposition is capped at 256, so regenerated facts (bug 4) can be truncated. |
| 6 | `src/helper.py:7-8,153-156` | Model clients are cached in a global dict keyed only by model name; kwargs from the *first* caller win. | If Expert and Patient use the same model, the Patient silently inherits the Expert's temperature / max_tokens / API account. |
| 7 | `src/helper.py:33-34`, `src/keys.py`, `src/args.py:38`, `src/patient.py:54` | API key looked up as `mykey[api_account]`; keys.py only defines `"mediQ"`; the Expert passes `api_account` but the Patient does not (defaults to `"openai"`). | A Patient that creates its client first fails with `KeyError: 'openai'`. |
| 8 | `src/helper.py:151` | `if 'gpt' in model_name or 'o1' in model_name: use_api = "openai"`; no `base_url` support. | Substring routing overrides flags; no OpenAI-compatible server or other providers. |

### Elsewhere (not on our path, but worth knowing)

| # | Location | Problem |
|---|---|---|
| 9 | `src/args.py:42-45`, `src/mediQ_benchmark.py:91` | `os.makedirs(os.path.dirname(path))` on default filenames like `log.log` gives `dirname == ""` and raises `FileNotFoundError`. All log/output paths must include a directory. |
| 10 | `src/args.py:29` vs `src/expert_functions.py:6,263` | `--abstain_threshold` defaults to 0.8, which overrides `SCALE_THRESHOLD = 4.0`; `ScaleExpert` (1-5 Likert) then effectively never abstains unless `--abstain_threshold 4` is passed. |
| 11 | `src/evaluate.py:48-50` | Evaluation metrics are `np.random.random()` placeholders; the file also imports `sentence_transformers`, which is not in `environment.yml`. Not wired into the benchmark. |
| 12 | `src/prompts.py` | `conformal_scores["prompt_score"]` (per-option likelihoods) and `expert_system["non_interactive"]` are defined but unused. There is no built-in per-option ranking and no full-information baseline. |
| 13 | `environment.yml` | Fully pinned linux-64 conda export with CUDA PyTorch and vLLM; there is no CPU or API-only install path. |
| 14 | data | `all_dev_good.jsonl` (MedQA, 1,272 cases) has 3 records with empty `facts` and a few with only 2-3. Both files are dev-style only; there is no separate test split in the repo. |

## 2. How our wrapper differs from stock MediQ

**What is unchanged:** we run MediQ's own `FactSelectPatient` class from `external/mediQ/src/patient.py`, unmodified, including its prompts, its "cannot answer" response, its initial-info rule (`context[0]`), and the MediQ data files.

**What differs:**

1. **LLM backend.** Before importing `patient.py` we register a stand-in `helper` module in `sys.modules` whose `get_response` calls LiteLLM (`demo/mediq_adapter.py`). This avoids bugs 1, 2, 3, 6, 7 and 8 without editing MediQ. Patient and Expert keep separate settings.
2. **Shipped facts are used.** We copy `facts` to `atomic_facts` on the sample before constructing the patient (fixes bug 4). The patient therefore never regenerates facts, and every run sees the same fact list.
3. **Patient max tokens = 256.** This reproduces the *effective* stock behaviour (bug 5), not the intended 50. It can be changed via `patient_max_tokens` in `demo/config.yaml`.
4. **Model and sampling.** The default is `claude-haiku-5-5` (via LiteLLM) for both Expert and Patient. Current Claude models reject any non-default `temperature`, so we send none (`temperature: null` in config) and runs are **not deterministic**: the same case can produce different questions and confidences on a re-run. Stock MediQ defaults to temperature 0.6, top_p 0.9. For OpenAI models set `temperature: 0.0`.
5. **Case selection.** Cases come from `all_dev_good.jsonl` (MedQA) by default. We skip records with fewer than 3 shipped facts (bug 14) and sample with `random.Random(seed)`. Stock MediQ iterates every record in file order.
6. **Expert.** We do not use any MediQ Expert class. `demo/agent.py` is our own minimal chart-section agent, a placeholder until Arda's baseline is integrated:
   - It writes questions targeted at unfilled fields of a Genie `ClinicalHistory` worksheet.
   - It tags patient answers into worksheet fields with an extra LLM call; the raw output is logged in the trace.
   - It ranks all options with probabilities after every turn.
   - Default `stop_rule: fixed`: it always asks `--max-questions` questions (or stops early only when no askable fields remain), like MediQ's `FixedExpert`. We chose this because, with a confidence threshold of 0.8, Haiku was often 72-80% confident from the chief complaint alone and asked 0-1 questions.
   - `stop_rule: confidence` additionally stops once the top option reaches `answer_confidence_threshold`.
   Stock MediQ experts instead produce one letter per turn plus a model-specific abstention score.
7. **Floor and ceiling** use the same option-ranking prompt as the agent:
   - The floor is the agent's own turn-0 ranking from the initial info only. It is equivalent in spirit to `FixedExpert` with `--max_questions 0`, but with our prompt.
   - The ceiling gives the full `context` up front.
   Stock MediQ has no ceiling run.
8. **Final answer** is the argmax of the last differential, not a separate "answer" prompt as in MediQ's experts.
9. **Output.** We write one JSON trace per case to `demo/traces/` (see README), instead of MediQ's JSONL with `interactive_system` / `info`. Token usage and cost are recorded per case.

## 3. Genie Worksheets usage

- Installed as the real package (`genie-worksheets` at commit `af3f2d32c198a9c03d84f473762e3c243dae86c9`).
- `ClinicalHistory` is built with Genie's Python API from `demo/chart_sections.yaml`: nested `GenieWorksheet`s (one per chart section) with `GenieField(str, ...)` leaves.
- State is read through Genie: a field is unfilled iff its value is `None`; fields are enumerated with `get_genie_fields_from_ws`; completeness comes from `is_complete`.
- We do **not** use Genie's agent loop (semantic parser, policy, response generator):
  - It calls LLMs through LangChain with Azure/OpenAI configs, not LiteLLM.
  - Its policy asks the first unfilled field in declaration order.
  The field tagger and question selection live in `demo/agent.py`.
- Genie has no YAML spec format (it reads CSV, JSON or Google Sheets), so `chart_sections.yaml` is our own format.
- Packaging bug: `worksheets/llm/llm.py` imports `overrides`, which is not declared in Genie's
  `pyproject.toml`; `import worksheets` fails with `ModuleNotFoundError` until it is installed
  (we list it in `demo/requirements.txt`). Also, plain `pip` stalled for 6+ minutes resolving
  Genie's dependency tree; `uv pip install` (Genie's recommended tool) resolves it in under 2.
- `import worksheets` imports the full agent stack (suql, langchain, chainlit, ...) and loads a `.env` from the package directory, even though we only use the data model. Genie logs every field lookup at DEBUG via loguru; `demo/worksheet.py` raises that to WARNING.
