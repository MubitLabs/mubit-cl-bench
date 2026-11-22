# Mubit on CL-BENCH — experiment guide & handoff

**Goal:** Reproduce the *Continual Learning Bench* (CL-BENCH, arXiv 2606.05661) experiment
but with **Mubit** as the continual-learning system, on the **Database Exploration** task,
backbone **Gemini 3 Flash**. Measure the paper's headline metrics — **reward** and
**gain** (stateful reward − stateless reward, isolating learning from raw model ability).

This file is self-contained: a fresh session should be able to read it and continue.

---

## 1. TL;DR status

- **Approach:** we did **not** re-implement the benchmark. We cloned the official repo
  (`github.com/pgasawa/continual-learning-bench`, this directory) and wrote a **new Mubit
  *system*** that plugs into the official harness. So the *real* environment, questions,
  scoring, and the stateless-baseline/gain machinery are reused; only the Mubit system is ours.
- **What works:** Mubit system (`src/systems/mubit/`) runs end-to-end on the real task with
  Gemini 3 Flash. Stateful run accumulates lessons and injects them; the stateless baseline is
  verified to carry **zero** memory; gain is computed correctly.
- **Smoke result (8 questions, 1 rollout, no drift — directional only):**

  | Arm | Stateful (mean reward) | Stateless baseline (mean) | **Gain** | Cost | LLM calls |
  |---|---|---|---|---|---|
  | **Mubit** + Flash | 0.083 | 0.142 | **−0.467** | $0.30 | 69 |
  | **ICL** (full-context) + Flash | 0.117 | 0.100 | **+0.133** | $0.055 | 31 |

  Even at n=8 this mirrors the paper's headline: naive ICL shows positive gain; the dedicated
  memory system (Mubit) shows *negative* gain at ~5× cost (memory introduced spurious
  confidence — e.g. answered q3 in 1 query → wrong; missed q4 the stateless arm got right).
  **n=8/1-rollout is NOT statistically meaningful** — it validates wiring + direction only.

- **Next step not yet run:** calibration (40 questions + Q20 schema drift, 1 rollout, both arms),
  then the full 5-rollout headline run. See §6.

---

## 2. CRITICAL Mubit gotcha (read before changing memory code)

On this Mubit account/SDK (`mubit-sdk==0.7.0`), the semantic-retrieval endpoints
**`get_context` and `recall` do NOT surface a run's freshly-`remember`'d lessons.** They
fall back to a fixed pool of ~8 **account-global** lessons (unrelated to this task), and a
freshly created *project* does **not** isolate from that pool (its query weights are all 0).
FirstTry's old probe only "passed" because its success check (`n_lessons >= 1`) was satisfied
by that global pool — a false positive.

**The working primitive is `client.lessons({"run_id": <run>})`** — strictly run-scoped,
immediately consistent, unaffected by global memory. Our system uses:
- **write:** `remember(content=…, intent="lesson", lesson_scope="run")`
- **read:** `lessons({"run_id": <run>, "limit": N})`
- **isolation:** a unique `run_id` (uuid) per logical run. A fresh run lists 0 lessons →
  the per-instance stateless baseline is genuinely stateless → gain is clean.

Evidence is reproducible via the probe scripts (no LLM cost):
```bash
.venv/bin/python scripts/probe_mubit_isolation.py   # global lessons leak into fresh runs
.venv/bin/python scripts/probe_mubit_project.py      # fresh project does NOT isolate; writes not retrievable via get_context
.venv/bin/python scripts/probe_mubit_recall.py       # recall also doesn't surface fresh writes
```
If you ever switch to `get_context`/`recall`, re-verify with a fresh-vs-written run probe first.

---

## 3. Setup from scratch (new machine / new session)

Prereqs: `uv`, Docker not needed for Database Exploration (it's SQLite). Python 3.13 is
fetched by `uv`. The two `.db` files (~400 MB each) come from HuggingFace.

```bash
cd /Users/eldaru/Mubit/Benchmarking/continual-learning-bench

# 1. Python 3.13 venv + all deps
uv sync --all-extras

# 2. Mubit SDK (pinned to the version whose API we coded against)
uv pip install "mubit-sdk==0.7.0"

# 3. Secrets: .env at repo root with GEMINI_API_KEY and MUBIT_API_KEY.
#    (Copied from ../mubit-eval/mubit-eval/.env during the original build.)
#    cli.py load_dotenv()s this automatically when run from the repo root.
cat .env   # should show GEMINI_API_KEY=… and MUBIT_API_KEY=…

# 4. Download the task data from HuggingFace (products.db, products_drifted.db)
.venv/bin/clbench setup database_exploration
ls -la data/database_exploration/*.db   # products.db, products_drifted.db present

# 5. Sanity: the mubit system is discovered
.venv/bin/clbench inspect system mubit
```

Backbone model id (LiteLLM, Google AI Studio via `GEMINI_API_KEY`):
`gemini/gemini-3-flash-preview`.

---

## 4. How to run

All commands assume cwd = repo root and use `.venv/bin/clbench`.

**Smoke (no drift, tiny, ~$0.05–0.30/arm)** — uses the custom `smoke` schedule we added:
```bash
.venv/bin/clbench run database_exploration --system mubit \
  --system.model gemini/gemini-3-flash-preview \
  --task.schedule smoke --task.num-questions 8 --runs 1 --max-workers 4 --no-live-dashboard

.venv/bin/clbench run database_exploration --system icl \
  --system.model gemini/gemini-3-flash-preview --system.max-tokens 200000 --system.reserve-tokens 4096 \
  --task.schedule smoke --task.num-questions 8 --runs 1 --max-workers 4 --no-live-dashboard
```

**Calibration (full task, with Q20 drift, 1 rollout, ~$3–6/arm):** the real 40-question
schema-drift schedule is `default`. Override `--runs 1`:
```bash
.venv/bin/clbench run database_exploration --config configs/database_exploration/database_exploration_mubit.json --runs 1 --max-workers 8
.venv/bin/clbench run database_exploration --config configs/database_exploration/database_exploration_icl.json  --runs 1 --max-workers 8
```

**Full headline run (paper protocol: 40q drift, 5 permuted rollouts, 95% CIs, ~$15–30/arm,
~1.5–3 h):** the `default` schedule's `run_defaults` is already `runs=5, mode=permute`, so just:
```bash
.venv/bin/clbench run database_exploration --config configs/database_exploration/database_exploration_mubit.json
.venv/bin/clbench run database_exploration --config configs/database_exploration/database_exploration_icl.json
```
Run these in the background and watch the log; each prints a final **Gain / Reward / Baseline**
block and a viewer artifact path.

**Flag notes:** dotted overrides use **hyphens**, e.g. `--task.num-questions`, `--system.max-tokens`
(not underscores — that errors). A schedule is **mandatory** (`--task.schedule` or a `--config`
that sets it). `--no-live-dashboard` keeps stdout clean for logging.

**Reading results:** the CLI prints `Final cumulative gain`, `Final cumulative reward`,
`Baseline reward total`, and per-run score. Full traces land in
`results/database_exploration/traces/<group>/` (`run_0000.json` = stateful rollout,
`baseline.json` = merged stateless baseline). To inspect lesson injection / outcomes, see the
trace-reading snippet in §8.

---

## 5. What we built (file inventory)

New / modified files (everything else is upstream CL-BENCH):

| Path | Purpose |
|---|---|
| `src/systems/mubit/system.py` | **The Mubit system.** Backbone LLM + Mubit memory (`remember`/`lessons`). |
| `src/systems/mubit/__init__.py` | Registers the system (`from .system import MubitMemorySystem`). |
| `src/systems/mubit/README.md` | System doc + params. |
| `configs/database_exploration/database_exploration_mubit.json` | Full-experiment config: Mubit + Flash + `default` (drift) schedule. |
| `configs/database_exploration/database_exploration_icl.json` | Full-experiment config: full-context ICL + Flash. |
| `src/tasks/database_exploration/schedules/smoke.json` | Tiny no-drift schedule for wiring checks. |
| `scripts/probe_mubit_isolation.py` | Probe: global lessons leak into fresh runs. |
| `scripts/probe_mubit_project.py` | Probe: fresh project does not isolate; `get_context` misses fresh writes. |
| `scripts/probe_mubit_recall.py` | Probe: `recall` also misses fresh writes. |
| `MUBIT_EXPERIMENT.md` | This file. |

---

## 6. Open decisions / next steps

1. **Calibration run (recommended next):** 40q + drift, 1 rollout, both arms. Confirms whether
   the negative-Mubit-gain holds on the full task (incl. the migration). Cheap.
2. **Full headline run:** 40q drift × 5 rollouts, both arms → the publishable gain/reward + CIs.
3. **Iterate on the Mubit system (optional, to give Mubit its best shot):** ideas, in
   `src/systems/mubit/system.py`:
   - Only `remember` a lesson when the instance was answered **correctly** (the feedback string
     starts with "CORRECT") — avoids memorizing wrong facts (the main cause of negative gain).
   - Dedup / consolidate lessons (the model writes near-duplicates); or periodically `reflect`.
   - Tighten `_LESSON_SYSTEM_PROMPT` to forbid speculative facts.
   - Tune `max_lessons` / `max_lessons_chars`.
   Re-run calibration after each change.
4. **Scope expansion (was deferred):** only Database Exploration is wired. Other tasks
   (poker, BSM, etc.) would reuse the SAME Mubit system unchanged (it's task-agnostic) but most
   need Docker / extra data. Decide if/when.

---

## 7. Method notes (for faithfulness to the paper)

- **Gain** = stateful reward − stateless reward, per instance, summed. The framework computes the
  stateless baseline by constructing a **fresh system per instance** in its own process and
  running that instance alone. Our `reset()`/`__init__` roll a fresh Mubit `run_id`, so each
  baseline instance has empty memory → stateless. Verified: `baseline.json` shows
  `lessons_injected = 0` on every interaction.
- **Reward** for Database Exploration = `1 − exploratory_queries_used / 15` on a correct answer;
  `0` (full regret) on wrong/timeout/over-budget. Run score = `1 − cumulative_regret/(N×15)`.
- **rollouts/permutation:** the `default` schedule permutes question order within each
  pre/post-drift stage across rollouts (paper's noise-reduction); baseline is run once.
- **Cost:** all billable LLM calls (response turns + Mubit's lesson-extraction call) record
  `UsageEvent`s, so the printed cost is comparable to the paper's per-system cost.

---

## 8. Handy trace-inspection snippet

```python
import json, os
tr = "results/database_exploration/traces/<GROUP>"          # fill in
d = json.load(open(os.path.join(tr, "run_0000.json")))      # stateful rollout
print([(o['instance_id'], o['reward'], o['success']) for o in d["instance_outcomes"]])
# lesson accumulation: metadata.lessons_injected / lessons_written_total per interaction
b = json.load(open(os.path.join(tr, "baseline.json")))      # stateless baseline
maxinj = max((((i.get("response",{}) or {}).get("metadata",{}) or {}).get("lessons_injected") or 0)
             for i in b["interactions"])
print("baseline max lessons_injected (must be 0):", maxinj)
```
