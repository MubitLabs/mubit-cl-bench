"""Probe project-scoped isolation + retrieval for the gain experiment.

Confirms (1) a fresh project starts empty (isolates from leaked global
lessons), (2) a written lesson becomes retrievable within the same run via
get_context (polling for indexing latency), and (3) a second fresh project is
clean. No LLM calls.
"""

import os
import time
import uuid

from dotenv import load_dotenv

load_dotenv("/Users/eldaru/Mubit/Benchmarking/continual-learning-bench/.env")
from mubit import Client  # noqa: E402

KEY = os.environ["MUBIT_API_KEY"]
Q = "which table stores office products; schema tables columns encodings"


def new_project_session(client, label):
    proj = client.create_project({"name": f"clbench-probe-{label}-{uuid.uuid4().hex[:8]}"})
    pid = proj["project"]["project_id"]
    sess = client.create_session(
        {"project_id": pid, "agent_id": "probe", "auto_register_agent": True}
    )
    sdata = sess.get("session", sess)
    run_id = sdata.get("run_id") or sdata.get("runId") or sdata.get("session_id")
    return pid, run_id


def ctx_lessons(client, run_id, query):
    ctx = client.get_context(session_id=run_id, query=query, mode="full", max_token_budget=600)
    if not isinstance(ctx, dict):
        return 0, ""
    counts = ctx.get("source_counts_by_entry_type") or {}
    return int(counts.get("lesson", 0)), (ctx.get("context_block") or "")


client = Client(api_key=KEY)
marker = f"MARKER{uuid.uuid4().hex[:6]}"
lesson = f"{marker}: table items_g1 stores OFFICE PRODUCTS; g2 prices are integer cents."

pid, run_id = new_project_session(client, "main")
print(f"project_id={pid}\nrun_id={run_id}")

n_before, _ = ctx_lessons(client, run_id, Q)
print(f"\n[fresh project] lessons BEFORE write: {n_before}   (want 0 == isolated from global)")

client.remember(session_id=run_id, agent_id="probe", content=lesson,
                intent="lesson", lesson_scope="global")
print("remember() returned; polling get_context up to 25s for retrievability…")
deadline = time.time() + 25
seen = False
while time.time() < deadline:
    n, block = ctx_lessons(client, run_id, Q)
    if marker in block or n > n_before:
        print(f"  ✓ retrievable after {25 - (deadline - time.time()):.1f}s: lessons={n}, marker_in_block={marker in block}")
        seen = True
        break
    time.sleep(1.0)
if not seen:
    n, block = ctx_lessons(client, run_id, Q)
    print(f"  ✗ NOT retrievable within 25s: lessons={n}, marker_in_block={marker in block}")

pid2, run_id2 = new_project_session(client, "baseline")
n2, _ = ctx_lessons(client, run_id2, Q)
print(f"\n[second fresh project] lessons: {n2}   (want 0 == clean baseline)")

print(f"\nSUMMARY: fresh_empty={n_before == 0}, within_run_retrievable={seen}, "
      f"baseline_clean={n2 == 0}")
