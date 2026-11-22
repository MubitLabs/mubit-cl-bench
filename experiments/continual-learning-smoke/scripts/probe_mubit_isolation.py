"""Probe Mubit memory isolation for the gain experiment.

Decides whether a unique run_id per logical run is enough to make the
stateless baseline truly stateless, or whether we need a fresh project.
No LLM calls — only Mubit API.
"""

import os
import uuid

from dotenv import load_dotenv

load_dotenv()
from mubit import Client  # noqa: E402

KEY = os.environ["MUBIT_API_KEY"]
Q = "office products table items columns schema encodings"


def ctx_lessons(client, run_id, query):
    ctx = client.get_context(
        session_id=run_id, query=query, mode="full", max_token_budget=600
    )
    block = (ctx.get("context_block") or "") if isinstance(ctx, dict) else ""
    counts = ctx.get("source_counts_by_entry_type") or {}
    return int(counts.get("lesson", 0)), block


run_a = f"probe-A-{uuid.uuid4().hex}"
run_b = f"probe-B-{uuid.uuid4().hex}"
marker = f"MARKER_{uuid.uuid4().hex[:6]}"

ca = Client(api_key=KEY, run_id=run_a)
n0, _ = ctx_lessons(ca, run_a, Q)
print(f"A) fresh run {run_a[:14]}… BEFORE any write: lessons={n0}")

ca.remember(
    session_id=run_a,
    agent_id="probe",
    content=f"{marker}: in this DB, table items_g1 stores OFFICE PRODUCTS.",
    intent="lesson",
)
n1, b1 = ctx_lessons(ca, run_a, Q)
print(f"B) same run AFTER write (default scope): lessons={n1}, marker_in_block={marker in b1}")

cb = Client(api_key=KEY, run_id=run_b)
n2, b2 = ctx_lessons(cb, run_b, Q)
print(f"C) DIFFERENT fresh run {run_b[:14]}…: lessons={n2}, A's marker leaked={marker in b2}")

print("\nVERDICT:")
if marker in b2:
    print("  ✗ run-scoping does NOT isolate: A's lesson leaked into run B. Need project isolation.")
elif n2 > 0:
    print(f"  ⚠ run B sees {n2} lessons (not A's marker) — pre-existing GLOBAL lessons leak. Need fresh project for clean baseline.")
else:
    print("  ✓ run-scoping isolates: fresh run sees 0 lessons; within-run retrieval works. Unique run_id per logical run is sufficient.")
