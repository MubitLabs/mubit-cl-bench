"""Test whether recall() (semantic search + working memory) surfaces a
freshly-remembered item within the same run, and isolates across runs.
This is the candidate fast path if get_context's lesson pipeline is inert.
"""

import json
import os
import time
import uuid

from dotenv import load_dotenv

load_dotenv("/Users/eldaru/Mubit/Benchmarking/continual-learning-bench/.env")
from mubit import Client  # noqa: E402

KEY = os.environ["MUBIT_API_KEY"]
Q = "which table stores office products; schema encodings"


def new_run(client):
    proj = client.create_project({"name": f"clbench-rc-{uuid.uuid4().hex[:8]}"})
    pid = proj["project"]["project_id"]
    sess = client.create_session({"project_id": pid, "agent_id": "probe", "auto_register_agent": True})
    s = sess.get("session", sess)
    return s.get("run_id") or s.get("session_id")


def try_recall(client, run_id, marker, **kw):
    try:
        r = client.recall(session_id=run_id, query=Q, limit=10, **kw)
    except Exception as e:
        return f"ERR {str(e)[:90]}"
    blob = r if isinstance(r, str) else json.dumps(r)
    return f"hit={marker in blob} len={len(blob)}"


c = Client(api_key=KEY)
run = new_run(c)
marker = f"MK{uuid.uuid4().hex[:6]}"
content = f"{marker}: table items_g1 = OFFICE PRODUCTS; g2 prices integer cents."

# Write three ways: lesson(run), lesson(global), and a plain memory (working).
c.remember(session_id=run, agent_id="probe", content=content + " [run]", intent="lesson", lesson_scope="run")
c.remember(session_id=run, agent_id="probe", content=content + " [glob]", intent="lesson", lesson_scope="global")
c.remember(session_id=run, agent_id="probe", content=content + " [plain]")
time.sleep(4)

print("SAME run recall variants:")
print("  default                :", try_recall(c, run, marker))
print("  semantic_search        :", try_recall(c, run, marker, mode="direct", direct_lane="semantic_search"))
print("  entry_types=[lesson]   :", try_recall(c, run, marker, entry_types=["lesson"]))
print("  working_memory only    :", try_recall(c, run, marker, include_working_memory=True, entry_types=[]))

run2 = new_run(c)
print("\nDIFFERENT run recall (want hit=False):")
print("  default                :", try_recall(c, run2, marker))
