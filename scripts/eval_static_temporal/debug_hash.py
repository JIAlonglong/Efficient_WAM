#!/usr/bin/env python3
"""Debug: check how text_emb filenames relate to task text."""
import hashlib
from pathlib import Path

# Read a few filenames
text_emb_dir = Path("/root/intern/jialongliu/projects/starVLA/data/shared_datasets/robotwin_fastwam/robotwin2.0/text_emb")
filenames = sorted(text_emb_dir.glob("*.pt"))[:5]
for f in filenames:
    hash_part = f.name.split(".")[0]
    print(f"File hash: {hash_part}")

# Read first few tasks
tasks_file = Path("/root/intern/jialongliu/projects/starVLA/data/shared_datasets/robotwin_fastwam/robotwin2.0/meta/tasks.jsonl")
with open(tasks_file) as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        import json
        task = json.loads(line.strip())
        text = task["task"]
        sha256 = hashlib.sha256(text.encode()).hexdigest()
        md5 = hashlib.md5(text.encode()).hexdigest()
        sha1 = hashlib.sha1(text.encode()).hexdigest()
        print(f"Task {task['task_index']}: '{text[:60]}...'")
        print(f"  SHA256: {sha256}")
        print(f"  MD5: {md5}")
        print(f"  SHA1: {sha1}")
