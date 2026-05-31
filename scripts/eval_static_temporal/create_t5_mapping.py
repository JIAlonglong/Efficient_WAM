#!/usr/bin/env python3
"""Create t5_embedding directory mapping episodes to their pre-computed T5 embeddings.

The dataset has pre-computed embeddings in text_emb/ named by task text hash.
This script creates symlinks in t5_embedding/ named by episode_index.
"""
import hashlib
import json
import os
from pathlib import Path

dataset_root = Path("/root/intern/jialongliu/projects/starVLA/data/shared_datasets/robotwin_fastwam/robotwin2.0")
text_emb_dir = dataset_root / "text_emb"
t5_out_dir = dataset_root / "t5_embedding"
t5_out_dir.mkdir(exist_ok=True)

# Build a lookup from task text hash -> file path
hash_to_file = {}
for f in text_emb_dir.glob("*.pt"):
    # filename is "{hash}.t5_len128.wan22ti2v5b.pt"
    # The hash is the SHA256 of the task text
    fname = f.name
    task_hash = fname.split(".")[0]
    hash_to_file[task_hash] = f

print(f"Found {len(hash_to_file)} pre-computed T5 embeddings")

# Load episodes
episodes = []
with open(dataset_root / "meta" / "episodes.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            episodes.append(json.loads(line))

mapped = 0
missing = 0
for ep in episodes:
    ep_idx = ep["episode_index"]
    tasks = ep.get("tasks", [])
    out_path = t5_out_dir / f"episode_{ep_idx:06d}.pt"
    if out_path.exists():
        mapped += 1
        continue

    found = False
    for task_text in tasks:
        task_hash = hashlib.sha256(task_text.encode()).hexdigest()
        if task_hash in hash_to_file:
            # Symlink to the pre-computed embedding
            src = hash_to_file[task_hash]
            try:
                os.symlink(str(src.resolve()), str(out_path))
                mapped += 1
                found = True
                break
            except OSError:
                pass

    if not found:
        missing += 1
        if missing <= 3:
            # Try first task text
            if tasks:
                print(f"  Episode {ep_idx}: no embedding found for task '{tasks[0][:60]}...'")

print(f"Mapped {mapped} episodes, {missing} missing")
