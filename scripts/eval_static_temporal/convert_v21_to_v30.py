#!/usr/bin/env python3
"""Convert LeRobot v2.1 dataset metadata to v3.0 format in-place."""
import json
import numpy as np
import pandas as pd
from pathlib import Path

dataset_root = Path("/root/intern/jialongliu/projects/starVLA/data/shared_datasets/robotwin_fastwam/robotwin2.0")
meta_dir = dataset_root / "meta"

# 1. Convert tasks.jsonl -> tasks.parquet
tasks = []
with open(meta_dir / "tasks.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            tasks.append(json.loads(line))
tasks_df = pd.DataFrame(tasks)
tasks_df.to_parquet(meta_dir / "tasks.parquet", index=False)
print(f"Created tasks.parquet with {len(tasks_df)} rows")

# 2. Convert episodes.jsonl -> meta/episodes/chunk-000/file-000.parquet
episodes = []
with open(meta_dir / "episodes.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            episodes.append(json.loads(line))
episodes_df = pd.DataFrame(episodes)

with open(meta_dir / "info.json") as f:
    info = json.load(f)
chunks_size = info.get("chunks_size", 1000)

episodes_df["data/chunk_index"] = (episodes_df["episode_index"] // chunks_size).astype(int)
episodes_df["data/file_index"] = (episodes_df["episode_index"] % chunks_size).astype(int)
episodes_df["meta/episodes/chunk_index"] = (episodes_df["episode_index"] // chunks_size).astype(int)
episodes_df["meta/episodes/file_index"] = (episodes_df["episode_index"] % chunks_size).astype(int)

# Add num_frames from episodes_stats.jsonl
ep_lengths = []
with open(meta_dir / "episodes_stats.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            ep_data = json.loads(line)
            stats = ep_data.get("stats", {})
            first_key = list(stats.keys())[0]
            count = stats[first_key].get("count", [0])[0]
            ep_lengths.append(int(count))
episodes_df["num_frames"] = ep_lengths[: len(episodes_df)]

ep_dir = meta_dir / "episodes" / "chunk-000"
ep_dir.mkdir(parents=True, exist_ok=True)
episodes_df.to_parquet(ep_dir / "file-000.parquet", index=False)
print(f"Created episodes parquet with {len(episodes_df)} rows")

# 3. Create stats.json from episodes_stats.jsonl
all_stats = {}
with open(meta_dir / "episodes_stats.jsonl") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        ep_data = json.loads(line)
        ep_stats = ep_data.get("stats", {})
        for key, val in ep_stats.items():
            if key not in all_stats:
                all_stats[key] = {}
            for stat_name, stat_val in val.items():
                if stat_name == "count":
                    continue
                if stat_name not in all_stats[key]:
                    all_stats[key][stat_name] = []
                arr = np.array(stat_val).flatten().tolist()
                all_stats[key][stat_name].append(arr)

final_stats = {}
for key in all_stats:
    final_stats[key] = {}
    for stat_name in all_stats[key]:
        arrs = all_stats[key][stat_name]
        stacked = np.stack([np.array(a) for a in arrs])
        if stat_name == "min":
            final_stats[key][stat_name] = stacked.min(axis=0).tolist()
        elif stat_name == "max":
            final_stats[key][stat_name] = stacked.max(axis=0).tolist()
        elif stat_name in ["mean", "std"]:
            final_stats[key][stat_name] = stacked.mean(axis=0).tolist()

with open(meta_dir / "stats.json", "w") as f:
    json.dump(final_stats, f, indent=2)
print("Created stats.json")

# 4. Update info.json to v3.0
info["codebase_version"] = "v3.0"
info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/episode_{episode_index:06d}.mp4"
with open(meta_dir / "info.json", "w") as f:
    json.dump(info, f, indent=2)
print("Updated info.json to v3.0")
print("Conversion done!")
