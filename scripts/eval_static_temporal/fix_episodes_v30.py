#!/usr/bin/env python3
"""Fix episodes parquet to have v3.0 video path columns and info.json video_path format."""
import json
import numpy as np
import pandas as pd
from pathlib import Path

dataset_root = Path("/root/intern/jialongliu/projects/starVLA/data/shared_datasets/robotwin_fastwam/robotwin2.0")
meta_dir = dataset_root / "meta"

# Load info.json
with open(meta_dir / "info.json") as f:
    info = json.load(f)

# Get video keys from features
video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
print(f"Video keys: {video_keys}")

# Fix video_path to use {file_index} (lerobot 0.4.x format)
info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/episode_{file_index:06d}.mp4"
with open(meta_dir / "info.json", "w") as f:
    json.dump(info, f, indent=2)
print("Updated video_path in info.json")

# Load current episodes parquet
ep_parquet = meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
df = pd.read_parquet(ep_parquet)
print(f"Loaded episodes with {len(df)} rows, columns: {list(df.columns)}")

chunks_size = info.get("chunks_size", 1000)

# Add video path columns for each video key
for vid_key in video_keys:
    chunk_col = f"videos/{vid_key}/chunk_index"
    file_col = f"videos/{vid_key}/file_index"
    df[chunk_col] = (df["episode_index"] // chunks_size).astype(int)
    df[file_col] = df["episode_index"]

print(f"Updated columns: {list(df.columns)}")

# Save back
df.to_parquet(ep_parquet, index=False)
print(f"Saved updated episodes parquet")
print("Done!")
