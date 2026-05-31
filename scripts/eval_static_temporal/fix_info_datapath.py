#!/usr/bin/env python3
"""Fix info.json data_path to match actual file naming."""
import json
from pathlib import Path

meta_dir = Path("/root/intern/jialongliu/projects/starVLA/data/shared_datasets/robotwin_fastwam/robotwin2.0/meta")
with open(meta_dir / "info.json") as f:
    info = json.load(f)

# Match actual naming: data/chunk-XXX/episode_XXXXXX.parquet
info["data_path"] = "data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet"
# Match actual naming: videos/{video_key}/chunk-XXX/episode_XXXXXX.mp4
info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/episode_{file_index:06d}.mp4"

with open(meta_dir / "info.json", "w") as f:
    json.dump(info, f, indent=2)
print("Fixed data_path and video_path in info.json")
