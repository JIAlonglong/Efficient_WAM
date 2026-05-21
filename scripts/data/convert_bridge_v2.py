"""
Convert Bridge V2 TFRecords to MP4 + NPZ format for probe experiments.

Source (read-only, shared):
    data/bridge/1.0.0/  — original TFRecords, never modified

Destination (personal, not shared):
    --output_dir (default: /root/intern/jialongliu/datasets/bridge_v2/)
        bridge_v2/
            train/
                000000000.mp4
                000000000.npz
                ...
            test/
                ...

Usage:
    python scripts/data/convert_bridge_v2.py
    python scripts/data/convert_bridge_v2.py --output_dir /path/to/personal/dataset --fps 20
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from tqdm import tqdm
from torchvision.io import write_video


# ── Paths ─────────────────────────────────────────────────────────────────
# Shared TFRecords (read-only)
BRIDGE_V2_TFRECORD_DIR = "/root/intern/jialongliu/projects/Motus/data/bridge/1.0.0"

# Default personal output (won't touch shared folders)
DEFAULT_OUTPUT_DIR = "/root/intern/jialongliu/datasets/bridge_v2"


# ── Action / Observation Mappings ─────────────────────────────────────────

def rescale_action_with_bound(
    actions: np.ndarray,
    low: float,
    high: float,
    safety_margin: float = 0,
    post_scaling_max: float = 1.0,
    post_scaling_min: float = -1.0,
) -> np.ndarray:
    resc_actions = (actions - low) / (high - low) * (
        post_scaling_max - post_scaling_min
    ) + post_scaling_min
    return tf.clip_by_value(
        resc_actions,
        post_scaling_min + safety_margin,
        post_scaling_max - safety_margin,
    )


def _rescale_action(
    action: Dict[str, np.ndarray],
    wv_lo: float = -0.05,
    wv_hi: float = 0.05,
    rd_lo: float = -0.25,
    rd_hi: float = 0.25,
) -> Dict[str, np.ndarray]:
    action["world_vector"] = rescale_action_with_bound(
        action["world_vector"], low=wv_lo, high=wv_hi,
        safety_margin=0.01, post_scaling_max=1.75, post_scaling_min=-1.75,
    )
    action["rotation_delta"] = rescale_action_with_bound(
        action["rotation_delta"], low=rd_lo, high=rd_hi,
        safety_margin=0.01, post_scaling_max=1.4, post_scaling_min=-1.4,
    )
    return action


def terminate_bool_to_act(terminate_episode: np.ndarray) -> np.ndarray:
    if terminate_episode == 1.0:
        return np.array([1, 0, 0], dtype=np.int32)
    else:
        return np.array([0, 1, 0], dtype=np.int32)


def map_observation(
    to_step: Dict[str, Any],
    from_step: Dict[str, Any],
    from_image_feature_names: Tuple[str, ...] = ("image",),
    to_image_feature_names: Tuple[str, ...] = ("image",),
) -> None:
    for from_name, to_name in zip(from_image_feature_names, to_image_feature_names):
        to_step["observation"][to_name] = from_step["observation"][from_name]


def bridge_v2_map_action(
    to_step: Dict[str, Any],
    from_step: Dict[str, Any],
) -> None:
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(
        from_step["is_terminal"]
    )
    to_step["action"]["world_vector"] = from_step["action"][0:3]
    to_step["action"]["rotation_delta"] = from_step["action"][3:6]
    open_gripper = from_step["action"][6:7]
    open_gripper = tf.round(open_gripper)
    open_gripper = -(open_gripper * 2 - 1)
    to_step["action"]["gripper_closedness_action"] = open_gripper
    to_step["action"] = _rescale_action(to_step["action"])


bridge_v2_map_observation = functools.partial(
    map_observation,
    from_image_feature_names=("image_0",),
    to_image_feature_names=("image",),
)


def step_map_fn(step, map_observation, map_action):
    transformed_step = {
        "observation": {},
        "action": {
            "gripper_closedness_action": np.zeros(1, dtype=np.float32),
            "rotation_delta": np.zeros(3, dtype=np.float32),
            "terminate_episode": np.zeros(3, dtype=np.int32),
            "world_vector": np.zeros(3, dtype=np.float32),
            "base_displacement_vertical_rotation": np.zeros(1, dtype=np.float32),
            "base_displacement_vector": np.zeros(2, dtype=np.float32),
        },
    }
    map_observation(transformed_step, step)
    map_action(transformed_step, step)
    action = np.concatenate([
        transformed_step["action"]["world_vector"],
        transformed_step["action"]["rotation_delta"],
        transformed_step["action"]["gripper_closedness_action"],
        transformed_step["action"]["base_displacement_vector"],
        transformed_step["action"]["base_displacement_vertical_rotation"],
    ], axis=0)
    transformed_step["action"] = action
    return transformed_step


def bridge_v2_extract_metadata(episode: Dict[str, Any]) -> Dict[str, Any]:
    steps = list(episode["steps"])
    reward_last = float(steps[-1].get("reward", 1.0))
    lang = steps[0].get("language_instruction", b"")
    if isinstance(lang, bytes):
        lang = lang.decode("utf-8")
    return {"success": reward_last > 0.5, "language_instruction": lang}


def episode_map_fn(episode, map_step, extract_metadata=None):
    steps = list(map(map_step, episode["steps"]))
    frames = np.stack([s["observation"]["image"] for s in steps], axis=0)
    result = {"video": frames, "action": np.stack([s["action"] for s in steps])}
    if extract_metadata is not None:
        result.update(extract_metadata(episode))
    return result


# ── Save ──────────────────────────────────────────────────────────────────

def save_episode(episode: Dict[str, Any], base_path: Path, fps: int) -> None:
    video = torch.from_numpy(episode["video"])
    write_video(str(base_path) + ".mp4", video, fps=fps)
    save_dict = {"actions": episode["action"]}
    if "success" in episode:
        save_dict["success"] = np.array(episode["success"], dtype=np.bool_)
    if "language_instruction" in episode:
        save_dict["language_instruction"] = np.array(episode["language_instruction"])
    np.savez(str(base_path) + ".npz", **save_dict)


# ── Main ──────────────────────────────────────────────────────────────────

def convert(
    tfrecord_dir: str = BRIDGE_V2_TFRECORD_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    fps: int = 20,
    test_ratio: float = 0.1,
    max_episodes: int | None = None,
) -> None:
    output_dir = Path(output_dir)
    for split in ("train", "test"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    print(f"Source (read-only): {tfrecord_dir}")
    print(f"Output (personal):  {output_dir}")

    builder = tfds.builder_from_directory(builder_dir=tfrecord_dir)
    step_fn = functools.partial(
        step_map_fn,
        map_observation=bridge_v2_map_observation,
        map_action=functools.partial(bridge_v2_map_action),
    )

    for split_name in ["train", "val"]:
        output_split = "test" if split_name == "val" else "train"
        split_out = output_dir / output_split

        print(f"\nConverting split: {split_name} → {output_split}")
        dataset = builder.as_dataset(split=split_name, shuffle_files=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = tfds.as_numpy(dataset)

        idx = 0
        for i, episode in tqdm(enumerate(dataset), desc=f"bridge_v2-{split_name}"):
            if max_episodes is not None and idx >= max_episodes:
                break
            try:
                ep = episode_map_fn(episode, step_fn, bridge_v2_extract_metadata)
                base_path = split_out / f"{idx:09d}"
                save_episode(ep, base_path, fps)
                idx += 1
            except Exception as e:
                print(f"  Skip episode {i}: {e}")
                continue

        print(f"  Saved {idx} episodes to {split_out}")

    print(f"\nDone. Output at: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Bridge V2 TFRecords → MP4+NPZ")
    parser.add_argument("--tfrecord_dir", default=BRIDGE_V2_TFRECORD_DIR,
                        help="Path to TFRecord directory (shared, read-only)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output path for MP4+NPZ (personal, not shared)")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Limit episodes per split (for testing)")
    args = parser.parse_args()
    convert(**vars(args))
