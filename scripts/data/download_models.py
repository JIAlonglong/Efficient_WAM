"""
Download Motus models via hf-mirror.com using curl (bypasses proxy reliably).

Usage:
    python scripts/data/download_models.py --all
    python scripts/data/download_models.py --repo qwen
    python scripts/data/download_models.py --repo motus
    python scripts/data/download_models.py --repo robotwin
"""

import argparse
import os
import subprocess
import sys


MIRROR = "https://hf-mirror.com"
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "pretrained_models")

REPOS = {
    "qwen": {
        "repo": "Qwen/Qwen3-VL-2B-Instruct",
        "dir": os.path.join(BASE_DIR, "Qwen3-VL-2B-Instruct"),
        "files": [
            (".gitattributes", False),
            ("README.md", False),
            ("chat_template.json", False),
            ("config.json", False),
            ("generation_config.json", False),
            ("merges.txt", False),
            ("preprocessor_config.json", False),
            ("tokenizer.json", False),
            ("tokenizer_config.json", False),
            ("video_preprocessor_config.json", False),
            ("vocab.json", False),
            ("model.safetensors", True),   # 4.3 GB
        ],
    },
    "motus": {
        "repo": "motus-robotics/Motus",
        "dir": os.path.join(BASE_DIR, "Motus"),
        "files": [
            (".gitattributes", False),
            ("LICENSE", False),
            ("README.md", False),
            ("config.json", False),
            ("mp_rank_00_model_states.pt", True),  # 16 GB
        ],
    },
    "robotwin": {
        "repo": "motus-robotics/Motus_robotwin2",
        "dir": os.path.join(BASE_DIR, "Motus_robotwin2"),
        "files": [
            (".gitattributes", False),
            ("LICENSE", False),
            ("README.md", False),
            ("config.json", False),
            ("mp_rank_00_model_states.pt", True),  # 16 GB
        ],
    },
}


def download_file(url: str, output: str) -> bool:
    """Download a single file using curl, bypassing proxy."""
    if os.path.exists(output):
        size = os.path.getsize(output)
        print(f"  [skip] {os.path.basename(output)} ({size // 1048576} MB, already exists)")
        return True

    os.makedirs(os.path.dirname(output), exist_ok=True)
    print(f"  [download] {os.path.basename(output)} ...", end="", flush=True)

    # Use curl with no proxy, resume support
    cmd = [
        "curl", "-L", "--noproxy", "*",
        "--connect-timeout", "30",
        "--max-time", "36000",  # 10 hours max per file
        "-#",  # progress bar
        "-C", "-",  # resume
        "-o", output,
        url,
    ]

    env = os.environ.copy()
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        env.pop(k, None)

    result = subprocess.run(cmd, env=env, capture_output=False)
    if result.returncode == 0:
        size = os.path.getsize(output)
        print(f" done ({size // 1048576} MB)")
        return True
    else:
        print(f" FAILED (exit code {result.returncode})")
        # Clean up partial file
        if os.path.exists(output):
            os.remove(output)
        return False


def download_repo(name: str) -> bool:
    """Download all files for a repo."""
    info = REPOS[name]
    repo = info["repo"]
    local_dir = info["dir"]

    print(f"\n{'='*60}")
    print(f"  {repo}")
    print(f"  → {local_dir}")
    print(f"{'='*60}")

    all_ok = True
    for filename, is_large in info["files"]:
        url = f"{MIRROR}/{repo}/resolve/main/{filename}"
        output = os.path.join(local_dir, filename)
        if not download_file(url, output):
            all_ok = False
            if is_large:
                print(f"  [ERROR] Failed to download large file: {filename}")
                return False

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Download Motus models via hf-mirror.com")
    parser.add_argument("--repo", choices=list(REPOS.keys()) + ["all"], default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not args.repo and not args.all:
        print("Usage: python download_models.py --repo [qwen|motus|robotwin|all]")
        print("       python download_models.py --all")
        sys.exit(1)

    targets = list(REPOS.keys()) if (args.all or args.repo == "all") else [args.repo]

    for name in targets:
        if not download_repo(name):
            print(f"\n[ABORTED] Failed to download {name}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("  All downloads complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
