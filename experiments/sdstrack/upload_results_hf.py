#!/usr/bin/env python3
"""
Upload SDSTrack VisEvent result files to Hugging Face Dataset.

Usage (on RunPod or any machine with the result files):
    export HF_TOKEN=hf_...
    python upload_results_hf.py \
        --results-dir /workspace/sdstrack/RGBE_workspace/results/VisEvent/cvpr2024_rgbe \
        --repo-id krisspy39/visevent-sdstrack-results

Requirements:
    pip install huggingface-hub datasets
"""

import os
import sys
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def upload_results(results_dir: str, repo_id: str, private: bool = False):
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    txt_files = sorted(results_path.glob("*.txt"))
    if not txt_files:
        print("ERROR: No .txt result files found.")
        sys.exit(1)

    print(f"Found {len(txt_files)} result files")
    print(f"Target repo: {repo_id}")

    api = HfApi()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN not set. Public uploads may fail for new repos.")

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
        print(f"Repo ready: {repo_id}")
    except Exception as e:
        print(f"Repo creation/check failed: {e}")
        sys.exit(1)

    # Upload files one by one
    uploaded = 0
    for txt_file in txt_files:
        path_in_repo = f"results/{txt_file.name}"
        try:
            api.upload_file(
                path_or_fileobj=str(txt_file),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            uploaded += 1
            if uploaded % 50 == 0:
                print(f"  Uploaded {uploaded}/{len(txt_files)}...")
        except Exception as e:
            print(f"  FAILED {txt_file.name}: {e}")

    print(f"\nDone! Uploaded {uploaded}/{len(txt_files)} files to {repo_id}")

    # Also upload a summary
    summary = {
        "tracker": "SDSTrack",
        "config": "cvpr2024_rgbe",
        "dataset": "VisEvent test",
        "num_sequences": len(txt_files),
        "note": "See experiments/sdstrack/METRICS.json for final metrics",
    }
    summary_path = Path("/tmp/summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    api.upload_file(
        path_or_fileobj=str(summary_path),
        path_in_repo="summary.json",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    print("Uploaded summary.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str,
                        default="/workspace/sdstrack/RGBE_workspace/results/VisEvent/cvpr2024_rgbe")
    parser.add_argument("--repo-id", type=str, default="krisspy39/visevent-sdstrack-results")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    upload_results(args.results_dir, args.repo_id, args.private)


if __name__ == "__main__":
    main()
