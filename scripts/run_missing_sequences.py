#!/usr/bin/env python3
"""
Evaluate missing VisEvent sequences on a machine with local dataset (e.g., AutoDL).

Usage:
    python run_missing_sequences.py \
        --dataset-path /root/autodl-tmp/data/VisEvent/test \
        --workspace /workspace/sdstrack \
        --sequences 00442_UAV_outdoor6,dvSave-2021_02_04_21_18_52,...

Or read from a file:
    python run_missing_sequences.py \
        --dataset-path /root/autodl-tmp/data/VisEvent/test \
        --workspace /workspace/sdstrack \
        --seq-file missing_sequences.txt

Requirements:
    - PyTorch with CUDA (GPU recommended)
    - huggingface_hub
    - opencv-python
    - numpy
"""

import os
import sys
import json
import shutil
import argparse
import time
from pathlib import Path
from typing import List

import numpy as np

# Import from sdstrack_eval.py (must be in same directory or on PYTHONPATH)
# We duplicate the necessary parts to keep it standalone.

def get_workspace() -> Path:
    env = os.environ.get("SDSTRACK_WORKSPACE", "")
    if env:
        return Path(env).resolve()
    return Path.cwd() / "sdstrack"


class Paths:
    def __init__(self, workspace: Path):
        self.ws = workspace
        self.data = self.ws / "data"
        self.visevent = self.data / "visevent"
        self.test_subset = self.visevent / "test" / "test_subset"
        self.pretrained = self.ws / "pretrained" / "vitb_256_mae_ce_32x4_ep300"
        self.models = self.ws / "models"
        self.checkpoint_dir = self.ws / "output" / "checkpoints" / "train" / "sdstrack" / "cvpr2024_rgbe"
        self.results = self.ws / "RGBE_workspace" / "results" / "VisEvent" / "cvpr2024_rgbe"
        self.progress = self.ws / "progress.json"
        self.cache = self.ws / ".cache" / "hf"

    def ensure_dirs(self):
        for p in [self.data, self.visevent, self.test_subset, self.pretrained,
                  self.models, self.checkpoint_dir, self.results, self.cache]:
            p.mkdir(parents=True, exist_ok=True)


# Add common paths to find sdstrack_eval.py
_script_dir = Path(__file__).resolve().parent
_search_paths = [
    _script_dir,
    _script_dir.parent / "code" / "SDSTrack",
    _script_dir / ".." / "code" / "SDSTrack",
    Path.cwd() / "code" / "SDSTrack",
    Path.cwd() / "SDSTrack",
    Path.cwd(),
]
for _search in _search_paths:
    if (_search / "sdstrack_eval.py").exists():
        sys.path.insert(0, str(_search))
        break
else:
    print("[Error] sdstrack_eval.py not found. Searched:")
    for p in _search_paths:
        print(f"  - {p}")
    print("\nWorkaround: set PYTHONPATH explicitly:")
    print("  PYTHONPATH=/path/to/code/SDSTrack python scripts/run_missing_sequences.py ...")
    sys.exit(1)

# Reuse setup functions from sdstrack_eval if available
try:
    from sdstrack_eval import clone_sdstrack, apply_patches, download_models, run_evaluation_direct, CHECKPOINT_FILE, PRETRAIN_FILE
except ImportError as e:
    print(f"[Error] Failed to import from sdstrack_eval: {e}")
    sys.exit(1)


def symlink_dataset(paths: Paths, dataset_path: Path):
    """Create symlinks from the local dataset to the workspace structure."""
    if not dataset_path.exists():
        print(f"[Error] Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    paths.test_subset.mkdir(parents=True, exist_ok=True)

    # List all sequence directories in the local dataset
    for seq_dir in dataset_path.iterdir():
        if not seq_dir.is_dir():
            continue
        # Check if it's a valid sequence
        if not (seq_dir / "groundtruth.txt").exists():
            continue

        dest = paths.test_subset / seq_dir.name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(seq_dir.resolve())

    print(f"[Dataset] Symlinked {len(list(paths.test_subset.iterdir()))} sequences to {paths.test_subset}")


def run_sequences(sequences: List[str], paths: Paths):
    """Run evaluation on a specific list of sequences."""
    # Filter to only sequences that exist in the dataset
    available = [s for s in sequences if (paths.test_subset / s).exists()]
    missing = [s for s in sequences if not (paths.test_subset / s).exists()]

    if missing:
        print(f"[Warning] {len(missing)} sequences not found in dataset:")
        for s in missing:
            print(f"  - {s}")

    if not available:
        print("[Error] No sequences available to evaluate")
        return []

    print(f"[Eval] Running {len(available)} sequences...")
    completed = run_evaluation_direct(paths, available)
    return completed


def main():
    parser = argparse.ArgumentParser(description="Evaluate missing VisEvent sequences")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to VisEvent test_subset directory")
    parser.add_argument("--workspace", type=str, default="",
                        help="SDSTrack workspace directory (default: ./sdstrack or env SDSTRACK_WORKSPACE)")
    parser.add_argument("--sequences", type=str, default="",
                        help="Comma-separated list of sequence names")
    parser.add_argument("--seq-file", type=str, default="",
                        help="File containing sequence names (one per line)")
    parser.add_argument("--skip-setup", action="store_true",
                        help="Skip setup (clone, patch, download)")
    args = parser.parse_args()

    # Set workspace
    if args.workspace:
        os.environ["SDSTRACK_WORKSPACE"] = args.workspace
    paths = Paths(get_workspace())
    paths.ensure_dirs()

    print("=" * 60)
    print("SDSTrack Missing Sequence Evaluation")
    print("=" * 60)
    print(f"Workspace: {paths.ws}")
    print(f"Dataset:   {args.dataset_path}")
    print("=" * 60)

    # Setup
    if not args.skip_setup:
        clone_sdstrack(paths)
        apply_patches(paths)
        download_models(paths)

    # Symlink dataset
    dataset_path = Path(args.dataset_path).resolve()
    symlink_dataset(paths, dataset_path)

    # Load sequences
    sequences = []
    if args.sequences:
        sequences.extend([s.strip() for s in args.sequences.split(",") if s.strip()])
    if args.seq_file:
        seq_file = Path(args.seq_file)
        if seq_file.exists():
            sequences.extend([s.strip() for s in seq_file.read_text().split("\n") if s.strip()])

    if not sequences:
        print("[Error] No sequences specified. Use --sequences or --seq-file")
        sys.exit(1)

    print(f"\n[Input] {len(sequences)} sequences to evaluate")

    # Run evaluation
    completed = run_sequences(sequences, paths)

    # Final stats
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Requested: {len(sequences)}")
    print(f"Completed: {len(completed)}")
    print(f"Results saved to: {paths.results}")
    print("=" * 60)


if __name__ == "__main__":
    main()
