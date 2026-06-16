#!/usr/bin/env python3
"""
SDSTrack VisEvent Evaluation — Cloud GPU Version
================================================
Standalone script for running SDSTrack evaluation on VisEvent test set.
Works on any GPU cloud (RunPod, Vast.ai, Lambda, etc.) — no Colab required.

Key features:
- Configurable workspace directory (env var or arg)
- Downloads models and dataset from Hugging Face
- Runs evaluation directly (no subprocess) — avoids deadlock
- Keeps model loaded in memory between sequences
- Streaming: one tar shard at a time, low disk usage
- Resumable: saves progress to local JSON

Usage:
    python sdstrack_eval.py --workspace /workspace/sdstrack

Environment Variables:
    SDSTRACK_WORKSPACE  — workspace directory (default: ./sdstrack)
    HF_TOKEN            — Hugging Face token (required for private/gated repos)
                        Get one at: https://huggingface.co/settings/tokens
"""

import os
import sys
import json
import tarfile
import shutil
import argparse
import time
import signal
from pathlib import Path
from typing import Set, List, Optional

import numpy as np
import subprocess

# =============================================================================
# Configuration
# =============================================================================

# Repositories
HF_DATASET_REPO = "krisspy39/visevent"
HF_CHECKPOINT_REPO = "krisspy39/sdstrack-rgbe"
HF_PRETRAIN_REPO = "krisspy39/vipt-ostrack"

# Model filenames
CHECKPOINT_FILE = "SDSTrack_cvpr2024_rgbe.pth.tar"
PRETRAIN_FILE = "OSTrack_ep0300.pth.tar"

# Direct download URLs
PRETRAIN_URL = f"https://huggingface.co/{HF_PRETRAIN_REPO}/resolve/main/{PRETRAIN_FILE}"
CHECKPOINT_URL = f"https://huggingface.co/{HF_CHECKPOINT_REPO}/resolve/main/{CHECKPOINT_FILE}"

# Dataset subset
DATASET_SUBSET = "test"
WEBDATASET_PATH = "webdataset/test"

# =============================================================================
# Paths
# =============================================================================

def get_workspace() -> Path:
    """Get workspace directory from env or arg."""
    env = os.environ.get("SDSTRACK_WORKSPACE", "")
    if env:
        return Path(env).resolve()
    # Default to current working directory / sdstrack
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
        self.testlist = self.test_subset / "testlist.txt"

    def ensure_dirs(self):
        for p in [self.data, self.visevent, self.test_subset, self.pretrained,
                  self.models, self.checkpoint_dir, self.results, self.cache]:
            p.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Setup and Patching
# =============================================================================

def clone_sdstrack(paths: Paths):
    """Clone SDSTrack repo if not present."""
    if (paths.ws / "lib").exists():
        print("[Setup] SDSTrack already cloned")
        return
    print("[Setup] Cloning SDSTrack...")
    import subprocess
    # Clone into a temp dir first, then move contents to workspace
    # (workspace dir may already have data/ results/ subdirs created by ensure_dirs)
    temp_dir = paths.ws / ".upstream_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    subprocess.run(["git", "clone", "https://github.com/hoqolo/SDSTrack.git", str(temp_dir)], check=True)
    for item in temp_dir.iterdir():
        dest = paths.ws / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest, ignore_errors=True)
            else:
                dest.unlink()
        shutil.move(str(item), str(paths.ws))
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("[Setup] Clone complete")


def apply_patches(paths: Paths):
    """Apply PyTorch 2.x compatibility patches."""
    flag = paths.ws / ".patches_applied"
    if flag.exists():
        print("[Setup] Patches already applied")
        return

    loader = paths.ws / "lib" / "train" / "data" / "loader.py"
    if loader.exists():
        print("[Setup] Applying compatibility patches...")
        content = loader.read_text()
        if "import collections.abc" not in content:
            content = content.replace("import collections", "import collections\nimport collections.abc")
        content = content.replace(
            "from torch._six import string_classes",
            "try:\n    from torch._six import string_classes\nexcept ImportError:\n    string_classes = (str, bytes)"
        )
        content = content.replace("collections.Mapping", "collections.abc.Mapping")
        content = content.replace("collections.Sequence", "collections.abc.Sequence")
        loader.write_text(content)

    # Patch weights_only in all Python files
    patched = 0
    for fpath in (paths.ws / "lib").rglob("*.py"):
        content = fpath.read_text()
        original = content
        content = content.replace("map_location='cpu')", "map_location='cpu', weights_only=False)")
        content = content.replace('map_location="cpu")', 'map_location="cpu", weights_only=False)')
        if content != original:
            fpath.write_text(content)
            patched += 1

    # Fix hardcoded paths
    local_py = paths.ws / "lib" / "train" / "admin" / "local.py"
    if local_py.exists():
        content = local_py.read_text()
        content = content.replace(
            "self.visevent_dir = '/home/houxiaojun/Workspace/SDSTrack/data/visevent/train'",
            f"self.visevent_dir = '{paths.visevent / 'train' / 'train_subset'}'"
        )
        local_py.write_text(content)

    local_test = paths.ws / "lib" / "test" / "evaluation" / "local.py"
    if local_test.exists():
        content = local_test.read_text()
        content = content.replace("/home/houxiaojun/Workspace/SDSTrack", str(paths.ws))
        local_test.write_text(content)

    test_script = paths.ws / "RGBE_workspace" / "test_rgbe_mgpus.py"
    if test_script.exists():
        content = test_script.read_text()
        content = content.replace(
            "seq_home = '/public/datasets_neo/VisEvent/VisEvent_dataset/testset/test_subset'",
            f"seq_home = '{paths.test_subset}'"
        )
        test_script.write_text(content)

    # Patch lmdb_utils to avoid broken lmdb import (we don't use LMDB for VisEvent)
    lmdb_utils = paths.ws / "lib" / "utils" / "lmdb_utils.py"
    if lmdb_utils.exists():
        print("[Setup] Patching lmdb_utils.py to avoid lmdb import...")
        lmdb_utils.write_text("""
# Stub: lmdb_utils.py — LMDB is not needed for VisEvent evaluation
# See sdstrack_eval.py apply_patches
import numpy as np
import cv2
import json

LMDB_ENVS = dict()
LMDB_HANDLES = dict()
LMDB_FILELISTS = dict()

def get_lmdb_handle(name):
    raise NotImplementedError("LMDB is not used for VisEvent evaluation")

def decode_img(lmdb_fname, key_name):
    raise NotImplementedError("LMDB is not used for VisEvent evaluation")

def decode_str(lmdb_fname, key_name):
    raise NotImplementedError("LMDB is not used for VisEvent evaluation")

def decode_json(lmdb_fname, key_name):
    raise NotImplementedError("LMDB is not used for VisEvent evaluation")
""")

    # Run create_default_local_file (optional — local files are already patched above)
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, "tracking/create_default_local_file.py",
             "--workspace_dir", str(paths.ws),
             "--data_dir", str(paths.data),
             "--save_dir", str(paths.ws / "output")],
            cwd=paths.ws, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[Setup Warning] create_default_local_file.py failed (non-fatal):\n{e.stderr}")
        print("[Setup Warning] Continuing with manually patched local files...")

    flag.write_text("done")
    print(f"[Setup] Patched {patched} files, fixed paths")


def _wget(url: str, dest: Path):
    """Download file via wget."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[wget] {url} -> {dest}")
    subprocess.run(
        ["wget", "-q", "--show-progress", "--no-clobber", "-O", str(dest), url],
        check=True,
    )


def download_models(paths: Paths):
    """Download pretrained and checkpoint models from Hugging Face via wget."""
    # Pretrained (OSTrack)
    pretrain_path = paths.pretrained / PRETRAIN_FILE
    if not pretrain_path.exists():
        print(f"[Setup] Downloading pretrained model from {HF_PRETRAIN_REPO}...")
        _wget(PRETRAIN_URL, pretrain_path)
        print(f"[Setup] Pretrained model ready: {pretrain_path}")
    else:
        print(f"[Setup] Pretrained model already exists")

    # Checkpoint (SDSTrack)
    ckpt_path = paths.checkpoint_dir / CHECKPOINT_FILE
    symlink = paths.models / CHECKPOINT_FILE
    if not symlink.exists():
        print(f"[Setup] Downloading checkpoint from {HF_CHECKPOINT_REPO}...")
        _wget(CHECKPOINT_URL, ckpt_path)
        # Create symlink
        if symlink.exists() or symlink.is_symlink():
            symlink.unlink()
        real = ckpt_path if ckpt_path.exists() else paths.checkpoint_dir / CHECKPOINT_FILE
        symlink.symlink_to(real)
        print(f"[Setup] Checkpoint ready: {symlink}")
    else:
        print(f"[Setup] Checkpoint already exists")

# =============================================================================
# Dataset Streaming
# =============================================================================

def load_tar_list(paths: Paths) -> List[str]:
    """Load or fetch list of test tar files."""
    cache = paths.cache / "test_tar_list.json"
    if cache.exists():
        return json.loads(cache.read_text())

    import requests
    api_url = f"https://huggingface.co/api/datasets/{HF_DATASET_REPO}/tree/main/{WEBDATASET_PATH}"
    print(f"[Dataset] Fetching tar list from {api_url}...")
    resp = requests.get(api_url)
    resp.raise_for_status()
    data = resp.json()
    tar_files = [f["path"].replace(f"{WEBDATASET_PATH}/", "") for f in data
                 if f["type"] == "file" and f["path"].endswith(".tar")]
    tar_files.sort()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(tar_files))
    print(f"[Dataset] Found {len(tar_files)} tar files")
    return tar_files


def get_seq_name(key: str) -> Optional[str]:
    """Extract sequence name from webdataset key."""
    parts = key.split("__")
    if len(parts) >= 3 and parts[0] == "test_subset":
        return parts[1]
    return None


def seq_is_complete(seq_dir: Path) -> bool:
    """Check if sequence has all required files."""
    gt = seq_dir / "groundtruth.txt"
    absent = seq_dir / "absent_label.txt"
    vis = seq_dir / "vis_imgs"
    evt = seq_dir / "event_imgs"
    if not all(p.exists() for p in [gt, absent, vis, evt]):
        return False
    try:
        gt_lines = len(gt.read_text().strip().split("\n"))
        vis_frames = len([f for f in vis.iterdir() if f.suffix == ".bmp"])
        evt_frames = len([f for f in evt.iterdir() if f.suffix == ".bmp"])
        return vis_frames == gt_lines and evt_frames == gt_lines
    except Exception:
        return False


def result_exists(seq: str, paths: Paths) -> bool:
    return (paths.results / f"{seq}.txt").exists()


def extract_tar(tar_path: Path, paths: Paths) -> Set[str]:
    """Extract sequences from tar file. Returns set of extracted sequence names."""
    new_seqs = set()
    with tarfile.open(tar_path, "r") as tf:
        members = tf.getmembers()
        tar_seqs = {}
        for m in members:
            if not m.isfile():
                continue
            seq = get_seq_name(m.name)
            if seq:
                tar_seqs.setdefault(seq, []).append(m)

        for seq, seq_members in tar_seqs.items():
            if result_exists(seq, paths):
                continue
            seq_dir = paths.test_subset / seq
            seq_dir.mkdir(parents=True, exist_ok=True)
            for m in seq_members:
                parts = m.name.split("__")
                if len(parts) < 3:
                    continue
                rest = "__".join(parts[2:]).replace("__", "/")
                out_path = seq_dir / rest
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fobj = tf.extractfile(m)
                if fobj:
                    out_path.write_bytes(fobj.read())
            if seq_is_complete(seq_dir):
                new_seqs.add(seq)
                print(f"    [Extract] {seq}")
    return new_seqs

# =============================================================================
# Direct Evaluation (No Subprocess)
# =============================================================================

def run_evaluation_direct(paths: Paths, sequences: List[str]) -> List[str]:
    """
    Run evaluation directly in Python without subprocess.
    This avoids the multiprocessing deadlock that occurs with test_rgbe_mgpus.py.
    """
    import torch
    import cv2
    
    # Add SDSTrack to path
    sys.path.insert(0, str(paths.ws))
    
    # Import tracker components
    from lib.test.tracker.sdstrack import SDSTrack
    import lib.test.parameter.sdstrack as rgbe_params
    from lib.train.dataset.depth_utils import get_x_frame
    from lib.utils.box_ops import clip_box

    # Load parameters and create tracker once
    print(f"[Eval] Loading model...")
    params = rgbe_params.parameters("cvpr2024_rgbe", 50)
    tracker = SDSTrack(params)
    
    class SDSTrack_RGBE:
        def __init__(self, tracker):
            self.tracker = tracker
            self.H = None
            self.W = None
        
        def initialize(self, image, region):
            self.H, self.W, _ = image.shape
            gt_bbox = np.array(region).astype(np.float32)
            init_info = {"init_bbox": list(gt_bbox)}
            self.tracker.initialize(image, init_info)
        
        def track(self, image):
            outputs = self.tracker.track(image)
            pred_bbox = outputs["target_bbox"]
            pred_score = outputs["best_score"]
            return pred_bbox, pred_score

    wrapper = SDSTrack_RGBE(tracker)
    
    completed = []
    
    for seq_name in sequences:
        seq_path = paths.test_subset / seq_name
        if not seq_path.exists():
            print(f"  [Skip] {seq_name} not found")
            continue
        
        result_file = paths.results / f"{seq_name}.txt"
        if result_file.exists():
            print(f"  [Skip] {seq_name} already done")
            completed.append(seq_name)
            continue

        print(f"\n  [Eval] {seq_name}")
        
        # Load sequence data
        vis_dir = seq_path / "vis_imgs"
        evt_dir = seq_path / "event_imgs"
        gt_file = seq_path / "groundtruth.txt"
        absent_file = seq_path / "absent_label.txt"
        
        vis_files = sorted([f for f in vis_dir.iterdir() if f.suffix == ".bmp"])
        evt_files = sorted([f for f in evt_dir.iterdir() if f.suffix == ".bmp"])
        gt = np.loadtxt(gt_file, delimiter=",")
        absent = np.loadtxt(absent_file)
        
        if gt.ndim == 1:
            gt = gt.reshape(1, -1)
        
        # Handle first frame absent
        if absent[0] == 0:
            first_present = absent.argmax()
            vis_files = vis_files[first_present:]
            evt_files = evt_files[first_present:]
            gt = gt[first_present:]
        
        if len(vis_files) != len(gt):
            print(f"  [Warning] {seq_name}: frame count mismatch ({len(vis_files)} vs {len(gt)})")
        
        # Run tracking with prefetch to overlap CPU I/O with GPU compute
        result = np.zeros_like(gt)
        result[0] = gt[0]
        
        start_time = time.time()
        
        try:
            # Prefetch first frame
            next_image = get_x_frame(str(vis_files[0]), str(evt_files[0]), dtype="rgbrgb")
            
            for frame_idx in range(len(vis_files)):
                image = next_image
                
                # Prefetch next frame while GPU is busy
                if frame_idx + 1 < len(vis_files):
                    next_image = get_x_frame(str(vis_files[frame_idx + 1]), str(evt_files[frame_idx + 1]), dtype="rgbrgb")
                
                if frame_idx == 0:
                    wrapper.initialize(image, gt[0].tolist())
                else:
                    region, _ = wrapper.track(image)
                    result[frame_idx] = np.array(region)
            
            # Save result
            np.savetxt(result_file, result, fmt="%.14f", delimiter=",")
            elapsed = time.time() - start_time
            fps = len(vis_files) / elapsed if elapsed > 0 else 0
            print(f"  [Done] {seq_name} ({fps:.1f} fps, {elapsed:.1f}s)")
            completed.append(seq_name)
            
        except Exception as e:
            print(f"  [Error] {seq_name}: {e}")
            import traceback
            traceback.print_exc()
            # Don't add to completed — will retry
    
    return completed

# =============================================================================
# Main Evaluation Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SDSTrack VisEvent Evaluation")
    parser.add_argument("--workspace", type=str, default="", help="Workspace directory")
    parser.add_argument("--batch-size", type=int, default=10, help="Sequences per batch")
    parser.add_argument("--skip-setup", action="store_true", help="Skip setup (clone, patch, download)")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only (assume data exists)")
    args = parser.parse_args()

    # Set workspace
    if args.workspace:
        os.environ["SDSTRACK_WORKSPACE"] = args.workspace
    paths = Paths(get_workspace())
    paths.ensure_dirs()

    print("=" * 60)
    print("SDSTrack VisEvent Evaluation — Cloud GPU")
    print("=" * 60)
    print(f"Workspace: {paths.ws}")
    print(f"Results:   {paths.results}")
    print(f"Progress:  {paths.progress}")
    print("=" * 60)

    # Setup
    if not args.eval_only:
        clone_sdstrack(paths)
        apply_patches(paths)
        download_models(paths)

    # Load progress
    if paths.progress.exists():
        progress = json.loads(paths.progress.read_text())
    else:
        progress = {"completed_tars": [], "completed_seqs": [], "total_seqs": 0}

    # Load tar list
    tar_files = load_tar_list(paths)
    total_tars = len(tar_files)
    
    print(f"\n[Progress] {len(progress['completed_tars'])}/{total_tars} tars done")
    print(f"[Progress] {len(set(progress['completed_seqs']))} sequences done")
    print(f"[Progress] {len(progress['completed_seqs'])} total evaluations (includes retries)")

    # Check existing results
    existing_results = set()
    if paths.results.exists():
        existing_results = {f.stem for f in paths.results.glob("*.txt")}
    print(f"[Progress] {len(existing_results)} result files found")

    # Process tars
    pending_seqs: Set[str] = set()
    completed_this_run = 0

    for i, tar_name in enumerate(tar_files):
        if tar_name in progress["completed_tars"]:
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total_tars}] Processing {tar_name}")
        print(f"{'='*60}")

        # Check disk on the workspace (not just root)
        total, used, free = shutil.disk_usage(paths.ws)
        print(f"  [Disk] Free: {free / 1e9:.1f} GB")
        if free < 10e9:  # 10GB threshold — be more aggressive
            print("  [Warning] Low disk space! Cleaning up...")
            for seq_dir in paths.test_subset.iterdir():
                if seq_dir.is_dir() and seq_dir.name in existing_results:
                    shutil.rmtree(seq_dir, ignore_errors=True)
            # Re-check
            total, used, free = shutil.disk_usage(paths.ws)
            if free < 10e9:
                # Force evaluation of pending batch to free more space
                if pending_seqs:
                    batch = sorted(pending_seqs)
                    print(f"\n[Eval] Force-running {len(batch)} sequences to free disk...")
                    done = run_evaluation_direct(paths, batch)
                    for seq in done:
                        seq_dir = paths.test_subset / seq
                        if seq_dir.exists():
                            shutil.rmtree(seq_dir, ignore_errors=True)
                        pending_seqs.discard(seq)
                        existing_results.add(seq)
                    progress["completed_seqs"].extend(done)
                    print(f"[Eval] Done: {len(done)}")
                total, used, free = shutil.disk_usage(paths.ws)
                print(f"  [Disk] After cleanup: {free / 1e9:.1f} GB")
                if free < 5e9:
                    print("  [ERROR] Still critically low on disk. Aborting.")
                    break

        # Download tar
        print(f"  [Download] {tar_name}...")
        tar_url = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/{WEBDATASET_PATH}/{tar_name}"
        tar_path = paths.cache / tar_name
        try:
            if not tar_path.exists():
                _wget(tar_url, tar_path)
            else:
                print(f"  [Cache] {tar_path} already exists")
        except Exception as e:
            print(f"  [Error] Failed to download {tar_name}: {e}")
            continue

        print(f"  [Download] Done: {tar_path}")

        # Extract
        print(f"  [Extract] ...")
        new_seqs = extract_tar(Path(tar_path), paths)
        print(f"  [Extract] {len(new_seqs)} new sequences")

        # Clean cache
        shutil.rmtree(paths.cache, ignore_errors=True)
        paths.cache.mkdir(parents=True, exist_ok=True)

        pending_seqs.update(new_seqs)

        # Evaluate batch
        if len(pending_seqs) >= args.batch_size or (i == total_tars - 1 and pending_seqs):
            batch = sorted(pending_seqs)
            print(f"\n[Eval] Running {len(batch)} sequences...")
            
            done = run_evaluation_direct(paths, batch)
            
            # Clean up completed sequences
            for seq in done:
                seq_dir = paths.test_subset / seq
                if seq_dir.exists():
                    shutil.rmtree(seq_dir, ignore_errors=True)
                pending_seqs.discard(seq)
                existing_results.add(seq)
            
            progress["completed_seqs"].extend(done)
            completed_this_run += len(done)
            print(f"[Eval] Completed {len(done)}/{len(batch)} in this batch")

        # Mark tar as done
        progress["completed_tars"].append(tar_name)
        paths.progress.write_text(json.dumps(progress, indent=2))
        print(f"[Progress] Saved. {len(progress['completed_tars'])}/{total_tars} tars done")

    # Final evaluation for any remaining
    if pending_seqs:
        batch = sorted(pending_seqs)
        print(f"\n[Eval] Final batch: {len(batch)} sequences...")
        done = run_evaluation_direct(paths, batch)
        for seq in done:
            seq_dir = paths.test_subset / seq
            if seq_dir.exists():
                shutil.rmtree(seq_dir, ignore_errors=True)
            pending_seqs.discard(seq)
        progress["completed_seqs"].extend(done)
        paths.progress.write_text(json.dumps(progress, indent=2))

    # Final stats
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    unique_completed = len(set(progress["completed_seqs"]))
    print(f"Total sequences evaluated: {unique_completed}")
    print(f"Results saved to: {paths.results}")
    print(f"Progress saved to: {paths.progress}")
    print("=" * 60)


if __name__ == "__main__":
    main()
