#!/usr/bin/env python3
"""
Verify VisEvent Test Set Completeness for SDSTrack Evaluation
=============================================================

Designed to run on a machine with the **full VisEvent dataset** locally
(e.g., AutoDL) and the **SDSTrack workspace** (e.g., copied from RunPod).

Usage:
    # On a machine with the full dataset (e.g., AutoDL)
    python verify_visevent_completeness.py \
        --dataset-path /path/to/VisEvent/test/test_subset \
        --workspace /path/to/sdstrack_workspace

    # If the workspace is on a different machine, copy progress.json + results/ locally:
    scp runpod:/workspace/sdstrack/progress.json ./workspace/
    scp -r runpod:/workspace/sdstrack/RGBE_workspace/results ./workspace/
    python verify_visevent_completeness.py \
        --dataset-path /path/to/VisEvent/test/test_subset \
        --workspace ./workspace

Outputs:
    - completeness_report.json
    - analysis summary to stdout

Requirements:
    - access to the workspace directory (progress.json + results/)
    - access to the full VisEvent test subset directory
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Set, List, Dict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_official_sequences(dataset_path: Path) -> Set[str]:
    """
    Build the official set of all test sequences by scanning the local
    VisEvent test_subset directory.
    """
    if not dataset_path.exists():
        print(f"[Error] Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    print(f"[Scan] Reading official sequences from {dataset_path}...")
    official: Set[str] = set()
    for seq_dir in dataset_path.iterdir():
        if not seq_dir.is_dir():
            continue
        # Check that it has the expected structure
        if (seq_dir / "groundtruth.txt").exists():
            official.add(seq_dir.name)

    print(f"[Scan] Found {len(official)} sequences in {dataset_path}")
    return official


def load_progress(progress_path: Path) -> Dict:
    """Load progress.json or return empty structure."""
    if progress_path.exists():
        return json.loads(progress_path.read_text())
    return {"completed_tars": [], "completed_seqs": [], "total_seqs": 0}


def load_completed_results(results_dir: Path) -> Set[str]:
    """Load set of sequences that have result files."""
    if not results_dir.exists():
        return set()
    return {f.stem for f in results_dir.glob("*.txt")}


def analyze_sequence(
    seq: str,
    dataset_path: Path,
) -> Dict:
    """
    Inspect a single sequence on disk to determine why
    it might have been skipped.
    """
    seq_dir = dataset_path / seq
    info = {
        "seq": seq,
        "exists_on_disk": seq_dir.exists(),
        "has_groundtruth": False,
        "has_absent_label": False,
        "has_vis_imgs": False,
        "has_event_imgs": False,
        "first_frame_absent": False,
        "frame_count": 0,
        "gt_count": 0,
        "vis_count": 0,
        "evt_count": 0,
        "recommendation": "unknown",
    }

    if not seq_dir.exists():
        info["recommendation"] = "not_in_dataset"
        return info

    gt_file = seq_dir / "groundtruth.txt"
    absent_file = seq_dir / "absent_label.txt"
    vis_dir = seq_dir / "vis_imgs"
    evt_dir = seq_dir / "event_imgs"

    info["has_groundtruth"] = gt_file.exists()
    info["has_absent_label"] = absent_file.exists()
    info["has_vis_imgs"] = vis_dir.exists()
    info["has_event_imgs"] = evt_dir.exists()

    if gt_file.exists():
        try:
            lines = gt_file.read_text().strip().split("\n")
            info["gt_count"] = len(lines)
        except Exception:
            pass

    if absent_file.exists():
        try:
            absent = [int(x) for x in absent_file.read_text().strip().split("\n")]
            if absent and absent[0] == 0:
                info["first_frame_absent"] = True
        except Exception:
            pass

    if vis_dir.exists():
        info["vis_count"] = len([f for f in vis_dir.iterdir() if f.suffix == ".bmp"])
    if evt_dir.exists():
        info["evt_count"] = len([f for f in evt_dir.iterdir() if f.suffix == ".bmp"])

    info["frame_count"] = info["gt_count"]

    # Decide recommendation
    if info["first_frame_absent"]:
        info["recommendation"] = "first_frame_absent"
    elif not all([info["has_groundtruth"], info["has_absent_label"], info["has_vis_imgs"], info["has_event_imgs"]]):
        info["recommendation"] = "missing_data"
    elif info["vis_count"] != info["gt_count"] or info["evt_count"] != info["gt_count"]:
        info["recommendation"] = "frame_mismatch"
    else:
        info["recommendation"] = "should_evaluate"

    return info


def main():
    parser = argparse.ArgumentParser(description="Verify VisEvent test completeness")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to VisEvent test_subset directory (e.g., /path/to/VisEvent/test/test_subset)")
    parser.add_argument("--workspace", type=str, required=True,
                        help="Path to SDSTrack workspace containing progress.json + results/")
    parser.add_argument("--report", type=str, default="completeness_report.json", help="Output JSON report")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    ws = Path(args.workspace).resolve()
    paths = {
        "progress": ws / "progress.json",
        "results": ws / "RGBE_workspace" / "results" / "VisEvent" / "cvpr2024_rgbe",
    }

    print("=" * 60)
    print("VisEvent Test Set Completeness Verification")
    print("=" * 60)
    print(f"Dataset:   {dataset_path}")
    print(f"Workspace: {ws}")
    print(f"Progress:  {paths['progress']}")
    print(f"Results:   {paths['results']}")
    print("=" * 60)

    # 1. Load official sequence list from local dataset
    official_seqs = load_official_sequences(dataset_path)
    print(f"\n[Official] Total VisEvent test sequences: {len(official_seqs)}")

    # 2. Load progress
    progress = load_progress(paths["progress"])
    completed_seqs = set(progress.get("completed_seqs", []))
    print(f"[Progress] Unique completed sequences: {len(completed_seqs)}")

    # 3. Load result files
    result_seqs = load_completed_results(paths["results"])
    print(f"[Results]  Result files on disk: {len(result_seqs)}")

    # 4. Cross-check
    # A sequence is truly "done" if it has a result file AND is in progress
    truly_done = completed_seqs & result_seqs
    missing_from_progress = official_seqs - completed_seqs
    missing_from_results = official_seqs - result_seqs
    missing = missing_from_progress | missing_from_results
    print(f"\n[Cross-check]")
    print(f"  Truly done (progress + result file): {len(truly_done)}")
    print(f"  Missing from progress: {len(missing_from_progress)}")
    print(f"  Missing from results:  {len(missing_from_results)}")
    print(f"  Total missing: {len(missing)}")

    # 5. Analyze each missing sequence
    print(f"\n[Analysis] Analyzing {len(missing)} missing sequences...")
    analysis = []
    for seq in sorted(missing):
        info = analyze_sequence(seq, dataset_path)
        analysis.append(info)
        print(f"  {seq:40s} -> {info['recommendation']}")

    # 6. Categorize
    categories: Dict[str, List[str]] = {}
    for info in analysis:
        categories.setdefault(info["recommendation"], []).append(info["seq"])

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for cat, seqs in sorted(categories.items()):
        print(f"  {cat:25s}: {len(seqs):3d} sequences")
        if cat in ("first_frame_absent", "missing_data", "frame_mismatch") and len(seqs) <= 20:
            for s in seqs:
                print(f"    - {s}")

    # 7. Write report
    report = {
        "dataset_path": str(dataset_path),
        "workspace": str(ws),
        "official_total": len(official_seqs),
        "unique_completed_seqs": len(completed_seqs),
        "result_files": len(result_seqs),
        "truly_done": len(truly_done),
        "missing": len(missing),
        "missing_sequences": sorted(missing),
        "categories": {k: sorted(v) for k, v in categories.items()},
        "analysis": analysis,
    }
    report_path = Path(args.report)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n[Report] Saved to {report_path}")

    # 8. Recommend next steps
    print("\n" + "=" * 60)
    print("Recommended Next Steps")
    print("=" * 60)
    reevaluate = categories.get("should_evaluate", [])
    if reevaluate:
        print(f"1. Re-evaluate {len(reevaluate)} sequences that appear complete but lack results:")
        for s in reevaluate:
            print(f"   {s}")
    else:
        print("1. No sequences are ready for re-evaluation.")

    if categories.get("first_frame_absent"):
        print(f"2. {len(categories['first_frame_absent'])} sequences have first-frame absent.")
        print("   Confirm if these should be excluded per the paper/dataset rules.")

    if categories.get("missing_data"):
        print(f"3. {len(categories['missing_data'])} sequences have missing data on disk.")
        print("   Check if the dataset was not fully extracted or the sequence is corrupted.")

    if categories.get("not_in_dataset"):
        print(f"4. {len(categories['not_in_dataset'])} sequences are not present in the local dataset.")
        print("   Verify the dataset path is correct.")


if __name__ == "__main__":
    main()
