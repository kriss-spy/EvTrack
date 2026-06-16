#!/usr/bin/env python3
"""
Export SDSTrack completion status from RunPod (or any workspace).

Usage on RunPod:
    python export_completion_status.py --workspace /workspace/sdstrack --output status.json

Then transfer the small status.json to AutoDL (or your laptop):
    scp runpod:/workspace/sdstrack/status.json .
"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, required=True)
    parser.add_argument("--output", type=str, default="completion_status.json")
    args = parser.parse_args()

    ws = Path(args.workspace).resolve()
    progress_path = ws / "progress.json"
    results_dir = ws / "RGBE_workspace" / "results" / "VisEvent" / "cvpr2024_rgbe"

    # Load progress
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        completed_seqs = list(set(progress.get("completed_seqs", [])))
        completed_tars = list(set(progress.get("completed_tars", [])))
    else:
        print(f"[Warning] progress.json not found at {progress_path}")
        completed_seqs = []
        completed_tars = []

    # Load result files
    if results_dir.exists():
        result_seqs = [f.stem for f in results_dir.glob("*.txt")]
    else:
        print(f"[Warning] results directory not found at {results_dir}")
        result_seqs = []

    status = {
        "source": str(ws),
        "completed_seqs": sorted(completed_seqs),
        "completed_tars": sorted(completed_tars),
        "result_seqs": sorted(result_seqs),
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(status, indent=2))
    print(f"[Export] Saved {len(completed_seqs)} completed sequences, {len(result_seqs)} result files")
    print(f"[Export] File: {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
