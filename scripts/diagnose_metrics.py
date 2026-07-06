#!/usr/bin/env python3
"""
Diagnose metrics breakdown by comparing original vs new sequences.

Usage:
    python diagnose_metrics.py \
        --results /root/autodl-tmp/sdstrack_results \
        --dataset /root/autodl-tmp/data/VisEvent/test \
        --progress /path/to/progress.json \
        --status /path/to/status.json
"""

import os
import json
import glob
import argparse
import numpy as np


def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0


def compute_seq_metrics(pred, gt):
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    ious, dists = [], []
    for p, g in zip(pred, gt):
        ious.append(compute_iou(p, g))
        pcx, pcy = p[0] + p[2]/2, p[1] + p[3]/2
        gcx, gcy = g[0] + g[2]/2, g[1] + g[3]/2
        dists.append(np.sqrt((pcx - gcx)**2 + (pcy - gcy)**2))
    return ious, dists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--progress", type=str, default="")
    parser.add_argument("--status", type=str, default="")
    parser.add_argument("--original-only", type=str, default="",
                        help="File with original 300 sequence names (one per line)")
    args = parser.parse_args()

    # Load sequence lists
    original_seqs = set()
    if args.original_only:
        with open(args.original_only) as f:
            original_seqs = {s.strip() for s in f if s.strip()}
    elif args.progress:
        with open(args.progress) as f:
            p = json.load(f)
            original_seqs = set(p.get("completed_seqs", []))
    elif args.status:
        with open(args.status) as f:
            s = json.load(f)
            original_seqs = set(s.get("completed_seqs", []))

    if not original_seqs:
        print("Warning: No original sequence list provided. Cannot split metrics.")

    # Group result files
    files = sorted(glob.glob(os.path.join(args.results, "*.txt")))
    original_files = []
    new_files = []
    for f in files:
        seq = os.path.basename(f).replace(".txt", "")
        if seq in original_seqs:
            original_files.append(f)
        else:
            new_files.append(f)

    print(f"Total result files: {len(files)}")
    print(f"Original sequences: {len(original_files)}")
    print(f"New sequences: {len(new_files)}")

    # Compute metrics for each group
    def compute_group(files, label):
        all_ious, all_dists = [], []
        seq_metrics = []
        for res_file in files:
            seq = os.path.basename(res_file).replace(".txt", "")
            gt_file = os.path.join(args.dataset, seq, "groundtruth.txt")
            if not os.path.exists(gt_file):
                continue
            try:
                pred = np.loadtxt(res_file, delimiter=",")
                gt = np.loadtxt(gt_file, delimiter=",")
            except Exception:
                continue
            if pred.ndim == 1: pred = pred.reshape(1, -1)
            if gt.ndim == 1: gt = gt.reshape(1, -1)
            ious, dists = compute_seq_metrics(pred, gt)
            all_ious.extend(ious)
            all_dists.extend(dists)
            seq_metrics.append({
                "seq": seq,
                "mean_iou": float(np.mean(ious)),
                "mean_dist": float(np.mean(dists)),
            })
        if not all_ious:
            print(f"[{label}] No valid data")
            return
        thresholds = np.arange(0, 1.01, 0.01)
        success_rates = [np.mean(np.array(all_ious) >= t) for t in thresholds]
        auc = np.mean(success_rates)
        precision = np.mean(np.array(all_dists) < 20)
        mean_iou = np.mean(all_ious)
        mean_dist = np.mean(all_dists)
        print(f"\n{'='*50}")
        print(f"{label} ({len(seq_metrics)} sequences)")
        print(f"{'='*50}")
        print(f"Success AUC:       {auc:.4f}")
        print(f"Precision @ 20px:  {precision:.4f}")
        print(f"Mean IoU:          {mean_iou:.4f}")
        print(f"Mean Dist (px):    {mean_dist:.4f}")
        # Worst sequences
        seq_metrics.sort(key=lambda x: x["mean_iou"])
        print("\nWorst 5 sequences (by mean IoU):")
        for s in seq_metrics[:5]:
            print(f"  {s['seq']:40s} IoU={s['mean_iou']:.4f} Dist={s['mean_dist']:.2f}")
        # Best sequences
        print("\nBest 5 sequences (by mean IoU):")
        for s in seq_metrics[-5:]:
            print(f"  {s['seq']:40s} IoU={s['mean_iou']:.4f} Dist={s['mean_dist']:.2f}")

    compute_group(original_files, "Original 300 sequences")
    compute_group(new_files, "New 19 sequences")
    
    # Also check the first-frame absent sequence
    absent_seq = "00331_UAV_outdoor5"
    absent_file = os.path.join(args.results, f"{absent_seq}.txt")
    if os.path.exists(absent_file):
        gt_file = os.path.join(args.dataset, absent_seq, "groundtruth.txt")
        if os.path.exists(gt_file):
            pred = np.loadtxt(absent_file, delimiter=",")
            gt = np.loadtxt(gt_file, delimiter=",")
            if pred.ndim == 1: pred = pred.reshape(1, -1)
            if gt.ndim == 1: gt = gt.reshape(1, -1)
            ious, dists = compute_seq_metrics(pred, gt)
            print(f"\n{'='*50}")
            print(f"First-frame absent: {absent_seq}")
            print(f"{'='*50}")
            print(f"Mean IoU: {np.mean(ious):.4f}")
            print(f"Mean Dist: {np.mean(dists):.4f}")
            print(f"Note: This sequence may have near-zero IoU since the first frame is absent")


if __name__ == "__main__":
    main()
