#!/usr/bin/env python3
"""
Compute SDSTrack metrics (Success AUC, Precision @ 20px) from result files.

Usage:
    python compute_metrics.py \
        --results /path/to/results \
        --dataset /path/to/test_subset \
        --output metrics.json
"""

import os
import sys
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


def compute_metrics(results_dir, gt_base):
    if not os.path.exists(results_dir):
        print("ERROR: Results directory not found."); return None
    files = sorted(glob.glob(os.path.join(results_dir, "*.txt")))
    if not files:
        print("ERROR: No result files found."); return None
    print(f"Processing {len(files)} result files...")
    all_ious, all_dists = [], []
    seq_metrics = []
    for res_file in files:
        seq = os.path.basename(res_file).replace(".txt", "")
        gt_file = os.path.join(gt_base, seq, "groundtruth.txt")
        if not os.path.exists(gt_file):
            print(f"  Skipping {seq} (no groundtruth)"); continue
        try:
            pred = np.loadtxt(res_file, delimiter=",")
            gt = np.loadtxt(gt_file, delimiter=",")
        except Exception:
            print(f"  Skipping {seq} (load error)"); continue
        if pred.ndim == 1: pred = pred.reshape(1, -1)
        if gt.ndim == 1: gt = gt.reshape(1, -1)
        n = min(len(pred), len(gt))
        pred, gt = pred[:n], gt[:n]
        seq_ious, seq_dists = [], []
        for p, g in zip(pred, gt):
            iou = compute_iou(p, g)
            seq_ious.append(iou)
            all_ious.append(iou)
            pcx, pcy = p[0] + p[2]/2, p[1] + p[3]/2
            gcx, gcy = g[0] + g[2]/2, g[1] + g[3]/2
            dist = np.sqrt((pcx - gcx)**2 + (pcy - gcy)**2)
            seq_dists.append(dist)
            all_dists.append(dist)
        seq_metrics.append({
            "seq": seq,
            "mean_iou": float(np.mean(seq_ious)),
            "mean_dist": float(np.mean(seq_dists)),
        })
    if not all_ious:
        print("No valid data to compute metrics."); return None
    thresholds = np.arange(0, 1.01, 0.01)
    success_rates = [np.mean(np.array(all_ious) >= t) for t in thresholds]
    auc = np.mean(success_rates)
    precision = np.mean(np.array(all_dists) < 20)
    print(f"\n{'='*50}")
    print(f"Overall Metrics")
    print(f"{'='*50}")
    print(f"Sequences evaluated: {len(seq_metrics)}")
    print(f"Success AUC:         {auc:.4f}")
    print(f"Precision @ 20px:    {precision:.4f}")
    print(f"{'='*50}")
    return {
        "auc": float(auc),
        "precision": float(precision),
        "num_sequences": len(seq_metrics),
        "per_sequence": seq_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, default="metrics.json")
    args = parser.parse_args()
    metrics = compute_metrics(args.results, args.dataset)
    if metrics:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[Save] Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
