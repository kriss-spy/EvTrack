#!/usr/bin/env python3
"""
Compute SDSTrack metrics with absent-label handling.

Usage:
    python compute_metrics_with_absent.py \
        --results /path/to/results \
        --dataset /path/to/test_subset \
        --output metrics_with_absent.json
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


def compute_metrics(results_dir, gt_base, absent_mode="include"):
    """
    absent_mode:
        "include" - compute metrics on all frames (including absent)
        "exclude" - exclude absent frames from metrics
        "zero"    - set IoU=0 for absent frames, but include them
    """
    files = sorted(glob.glob(os.path.join(results_dir, "*.txt")))
    print(f"Processing {len(files)} result files... (absent_mode={absent_mode})")
    all_ious, all_dists = [], []
    seq_metrics = []
    total_absent = 0
    total_present = 0
    
    for res_file in files:
        seq = os.path.basename(res_file).replace(".txt", "")
        gt_file = os.path.join(gt_base, seq, "groundtruth.txt")
        absent_file = os.path.join(gt_base, seq, "absent_label.txt")
        if not os.path.exists(gt_file):
            continue
        try:
            pred = np.loadtxt(res_file, delimiter=",")
            gt = np.loadtxt(gt_file, delimiter=",")
            absent = np.loadtxt(absent_file) if os.path.exists(absent_file) else np.ones(len(gt))
        except Exception:
            continue
        if pred.ndim == 1: pred = pred.reshape(1, -1)
        if gt.ndim == 1: gt = gt.reshape(1, -1)
        n = min(len(pred), len(gt), len(absent))
        pred, gt, absent = pred[:n], gt[:n], absent[:n]
        
        seq_ious, seq_dists = [], []
        for p, g, a in zip(pred, gt, absent):
            is_absent = (a == 0)
            if is_absent:
                total_absent += 1
            else:
                total_present += 1
            
            if absent_mode == "exclude" and is_absent:
                continue
            
            iou = compute_iou(p, g)
            if absent_mode == "zero" and is_absent:
                iou = 0.0
            
            seq_ious.append(iou)
            all_ious.append(iou)
            pcx, pcy = p[0] + p[2]/2, p[1] + p[3]/2
            gcx, gcy = g[0] + g[2]/2, g[1] + g[3]/2
            dist = np.sqrt((pcx - gcx)**2 + (pcy - gcy)**2)
            seq_dists.append(dist)
            all_dists.append(dist)
        
        if seq_ious:
            seq_metrics.append({
                "seq": seq,
                "mean_iou": float(np.mean(seq_ious)),
                "mean_dist": float(np.mean(seq_dists)),
            })
    
    if not all_ious:
        print("No valid data to compute metrics.")
        return None
    
    thresholds = np.arange(0, 1.01, 0.01)
    success_rates = [np.mean(np.array(all_ious) >= t) for t in thresholds]
    auc = np.mean(success_rates)
    precision = np.mean(np.array(all_dists) < 20)
    
    print(f"\n{'='*50}")
    print(f"Metrics (absent_mode={absent_mode})")
    print(f"{'='*50}")
    print(f"Sequences:         {len(seq_metrics)}")
    print(f"Present frames:    {total_present}")
    print(f"Absent frames:     {total_absent}")
    print(f"Success AUC:       {auc:.4f}")
    print(f"Precision @ 20px:  {precision:.4f}")
    print(f"{'='*50}")
    
    return {
        "absent_mode": absent_mode,
        "auc": float(auc),
        "precision": float(precision),
        "num_sequences": len(seq_metrics),
        "present_frames": total_present,
        "absent_frames": total_absent,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, default="metrics_with_absent.json")
    args = parser.parse_args()
    
    results = {}
    for mode in ["include", "exclude", "zero"]:
        metrics = compute_metrics(args.results, args.dataset, absent_mode=mode)
        if metrics:
            results[mode] = metrics
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Save] Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
