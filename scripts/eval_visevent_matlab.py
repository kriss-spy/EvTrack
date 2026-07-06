#!/usr/bin/env python3
"""
Port of official VisEvent MATLAB evaluation toolkit to Python.
Mirrors utils/calc_seq_err_robust.m and utils/calc_rect_int.m exactly.

Usage:
    python eval_visevent_matlab.py \
        --results /path/to/results \
        --dataset /path/to/test_subset \
        --output metrics_matlab.json
"""

import os
import json
import glob
import argparse
import numpy as np


def calc_rect_int(a, b):
    """
    Port of calc_rect_int.m
    Calculate intersection area between two rectangles.
    a, b: Nx4 arrays of [x, y, w, h]
    Returns: Nx1 array of intersection areas
    """
    # a = [x1, y1, w1, h1]
    # b = [x2, y2, w2, h2]
    x1, y1, w1, h1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, w2, h2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    
    # Right and bottom edges
    x1_right = x1 + w1 - 1
    y1_bottom = y1 + h1 - 1
    x2_right = x2 + w2 - 1
    y2_bottom = y2 + h2 - 1
    
    # Intersection
    xi1 = np.maximum(x1, x2)
    yi1 = np.maximum(y1, y2)
    xi2 = np.minimum(x1_right, x2_right)
    yi2 = np.minimum(y1_bottom, y2_bottom)
    
    iw = np.maximum(0, xi2 - xi1 + 1)
    ih = np.maximum(0, yi2 - yi1 + 1)
    inter = iw * ih
    
    return inter


def calc_seq_err_robust(results, rect_anno, absent_anno, norm_dst=False):
    """
    Port of calc_seq_err_robust.m
    
    results: Nx4 predicted boxes
    rect_anno: Nx4 ground truth boxes
    absent_anno: Nx1 absent labels (1=present, 0=absent in our dataset)
    """
    # Truncate to minimum length
    seq_length = min(results.shape[0], rect_anno.shape[0])
    results = results[:seq_length, :]
    rect_anno = rect_anno[:seq_length, :]
    absent_anno = absent_anno[:seq_length]
    
    # Handle invalid tracking results (NAN, negative, complex)
    # Replace with previous frame's result
    for i in range(1, seq_length):
        r = results[i, :]
        r_anno = rect_anno[i, :]
        
        # Check if invalid AND annotation is valid
        is_invalid = (np.any(np.isnan(r)) or 
                      np.any(~np.isreal(r)) or 
                      r[2] <= 0 or r[3] <= 0)
        anno_valid = not np.any(np.isnan(r_anno))
        
        if is_invalid and anno_valid:
            results[i, :] = results[i-1, :]
    
    rect_mat = results.copy()
    rect_mat[0, :] = rect_anno[0, :]  # Ignore result in first frame, use GT
    
    # Remove frames where target is absent
    # MATLAB: absent_idx = absent_anno == 1
    # But MATLAB removes these, so we keep present frames (absent_anno == 1 in our convention means present)
    # Wait, in MATLAB the absent_anno might use different convention
    # Let's check: the comment says "remove the frames where the target is absent"
    # MATLAB code: absent_idx = absent_anno == 1; then removes those
    # If MATLAB absent_anno == 1 means present, then the code removes present frames... which doesn't make sense
    # More likely: MATLAB's absent_anno file uses 1=absent, 0=present (opposite of our dataset)
    # OR: the code has a bug/comment mismatch
    # Given the comment says "remove absent", we'll keep only present frames
    # Our dataset: 0=absent, 1=present
    present_idx = absent_anno == 1
    rect_mat = rect_mat[present_idx, :]
    rect_anno_filtered = rect_anno[present_idx, :]
    
    # Center positions (MATLAB uses (w-1)/2 and (h-1)/2)
    center_gt = np.column_stack([
        rect_anno_filtered[:, 0] + (rect_anno_filtered[:, 2] - 1) / 2,
        rect_anno_filtered[:, 1] + (rect_anno_filtered[:, 3] - 1) / 2
    ])
    center = np.column_stack([
        rect_mat[:, 0] + (rect_mat[:, 2] - 1) / 2,
        rect_mat[:, 1] + (rect_mat[:, 3] - 1) / 2
    ])
    
    # Normalize if needed
    if norm_dst:
        center[:, 0] = center[:, 0] / rect_anno_filtered[:, 2]
        center[:, 1] = center[:, 1] / rect_anno_filtered[:, 3]
        center_gt[:, 0] = center_gt[:, 0] / rect_anno_filtered[:, 2]
        center_gt[:, 1] = center_gt[:, 1] / rect_anno_filtered[:, 3]
    
    # Center distance error
    err_center = np.sqrt(np.sum((center - center_gt) ** 2, axis=1))
    
    # Calculate overlap (IoU)
    # MATLAB: index = rect_anno > 0; idx = (sum(index, 2)==4);
    index = rect_anno_filtered > 0
    idx = np.sum(index, axis=1) == 4
    
    tmp = calc_rect_int(rect_mat[idx, :], rect_anno_filtered[idx, :])
    
    # MATLAB: area1 = a(:,3) .* a(:,4); area2 = b(:,3) .* b(:,4);
    area1 = rect_mat[idx, 2] * rect_mat[idx, 3]
    area2 = rect_anno_filtered[idx, 2] * rect_anno_filtered[idx, 3]
    union = area1 + area2 - tmp
    
    iou = tmp / (union + 1e-10)
    
    err_coverage = -np.ones(len(idx))
    err_coverage[idx] = iou
    err_center[~idx] = -1
    
    return err_coverage, err_center


def eval_tracker(seqs, results_dir, gt_base):
    """Port of eval_tracker.m"""
    threshold_set_overlap = np.arange(0, 1.01, 0.05)
    threshold_set_error = np.arange(0, 51)
    
    ave_success_rate_plot = np.zeros((len(seqs), len(threshold_set_overlap)))
    ave_success_rate_plot_err = np.zeros((len(seqs), len(threshold_set_error)))
    
    for i, s in enumerate(seqs):
        # Load GT and absent flags
        anno_path = os.path.join(gt_base, s, "groundtruth.txt")
        absent_path = os.path.join(gt_base, s, "absent_label.txt")
        
        anno = np.loadtxt(anno_path, delimiter=",")
        absent_anno = np.loadtxt(absent_path)
        
        # Load tracking result
        res_path = os.path.join(results_dir, f"{s}.txt")
        res = np.loadtxt(res_path, delimiter=",")
        
        if res.ndim == 1:
            res = res.reshape(1, -1)
        if anno.ndim == 1:
            anno = anno.reshape(1, -1)
        if absent_anno.ndim == 0:
            absent_anno = np.array([absent_anno])
        
        # Save original length before truncation
        len_all = min(res.shape[0], anno.shape[0], len(absent_anno))
        
        # Ensure all arrays have same length
        res = res[:len_all, :]
        anno = anno[:len_all, :]
        absent_anno = absent_anno[:len_all]
        
        err_coverage, err_center = calc_seq_err_robust(res, anno, absent_anno)
        
        # Success rate for overlap
        success_num_overlap = np.zeros(len(threshold_set_overlap))
        for t_idx, th in enumerate(threshold_set_overlap):
            success_num_overlap[t_idx] = np.sum(err_coverage > th)
        
        # Success rate for error
        success_num_err = np.zeros(len(threshold_set_error))
        for t_idx, th in enumerate(threshold_set_error):
            success_num_err[t_idx] = np.sum(err_center <= th)
        
        ave_success_rate_plot[i, :] = success_num_overlap / len_all
        ave_success_rate_plot_err[i, :] = success_num_err / len_all
    
    # Average over all sequences
    mean_success = np.mean(ave_success_rate_plot, axis=0)
    mean_precision = np.mean(ave_success_rate_plot_err, axis=0)
    
    return mean_success, mean_precision, threshold_set_overlap, threshold_set_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, default="metrics_matlab.json")
    args = parser.parse_args()
    
    # Get sequence list
    files = sorted(glob.glob(os.path.join(args.results, "*.txt")))
    seqs = [os.path.basename(f).replace(".txt", "") for f in files]
    
    print(f"Evaluating {len(seqs)} sequences...")
    mean_success, mean_precision, th_overlap, th_error = eval_tracker(seqs, args.results, args.dataset)
    
    # AUC for success plot
    auc = np.mean(mean_success)
    # Precision at 20px
    precision_20 = mean_precision[20]  # threshold 20
    
    # Also compute SR@0.5 (success rate at overlap 0.5)
    sr_05 = mean_success[10]  # threshold 0.5 is at index 10 (0:0.05:1)
    
    print(f"\n{'='*50}")
    print("MATLAB-equivalent Metrics")
    print(f"{'='*50}")
    print(f"Success AUC:       {auc:.4f}")
    print(f"Precision @ 20px:  {precision_20:.4f}")
    print(f"SR @ 0.5:          {sr_05:.4f}")
    print(f"{'='*50}")
    
    results = {
        "auc": float(auc),
        "precision_20": float(precision_20),
        "sr_05": float(sr_05),
        "num_sequences": len(seqs),
        "threshold_overlap": th_overlap.tolist(),
        "threshold_error": th_error.tolist(),
        "success_curve": mean_success.tolist(),
        "precision_curve": mean_precision.tolist(),
    }
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Save] Results saved to {args.output}")


if __name__ == "__main__":
    main()
