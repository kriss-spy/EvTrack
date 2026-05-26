# Evaluation

Shared evaluation toolkit for event-camera tracking benchmarks.

## Metrics

- **Precision (P)**: Center location error within a threshold (typically 20 pixels).
- **Success Rate (S)**: Intersection-over-union (IoU) overlap above a threshold (typically 0.5).
- **Robustness**: Failure rate or re-initialization counts under challenging conditions.

## Usage

```bash
python evaluation/run_eval.py \
    --tracker_dir <path_to_tracker_results> \
    --dataset <VisEvent|COESOT> \
    --output_dir results/
```

## Structure

- `metrics.py`: Core metric implementations (precision, success, robustness).
- `run_eval.py`: CLI entry point for running evaluation on a tracker output directory.
