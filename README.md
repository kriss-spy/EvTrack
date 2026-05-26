# EvTrack: Event Camera-based Object Tracking

## Overview

EvTrack is a research project focused on object tracking using event cameras. Event cameras offer microsecond temporal resolution and high dynamic range, making them ideal for high-speed and extreme lighting scenarios where traditional frame-based cameras struggle.

This project reproduces and evaluates the [ViPT (Visual Prompt Multi-Modal Tracking)](https://github.com/jiawen-zhu/ViPT) algorithm on event-camera tracking benchmarks.

## Datasets

- [VisEvent](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark)
- [COESOT](https://github.com/Event-AHU/COESOT)

See [`docs/dataset-setup.md`](docs/dataset-setup.md) for download and preparation instructions.

## Project Structure

```
.
├── code/           # Tracker implementation (ViPT)
├── evaluation/     # Shared evaluation metrics and scripts
├── data/           # Dataset paths and setup guides (not raw data)
├── results/        # Evaluation outputs, plots, videos
├── docs/           # Documentation and guides
└── requirements.txt
```

## Quick Start

```bash
# Install base dependencies
pip install -r requirements.txt

# Install tracker-specific dependencies
pip install -r code/requirements.txt

# Run evaluation
python evaluation/run_eval.py --tracker_dir <path> --dataset <name>
```

## References

[1] Zhu J, Lai S, Chen X, et al. Visual prompt multi-modal tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition 2023: 9516-9526.

[2] Wang X, Li J, Zhu L, et al. VisEvent: Reliable object tracking via collaboration of frame and event flows. IEEE transactions on cybernetics. 2023, 54(3):1997-2010.

[3] Tang C, Wang X, Huang J, et al. Revisiting color-event based tracking: A unified network, dataset, and metric. Pattern Recognition. 2025, 7:112718.

## License

This project is for academic and research purposes.
