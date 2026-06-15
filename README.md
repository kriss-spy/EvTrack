# EvTrack: Event Camera-based Object Tracking

## Overview

EvTrack is a research project focused on object tracking using event cameras. Event cameras offer microsecond temporal resolution and high dynamic range, making them ideal for high-speed and extreme lighting scenarios where traditional frame-based cameras struggle.

This project reproduces and evaluates the [ViPT (Visual Prompt Multi-Modal Tracking)](https://github.com/jiawen-zhu/ViPT) algorithm on event-camera tracking benchmarks.

## Documentation

- [`docs/project-proposal.md`](docs/project-proposal.md) — 开题报告 (project proposal, in Chinese)
- [`docs/dataset-setup.md`](docs/dataset-setup.md) — Dataset download and preparation guide
- [`docs/course-project-guide.md`](docs/course-project-guide.md) — Course design requirements and grading rubric

## Datasets

- [VisEvent](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark)
- [COESOT](https://github.com/Event-AHU/COESOT)

## Project Structure

```
.
├── code/               # Tracker implementation
│   ├── vipt/           #   ViPT upstream (submodule)
│   └── patches/        #   Team modifications
├── evaluation/         # Shared evaluation metrics and scripts
├── data/               # Dataset paths and setup guides (not raw data)
├── results/            # Evaluation outputs, plots, videos
├── docs/               # Documentation and guides
├── AGENTS.md           # Agent workflow instructions
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/kriss-spy/EvTrack

# Or if already cloned:
# git submodule update --init

# Install base dependencies
pip install -r requirements.txt

# Install tracker-specific dependencies
pip install -r code/requirements.txt

# Run evaluation
python evaluation/run_eval.py --tracker_dir <path> --dataset <name>
```

## Improvement

And we have found some problems in the original vipt and ……  
🎆SURPRISINGLY!!!🎆  
we also have made a improvement to the original ViPT tracker to make it have better performance!

- add a pair of online templates(include rgb and event),with simple alogrithm and clever little trick to avoid updating error template.
- TOTALLY WITHOUT ANY OTHER ADDITIONAL TRANING.Just use the orig pth file!

**WARNING:Our approach cannot fundamentally solve vipt's problem; it is alike a "remedial measure". When orig result is quite bad, this method still cannot solve the problem.**

The following GIF demonstrates the tracking performance improvements on some VisEvent benchmark sequences:
![visevent video0079](./gif/video0079.gif)
![visevent video0079](./gif/compare2.gif)
![visevent video0079](./gif/compare3.gif)
![visevent video0079](./gif/compare4.gif)

**AND WE STILL FIND IN SOME SITUATIONS THAT THIS METHOD WILL HURT PERFORMANCE🫥**

## References

[1] Zhu J, Lai S, Chen X, et al. Visual prompt multi-modal tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition 2023: 9516-9526.

[2] Wang X, Li J, Zhu L, et al. VisEvent: Reliable object tracking via collaboration of frame and event flows. IEEE transactions on cybernetics. 2023, 54(3):1997-2010.

[3] Tang C, Wang X, Huang J, et al. Revisiting color-event based tracking: A unified network, dataset, and metric. Pattern Recognition. 2025, 7:112718.

## License

This project is for academic and research purposes.
