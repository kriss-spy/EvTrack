# SDSTrack VisEvent Reproduction Experiment

**Parent Issue:** [#4](https://github.com/kriss-spy/EvTrack/issues/4)  
**Archive Issue:** [#18](https://github.com/kriss-spy/EvTrack/issues/18)

## Overview

This directory archives the artifacts from reproducing **SDSTrack (CVPR 2024)** on the **VisEvent** test set (RGB-E modality).

| Item | Value |
|------|-------|
| Tracker | SDSTrack (`cvpr2024_rgbe`) |
| Dataset | VisEvent test set |
| Evaluated Sequences | 319 / 320 |
| Excluded Sequence | `00331_UAV_outdoor5` (target absent in first frame) |

## Final Metrics (MATLAB-equivalent Protocol)

Computed with `scripts/eval_visevent_matlab.py` (official VisEvent MATLAB toolkit port). Absent frames are excluded.

| Metric | Reproduction | Paper (CVPR 2024) | Delta |
|--------|-------------|-------------------|-------|
| Success AUC | **0.5829** | ~0.597 | -1.4% |
| Precision @ 20px | **0.7506** | ~0.767 | -1.6% |
| SR @ 0.50 | **0.6929** | — | — |

**Verdict:** Reproduction successful. Metrics are within 2% of the paper.

## Directory Structure

```
experiments/sdstrack/
├── README.md                 # This file
├── METRICS.json              # Final metrics (JSON)
├── ENVIRONMENT.md            # Environment snapshot
├── REPRODUCTION_LOG.md       # Step-by-step reproduction log
├── requirements.txt          # Python dependencies
└── upload_results_hf.py      # Script to upload results to Hugging Face
```

## Full Results Archive

The per-sequence tracker prediction files (`*.txt`) are archived on Hugging Face:

**Dataset:** [`krisspy39/visevent-sdstrack-results`](https://huggingface.co/datasets/krisspy39/visevent-sdstrack-results)

Each `.txt` file contains the predicted bounding boxes for one VisEvent test sequence in `x,y,w,h` format (comma-separated).

## Reproduction Workflow

For the full reproduction workflow, see:
- `code/SDSTrack/SDSTrack_VisEvent_eval.ipynb` — Colab / cloud notebook
- `code/SDSTrack/sdstrack_eval.py` — Standalone cloud script
- `scripts/compute_metrics.py` — Python metrics (corrected for absent frames)
- `scripts/eval_visevent_matlab.py` — MATLAB toolkit exact port
- `scripts/verify_visevent_completeness.py` — Sequence completeness checker

## Hardware Used

| Phase | Platform | GPU | Status |
|-------|----------|-----|--------|
| Initial | Google Colab | A100 | Deadlocked (subprocess issue) |
| Main eval | RunPod | RTX 3090 | Completed 300 sequences |
| Missing 20 | AutoDL | — | Completed 19 sequences (1 excluded) |

## Known Issues & Fixes

See sub-issues for detailed investigation:
- **#14** — VisEvent test set sequence completeness verification
- **#15** — MATLAB vs Python evaluation tool comparison (absent-frame handling fix)
- **#16** — Checkpoint source and hash verification
- **#17** — AUC deviation investigation (closed as not planned after #15 fix)

## Contact

For questions about this reproduction, open an issue in the [EvTrack repository](https://github.com/kriss-spy/EvTrack).
