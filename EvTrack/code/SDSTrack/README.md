# SDSTrack Reproduction

Reproduction of [SDSTrack](https://github.com/hoqolo/SDSTrack) for the Pattern Recognition course design (topic #65: event-camera-based object tracking).

## Structure

- `upstream/` — cloned upstream SDSTrack repository (gitignored, do not commit)
- `docs/` — documentation for dataset mounting and experiment notes
- `PLAN.md` — detailed reproduction plan

## Quick Start

All heavy compute (training, evaluation, data preprocessing) runs in the **`SDSTrack.ipynb`** Colab notebook via Colab MCP. This local directory is for code editing and documentation only.

### Prerequisites

- Python 3.13+ (local editing)
- [uv](https://docs.astral.sh/uv/) for local dependency management
- Google Colab with GPU runtime for experiments

### Local Setup (Coding Only)

```bash
uv sync
```

### Experiment Environment (Colab)

The upstream code requires:
- Python 3.8
- PyTorch 1.11.0 + CUDA 11.3
- See `upstream/install_sdstrack.sh` for full dependency list

In Colab, use `mamba` (not `conda`) if a Conda-compatible environment is needed.

## Datasets

- **VisEvent** (required) — downloaded to Google Drive (`MyDrive/EvTrack/datasets/VisEvent/`)
- **COESOT** (required) — pending acquisition (hosted on Baidu Netdisk)

## Training

```bash
# In Colab, after mounting Google Drive and setting up environment
bash train_sdstrack_rgbe.sh
```

## Evaluation

```bash
# In Colab
bash eval_rgbe.sh
```

## Upstream Resources

- Paper: Hou X, et al. SDSTrack: Self-Distillation Symmetric Adapter Learning for Multi-Modal Visual Object Tracking. CVPR 2024.
- Code: https://github.com/hoqolo/SDSTrack
