# SDSTrack Reproduction

Reproduction of [SDSTrack](https://github.com/hoqolo/SDSTrack) for the Pattern Recognition course design (topic #65: event-camera-based object tracking).

## Structure

- `upstream/` — cloned upstream SDSTrack repository (gitignored, do not commit)
- `docs/` — documentation for dataset mounting and experiment notes

## Quick Start

All heavy compute (training, evaluation, data preprocessing) runs in the **`SDSTrack.ipynb`** Colab notebook via Colab MCP.

### Experiment Environment (Colab)

The upstream code requires:
- Python 3.8
- PyTorch 1.11.0 + CUDA 11.3
- See `upstream/install_sdstrack.sh` for full dependency list

In Colab, use `mamba` (not `conda`) if a Conda-compatible environment is needed.

## Datasets

- **VisEvent** (required) — [GitHub](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark), [Hugging Face](https://huggingface.co/datasets/krisspy39/visevent)
- **COESOT** (required) — [GitHub](https://github.com/Event-AHU/COESOT), pending acquisition (hosted on Baidu Netdisk)
