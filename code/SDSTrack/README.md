# SDSTrack Reproduction

Reproduction of [SDSTrack](https://github.com/hoqolo/SDSTrack) for the Pattern Recognition course design (topic #65: event-camera-based object tracking).

## Structure

- `upstream/` — cloned upstream SDSTrack repository (gitignored, do not commit)
- `docs/` — documentation for dataset mounting and experiment notes

## Quick Start

All heavy compute (evaluation, data download) runs in the **`SDSTrack_VisEvent_eval.ipynb`** Colab notebook via Colab MCP.

> **Note:** `SDSTrack_VisEvent_eval_backup.ipynb` is the previous broken/legacy version. Use the new notebook above.

### Experiment Environment (Colab)

The upstream code requires:
- Python 3.8
- PyTorch 1.11.0 + CUDA 11.3
- See `upstream/install_sdstrack.sh` for full dependency list

In Colab, use `mamba` (not `conda`) if a Conda-compatible environment is needed.

### What's New (vs. Backup Notebook)

| | Old (`_backup.ipynb`) | New (`SDSTrack_VisEvent_eval.ipynb`) |
|---|---|---|
| **Phases** | ~70 cells, broken order, fix cells scattered | 24 cells, strict linear flow |
| **Dataset** | Dropbox zip splits + manual 7z extraction (~102 GB at once) | **Streaming**: one tar shard at a time, disk stays < 5 GB |
| **Training** | Skipped but cluttering cells present | **Removed entirely** (cancelled per Issue #4) |
| **Config** | Hardcoded `/content/...` strings everywhere | Single `CONFIG` dict in P0 |
| **Patches** | Spread across wrong phases | Consolidated in P1.5 + safety P3.2 |
| **Outputs** | Embedded error traces from failed runs | Clean, no outputs |
| **Resumability** | None; Colab disconnect = start over | Progress saved to Drive after **every tar shard** |
| **Results** | Lost on disconnect | Persisted to Google Drive via symlink |
| **Model source** | Google Drive shortcuts only | Hugging Face Hub (with Drive fallback) |

## Original Experimental Setup (CVPR 2024)

The authors trained on **4× NVIDIA RTX 3090 Ti** (batch size 64, distributed over 4 GPUs) and evaluated on a single 3090 Ti (~20.86 fps).

### Steps

1. **Create environment**
   ```bash
   conda create -n sdstrack python=3.8
   conda activate sdstrack
   bash install_sdstrack.sh
   ```

2. **Prepare data** — Place datasets under `./data/`:
   ```
   ./data/visevent/train/...
   ./data/lasher/trainingset/...
   ./data/depthtrack/train/...
   ```

3. **Set project paths**
   ```bash
   python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
   ```

4. **Download pretrained foundation model** (OSTrack ViT-B) to `./pretrained/vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar`

5. **Train** (example for RGB-E / VisEvent)
   ```bash
   bash train_sdstrack_rgbe.sh
   ```
   Other variants: `train_sdstrack_rgbt.sh` (RGB-T), `train_sdstrack_rgbd.sh` (RGB-D)

6. **Evaluate** (example for VisEvent)
   ```bash
   bash eval_rgbe.sh
   ```
   Other variants: `bash eval_rgbt.sh`, `bash eval_rgbd.sh`

### Training Config Summary

| Modality | Dataset | Config | Epochs | Batch Size | LR |
|----------|---------|--------|--------|------------|----|
| RGB-D | DepthTrack | `cvpr2024_rgbd` | 15 | 64 | 1e-4 |
| RGB-T | LasHeR | `cvpr2024_rgbt` | 40 | 64 | 1e-4 |
| RGB-E | VisEvent | `cvpr2024_rgbe` | 50 | 16 | 1e-4 |

Architecture: ViT-B backbone (MAE pretrained) with lightweight adapters, symmetric multimodal fusion, and complementary masked patch distillation. See `upstream/experiments/sdstrack/` for full configs.

## Datasets

- **VisEvent** (required) — [GitHub](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark), [Hugging Face](https://huggingface.co/datasets/krisspy39/visevent)
- **COESOT** (required) — [GitHub](https://github.com/Event-AHU/COESOT), pending acquisition (hosted on Baidu Netdisk)
