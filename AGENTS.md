# AGENTS

## Project Context

- This is a university **Pattern Recognition (模式识别) course design** repository.
- Active project: **`EvTrack`** — event-camera-based object tracking (topic #65).

## Working Notes

- **Package/Env management:** Prefer **`uv`** (for Python projects) and **`mamba`** (for Conda-compatible envs). Do **not** use `conda` directly.
- **Compute:** **Do not run experiments (training, evaluation, heavy data processing) locally.** All experiments run in cloud GPU, like **`SDSTrack_VisEvent_eval.ipynb` Colab notebook** connected via Colab MCP.
- The project involves deep-learning trackers (ViPT, SDSTrack). If cloning those upstream repos as subdirectories, consider adding them to `.gitignore` to avoid committing vendored code.

## Current Progress (as of 2026-06-06)

### ViPT

mostly done

writing report

### SDSTrack

#### before 2026-06-06

reproduction on broken visevent dataset
full visevent dataset uploaded to huggingface
full test reproduction to be done

### Dataset Acquisition

- **VisEvent dataset (232 GB)** has been successfully downloaded from Dropbox and uploaded to a Hugging Face dataset (https://huggingface.co/datasets/krisspy39/visevent), webdataset available.
- The download consists of 13 zip parts:
  - `VisEvent_test.zip` + `.z01`–`.z05` (~102 GB total)
  - `VisEvent_train.zip` + `.z01`–`.z06` (~130 GB total)
- **VisEvent extracted structure:**
  - `train/train_subset/` — 120 sequences (each with `vis_imgs/`, `event_imgs/`, `groundtruth.txt`, `absent_label.txt`)
  - `test/test_subset/` — 77 sequences (same structure)
- **COESOT dataset (required)** — acquisition is still pending. COESOT is hosted on Baidu Netdisk, which is not easily downloadable from Colab; this needs a separate strategy (e.g., local download or shared drive sync).

### Colab Environment

- **GPU:** Tesla T4 (CUDA 12.8) by default (better GPU like A100 available)
