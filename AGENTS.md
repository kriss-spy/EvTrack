# AGENTS

## Project Context

- This is a university **Pattern Recognition (模式识别) course design** repository.
- Active project: **`EvTrack/`** — event-camera-based object tracking (topic #65).
- The repo currently contains **documentation only** (`README.md` files + a PDF course guide). No source code, build system, or dependencies are configured yet.

## Structure

- `/README.md` — Course logistics, grading criteria, team formation rules (Chinese).
- `/EvTrack/README.md` — Project specification: literature review + reproduction of **ViPT** or **SDSTrack** on VisEvent/COESOT datasets.
- `/.gitignore` — Standard Python gitignore (implies Python is the expected implementation language).

## Working Notes

- **No build, test, or lint tooling is present.** Any implementation added under `EvTrack/` will need its own environment setup.
- **Package/Env management:** Prefer **`uv`** (for Python projects) and **`mamba`** (for Conda-compatible envs). Do **not** use `conda` directly.
- **Compute:** This laptop is for coding and documentation only. **Do not run experiments (training, evaluation, heavy data processing) locally.** All experiments run in the **`SDSTrack.ipynb` Colab notebook** connected via Colab MCP.
- The project involves deep-learning trackers (ViPT, SDSTrack). If cloning those upstream repos as subdirectories, consider adding them to `.gitignore` to avoid committing vendored code.
- Datasets (VisEvent, COESOT) are large; do not commit them. Use `.gitignore` or symlink strategies.

## Active Issues

- **#3** `ViPT复现` — CYJ
- **#4** `SDSTrack复现` — @kriss-spy (topic #65 programmer 2)
- **#5** `文献调研和综述报告` — @melioll, @zhangrongxuan, JTY

## Current Progress (as of 2026-05-06)

### Dataset Acquisition
- **VisEvent dataset (232 GB)** has been successfully downloaded from Dropbox to a shared Google Drive (`MyDrive/EvTrack/datasets/VisEvent/`).
- The download consists of 13 zip parts:
  - `VisEvent_test.zip` + `.z01`–`.z05` (~102 GB total)
  - `VisEvent_train.zip` + `.z01`–`.z06` (~130 GB total)
- **VisEvent extracted structure:**
  - `train/train_subset/` — 120 sequences (each with `vis_imgs/`, `event_imgs/`, `groundtruth.txt`, `absent_label.txt`)
  - `test/test_subset/` — 77 sequences (same structure)
- **COESOT dataset (required)** — acquisition is still pending. COESOT is hosted on Baidu Netdisk, which is not easily downloadable from Colab; this needs a separate strategy (e.g., local download or shared drive sync).

### Local Codebase
- `EvTrack/code/SDSTrack/` — uv project initialized (Python 3.13, placeholder `main.py`).
- `EvTrack/code/SDSTrack/docs/dataset-mount.md` — guide for downloading datasets into Colab via Google Drive.
- `EvTrack/code/SDSTrack/upstream/` — cloned SDSTrack upstream repository (gitignored).
- `EvTrack/code/SDSTrack/scripts/patch_colab_compat.py` — compatibility patches for PyTorch 2.x + Python 3.10+.

### Colab Environment
- **GPU:** Tesla T4 (CUDA 12.8)
- **PyTorch:** 2.10.0+cu128 (upstream requires 1.11.0; compatibility patches applied)
- **Status:** Environment setup complete, imports verified successfully
- **Evaluation:** Phase 4 complete for VisEvent — **Success AUC: 0.6238, Precision (20px): 0.7356** (76/77 sequences, 1 incomplete sequence excluded)
- **Datasets:** Both train (120 seqs) and test (77 seqs) extracted and verified

## Verification

- If code is added, prefer creating `EvTrack/requirements.txt`, `pyproject.toml` (uv), or `EvTrack/environment.yml` (mamba) and document how to run training/evaluation there.
