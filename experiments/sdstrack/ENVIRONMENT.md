# Environment Snapshot — SDSTrack VisEvent Reproduction

## Primary Environment (RunPod)

| Item | Version |
|------|---------|
| Platform | RunPod (cloud GPU instance) |
| Container | PyTorch 2.4.1 + CUDA 12.4 (NGC-based) |
| Python | 3.11.10 |
| PyTorch | 2.4.1+cu124 |
| CUDA (PyTorch) | 12.4 |
| GPU | RTX 3090 (24 GB) |

## Secondary Environment (AutoDL)

Used to evaluate the 20 missing sequences after discovering the RunPod subset was incomplete.

| Item | Version |
|------|---------|
| Platform | AutoDL (Chinese cloud GPU) |

## Tertiary Environment (Google Colab — Aborted)

| Item | Version |
|------|---------|
| Platform | Google Colab Pro |
| GPU | NVIDIA A100-SXM4-80GB |
| CUDA | 12.8 |
| PyTorch | 2.10.0 |
| Python | 3.12 |
| Status | **Aborted** — subprocess deadlock during streaming evaluation |

## SDSTrack Upstream

| Item | Value |
|------|-------|
| Repository | https://github.com/hoqolo/SDSTrack |
| Commit (approx) | `master` branch as of 2024-06 |

## Key Patches Applied

All patches are automated in `code/SDSTrack/sdstrack_eval.py`:

1. **PyTorch 2.x compatibility**
   - `collections.abc` instead of `collections.Mapping/Sequence`
   - `torch._six` fallback for `string_classes`
   - `weights_only=False` in all `torch.load()` calls
2. **Path fixes**
   - Rewrote hardcoded paths in `local.py` and test scripts
3. **LMDB stub**
   - Replaced `lmdb_utils.py` with a no-op stub (LMDB not needed for VisEvent)

## Python Dependencies (RunPod Base)

See `requirements.txt` for the full list. Key packages:

- `torch==2.4.1+cu124`
- `torchvision==0.19.1+cu124`
- `numpy==1.26.3`
- `pillow==10.2.0`
- `PyYAML==6.0.2`

**Note:** The experiment also requires SDSTrack-specific packages installed by `code/SDSTrack/upstream/install_sdstrack.sh`:
- `timm==0.5.4`
- `opencv-python`
- `easydict`
- `cython`
- `pycocotools`
- `scipy`
- `pandas`
- `jpeg4py`
- `tb-nightly`
- `lmdb`
- `visdom`
- `wandb`
- `vot-toolkit==0.5.3`
- `vot-trax==3.0.3`
- `tqdm`
- `huggingface-hub`

## Hugging Face Resources

| Resource | Repo |
|----------|------|
| VisEvent Dataset | [`krisspy39/visevent`](https://huggingface.co/datasets/krisspy39/visevent) |
| SDSTrack RGB-E Checkpoint | [`krisspy39/sdstrack-rgbe`](https://huggingface.co/krisspy39/sdstrack-rgbe) |
| OSTrack Pretrained | [`krisspy39/vipt-ostrack`](https://huggingface.co/krisspy39/vipt-ostrack) |
