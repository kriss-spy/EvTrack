# SDSTrack Reproduction Log

## Timeline

### Phase 1: Initial Attempt (Google Colab)
- **Date:** May 2026
- **Platform:** Google Colab Pro (A100)
- **Issue:** Subprocess deadlock during streaming evaluation (`test_rgbe_mgpus.py` hangs after ~40-50 sequences)
- **Root Cause:** Model reloaded for each sequence + stdout blocking in subprocess communication
- **Decision:** Abandon Colab, migrate to RunPod

### Phase 2: Main Evaluation (RunPod)
- **Date:** Early June 2026
- **Platform:** RunPod (RTX 3090)
- **Approach:** Rewrote evaluation as direct Python (`sdstrack_eval.py`) — no subprocess, model kept in memory
- **Streaming:** One webdataset tar shard at a time, extract-evaluate-delete to stay under ~5 GB disk
- **Result:** 300 sequences evaluated successfully
- **Issue Discovered:** `progress.json` showed 300 completed, but VisEvent test set has 320 sequences

### Phase 3: Completeness Verification (Issue #14)
- **Date:** June 16, 2026
- **Finding:** 20 sequences missing from RunPod evaluation
- **Cause:** Data download/extraction gaps during streaming
- **Action:** Evaluated missing 20 sequences on AutoDL
- **Result:** 19 sequences completed, 1 (`00331_UAV_outdoor5`) excluded (target absent in first frame)
- **Final Coverage:** 319/320 evaluable sequences

### Phase 4: Metrics Validation (Issue #15)
- **Date:** June 16-17, 2026
- **Finding:** Original Python metrics (Success AUC=0.6252, Precision=0.7715) were ~2.8% higher than paper
- **Investigation:** Python evaluation incorrectly included absent frames in IoU computation
- **Fix:** Ported official VisEvent MATLAB toolkit to Python (`eval_visevent_matlab.py`)
- **Corrected Metrics:**
  - Success AUC: **0.5829**
  - Precision @ 20px: **0.7506**
  - SR @ 0.50: **0.6929**
- **Conclusion:** Now within 2% of paper; reproduction validated

### Phase 5: Checkpoint Verification (Issue #16)
- **Date:** June 16, 2026
- **Action:** Verified SHA256 and source of `SDSTrack_cvpr2024_rgbe.pth.tar`
- **Result:** Checkpoint from official Hugging Face repo, integrity confirmed

### Phase 6: AUC Deviation Investigation (Issue #17)
- **Date:** June 16, 2026
- **Status:** Closed as **not planned**
- **Reason:** Deviation explained and fixed by issue #15 (absent-frame bug). No further action needed.

## Key Files Modified / Created

| File | Purpose |
|------|---------|
| `code/SDSTrack/sdstrack_eval.py` | Standalone cloud evaluation script |
| `code/SDSTrack/SDSTrack_VisEvent_eval.ipynb` | Colab notebook (template) |
| `scripts/compute_metrics.py` | Python metrics (absent-frame aware) |
| `scripts/eval_visevent_matlab.py` | MATLAB toolkit exact port |
| `scripts/compute_metrics_with_absent.py` | Diagnostic: metrics with/without absent frames |
| `scripts/diagnose_metrics.py` | Diagnostic: per-sequence metric comparison |
| `scripts/verify_visevent_completeness.py` | Sequence completeness checker |

## Lessons Learned

1. **Subprocess evaluation is risky** — Keep model in memory, evaluate directly in Python.
2. **Streaming needs verification** — Always check that all sequences were processed; `progress.json` alone is not enough.
3. **Absent frames matter** — VisEvent's absent labels must be respected; including them inflates metrics artificially.
4. **Cloud instances differ** — Colab's subprocess limitations, RunPod's disk constraints, and AutoDL's environment each introduced unique challenges.
5. **Archive early** — Result files should be saved to persistent storage (Drive / HF) immediately, not just locally.
