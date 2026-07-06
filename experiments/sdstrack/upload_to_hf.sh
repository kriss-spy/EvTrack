#!/bin/bash
# Upload SDSTrack VisEvent results to Hugging Face
# Run this on RunPod after: curl -LsSf https://hf.co/cli/install.sh | bash

set -e

REPO_ID="krisspy39/sdstrack-rgbe"
RESULTS_DIR="/workspace/sdstrack/RGBE_workspace/results/VisEvent/cvpr2024_rgbe"
UPLOAD_PATH="results/vis_event_test"

# Try to find hf CLI (may need PATH update after install)
HF_BIN="$(command -v hf 2>/dev/null || echo "/root/.local/bin/hf")"
if [ ! -x "$HF_BIN" ]; then
    echo "ERROR: hf CLI not found. Install with:"
    echo "  curl -LsSf https://hf.co/cli/install.sh | bash"
    echo "Then reload your shell: source ~/.zshrc"
    exit 1
fi

if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Results directory not found: $RESULTS_DIR"
    exit 1
fi

NUM_FILES=$(ls "$RESULTS_DIR"/*.txt 2>/dev/null | wc -l)
echo "Found $NUM_FILES result files in $RESULTS_DIR"

# Upload all result files to the model repo
echo "Uploading to $REPO_ID:$UPLOAD_PATH ..."
"$HF_BIN" upload "$REPO_ID" "$RESULTS_DIR" "$UPLOAD_PATH" \
    --repo-type model \
    --commit-message "Add SDSTrack VisEvent test results" \
    --commit-description "Tracker predictions for 320 VisEvent test sequences. See https://github.com/kriss-spy/EvTrack/issues/18"

# Also upload a small README for context
cat > /tmp/results_readme.md << 'INNEREOF'
# SDSTrack VisEvent Test Results

Tracker: SDSTrack (cvpr2024_rgbe)
Dataset: VisEvent test set (320 sequences)
Format: x,y,w,h (comma-separated), one line per frame

## Metrics (MATLAB-equivalent protocol, absent frames excluded)

| Metric | Value |
|--------|-------|
| Success AUC | 0.5829 |
| Precision @ 20px | 0.7506 |
| SR @ 0.50 | 0.6929 |

For full reproduction details see:
https://github.com/kriss-spy/EvTrack/tree/sdstrack/experiments/sdstrack
INNEREOF

"$HF_BIN" upload "$REPO_ID" /tmp/results_readme.md "$UPLOAD_PATH/README.md" \
    --repo-type model \
    --commit-message "Add README for VisEvent results"

echo ""
echo "Done! Results uploaded to: https://huggingface.co/$REPO_ID/tree/main/$UPLOAD_PATH"
