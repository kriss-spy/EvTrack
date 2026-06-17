#!/bin/bash
# Upload SDSTrack VisEvent results to Hugging Face
# Run this on RunPod after: pip install huggingface-hub

set -e

REPO_ID="krisspy39/sdstrack-rgbe"
RESULTS_DIR="/workspace/sdstrack/RGBE_workspace/results/VisEvent/cvpr2024_rgbe"
UPLOAD_PATH="results/vis_event_test"

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo "Get your token at: https://huggingface.co/settings/tokens"
    echo "Then run: export HF_TOKEN=hf_..."
    exit 1
fi

if ! command -v huggingface-cli &> /dev/null; then
    echo "ERROR: huggingface-cli not found. Install with:"
    echo "  pip install huggingface-hub"
    exit 1
fi

if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Results directory not found: $RESULTS_DIR"
    exit 1
fi

NUM_FILES=$(ls "$RESULTS_DIR"/*.txt 2>/dev/null | wc -l)
echo "Found $NUM_FILES result files in $RESULTS_DIR"

# Upload all .txt files to the model repo
echo "Uploading to $REPO_ID:$UPLOAD_PATH ..."
huggingface-cli upload "$REPO_ID" "$RESULTS_DIR" "$UPLOAD_PATH" \
    --repo-type model \
    --token "$HF_TOKEN"

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

huggingface-cli upload "$REPO_ID" /tmp/results_readme.md "$UPLOAD_PATH/README.md" \
    --repo-type model \
    --token "$HF_TOKEN"

echo ""
echo "Done! Results uploaded to: https://huggingface.co/$REPO_ID/tree/main/$UPLOAD_PATH"
