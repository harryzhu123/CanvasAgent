#!/bin/bash
# Batch Image Editing with Optimized GPU Utilization
# This script runs the image editing with optimized configuration for 8 A800 GPUs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_FILE="$SCRIPT_DIR/test_image_edit.py"
LOG_FILE="/nfsdata4/zhuhairui/EDIT/processing.log"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')

echo "======================================================================"
echo "Image Editing Job Launcher - GPU Optimized"
echo "======================================================================"
echo "Timestamp: $TIMESTAMP"
echo "Script: $SCRIPT_FILE"
echo "Log file: $LOG_FILE"
echo ""

# Check if script exists
if [ ! -f "$SCRIPT_FILE" ]; then
    echo "❌ Error: Script not found at $SCRIPT_FILE"
    exit 1
fi

# Check if GPU is available
echo "Checking GPU availability..."
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "✓ Found $GPU_COUNT GPU(s)"
echo ""

# Show GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.free,memory.max,utilization.gpu --format=csv,noheader | \
    awk -F', ' '{printf "  GPU %s: %s (%s/%s MB) - Util: %s\n", $1, $2, $3, $4, $5}'
echo ""

# Show current configuration
echo "Configuration:"
echo "  - Workers: 6"
echo "  - GPUs per worker: 1.0"
echo "  - Total GPUs used: 6"
echo "  - Rate limit: 100 req/sec"
echo ""

# Run the script
echo "Starting image editing process..."
echo "======================================================================"
echo ""

cd "$SCRIPT_DIR"

# Run in background with logging
python3 "$SCRIPT_FILE" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "Job completed with exit code: $EXIT_CODE"
echo "======================================================================"
echo ""

# Show final GPU status
echo "Final GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.max,utilization.gpu --format=csv,noheader | \
    awk -F', ' '{printf "  GPU %s: %s (%s/%s MB) - Util: %s\n", $1, $2, $3, $4, $5}'

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Processing completed successfully"
    echo "Output files:"
    echo "  - JSON: /nfsdata4/zhuhairui/EDIT/processed_data_20k_edited.json"
    echo "  - Images: /nfsdata4/zhuhairui/EDIT/edited_images/"
else
    echo "❌ Processing failed with exit code: $EXIT_CODE"
    echo "Check log file: $LOG_FILE"
fi

exit $EXIT_CODE
