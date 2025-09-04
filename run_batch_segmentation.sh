#!/bin/bash

# Batch Video Segmentation Script
# Example usage for processing multiple videos with DINO-X and SAM2

# Set your API token (replace with your actual token)
export DDS_API_TOKEN="your_api_token_here"

# Configuration
INPUT_VIDEO_DIR="~/exp/motion_data/prompt4_i2v"
PROMPTS_FILE="/home/oshrihalimi/video-stack/syn_data/segmentation_prompts4.txt"
OUTPUT_DIR="~/exp/motion_data/prompt4_i2v/segmentation_dinox_sam2"

# Expand tilde paths
INPUT_VIDEO_DIR=$(eval echo $INPUT_VIDEO_DIR)
OUTPUT_DIR=$(eval echo $OUTPUT_DIR)

echo "=== Batch Video Segmentation with DINO-X and SAM2 ==="
echo "Input directory: $INPUT_VIDEO_DIR"
echo "Prompts file: $PROMPTS_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if input directory exists
if [ ! -d "$INPUT_VIDEO_DIR" ]; then
    echo "❌ Error: Input video directory does not exist: $INPUT_VIDEO_DIR"
    exit 1
fi

# Check if prompts file exists
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "❌ Error: Prompts file does not exist: $PROMPTS_FILE"
    exit 1
fi

# Count videos and prompts
VIDEO_COUNT=$(find "$INPUT_VIDEO_DIR" -name "*.mp4" | wc -l)
PROMPT_COUNT=$(wc -l < "$PROMPTS_FILE")

echo "Found $VIDEO_COUNT MP4 files and $PROMPT_COUNT prompts"

if [ $VIDEO_COUNT -ne $PROMPT_COUNT ]; then
    echo "❌ Error: Number of videos ($VIDEO_COUNT) doesn't match number of prompts ($PROMPT_COUNT)"
    exit 1
fi

# Activate conda environment
echo "Activating conda environment..."
conda activate grounded-sam2

# Run the batch processing script
python batch_video_segmentation_dinox.py \
    --input_video_dir "$INPUT_VIDEO_DIR" \
    --prompts_file "$PROMPTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --box_threshold 0.2 \
    --iou_threshold 0.8 \
    --prompt_type box \
    --sam2_checkpoint "./checkpoints/sam2.1_hiera_large.pt" \
    --sam2_config "configs/sam2.1/sam2.1_hiera_l.yaml"

echo ""
echo "=== Batch processing complete ==="
echo "Results saved to: $OUTPUT_DIR"
