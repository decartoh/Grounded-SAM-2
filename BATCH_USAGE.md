# Batch Video Segmentation with DINO-X and SAM2

This script processes multiple MP4 videos with corresponding text prompts for object segmentation and tracking.

## Files Created

- `batch_video_segmentation_dinox.py` - Main batch processing script
- `run_batch_segmentation.sh` - Shell script with example usage  
- `example_prompts.txt` - Example prompts file format

## Usage

### 1. Prepare Your Data

**Video Directory Structure:**
```
~/exp/motion_data/prompt4_i2v/
├── video_001.mp4
├── video_002.mp4
├── video_003.mp4
└── ...
```

**Prompts File Format** (one prompt per line, corresponding to video index):
```
person walking
car driving down the street  
dog running in the park
bird flying in the sky
```

### 2. Set Environment Variable

```bash
export DDS_API_TOKEN="your_actual_api_token_here"
```

### 3. Run Batch Processing

#### Option A: Using the Shell Script
```bash
# Edit run_batch_segmentation.sh to set your paths
./run_batch_segmentation.sh
```

#### Option B: Direct Python Command
```bash
conda activate grounded-sam2

python batch_video_segmentation_dinox.py \
    --input_video_dir "/home/oshrihalimi/exp/motion_data/prompt4_i2v" \
    --prompts_file "/home/oshrihalimi/video-stack/syn_data/segmentation_prompts4.txt" \
    --output_dir "/home/oshrihalimi/exp/motion_data/prompt4_i2v/segmentation_dinox_sam2" \
    --box_threshold 0.2 \
    --iou_threshold 0.8 \
    --prompt_type box
```

## Output Directory Structure

For each processed video, the script creates:

```
output_dir/
└── video_001/
    ├── frames/                    # Extracted video frames
    │   ├── 00000.jpg
    │   ├── 00001.jpg
    │   └── ...
    ├── tracking_results/          # Annotated frames with segmentation
    │   ├── annotated_frame_00000.jpg
    │   ├── annotated_frame_00001.jpg
    │   └── ...
    ├── video_001_segmented.mp4    # Output video with tracking
    └── metadata.json              # Processing metadata
```

## Arguments

- `--input_video_dir`: Directory containing MP4 videos
- `--prompts_file`: Text file with prompts (one per line)  
- `--output_dir`: Output directory for results
- `--box_threshold`: Detection confidence threshold (default: 0.2)
- `--iou_threshold`: IoU threshold for filtering (default: 0.8)
- `--prompt_type`: SAM2 prompt type - "box", "point", or "mask" (default: "box")
- `--sam2_checkpoint`: Path to SAM2 checkpoint (default: "./checkpoints/sam2.1_hiera_large.pt")
- `--sam2_config`: Path to SAM2 config (default: "configs/sam2.1/sam2.1_hiera_l.yaml")

## Key Features

✅ **Batch Processing**: Process multiple videos automatically  
✅ **Flexible Prompts**: Each video gets its own text prompt for segmentation  
✅ **High Accuracy**: Uses DINO-X cloud API for detection + SAM2 for segmentation  
✅ **Organized Output**: Creates structured output directories  
✅ **Visualization**: Generates both annotated frames and output videos  
✅ **Metadata**: Saves processing information for each video  
✅ **Error Handling**: Continues processing even if individual videos fail  

## Error Troubleshooting

1. **"DDS_API_TOKEN not set"**: Make sure to export your API token
2. **"Number of videos doesn't match prompts"**: Ensure your prompts file has one line per video
3. **"No objects detected"**: Try lowering the box_threshold or refining your text prompt
4. **Memory issues**: Process smaller batches or use a GPU with more memory

## Example Workflow

```bash
# 1. Set up your environment
conda activate grounded-sam2
export DDS_API_TOKEN="your_token_here"

# 2. Create prompts file
cat > my_prompts.txt << EOF
person walking
car driving
dog playing
EOF

# 3. Run batch processing
python batch_video_segmentation_dinox.py \
    --input_video_dir "/path/to/videos" \
    --prompts_file "my_prompts.txt" \
    --output_dir "/path/to/output"

# 4. Check results
ls /path/to/output/*/
```
