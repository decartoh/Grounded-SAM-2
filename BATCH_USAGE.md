# Batch Processing Guide

**🚀 Process images and videos locally with text prompts using HuggingFace GroundingDINO + SAM2**

## Overview

This guide covers batch processing capabilities that run completely locally without API tokens or quotas:
- **Video Processing**: Full video segmentation with object tracking
- **Image Processing**: Single-frame segmentation with automatic prompt detection  
- **Mixed Processing**: Handle both images and videos in the same directory

## Quick Start

### Mixed Image/Video Processing (NEW!)
```bash
conda activate grounded-sam2

# Process both images and videos in the same directory
python batch_mixed_segmentation.py \
    --input_dir "/path/to/mixed/media" \
    --output_dir "/path/to/results" \
    --box_threshold 0.05 \
    --text_threshold 0.05 \
    --grounding_model "IDEA-Research/grounding-dino-base"
```

### Video-Only Processing
```bash
conda activate grounded-sam2

python batch_video_segmentation_hf.py \
    --input_video_dir "/home/oshrihalimi/exp/motion_data/prompt4_i2v" \
    --prompts_file "/home/oshrihalimi/video-stack/syn_data/segmentation_prompts4.txt" \
    --output_dir "/home/oshrihalimi/exp/motion_data/prompt4_i2v/segmentation_hf_sam2" \
    --grounding_model "IDEA-Research/grounding-dino-base" \
    --box_threshold 0.6 \
    --text_threshold 0.5 \
    --overlap_threshold 0.8 \
    --prompt_type box

# Alternative example with different parameters
python batch_video_segmentation_hf.py \
    --input_video_dir "/home/oshrihalimi/exp/motion_data/prompts11_F161_G10" \
    --prompts_file "/home/oshrihalimi/video-stack/syn_data/segmentation_prompts11.txt" \
    --output_dir "/home/oshrihalimi/exp/motion_data/prompts11_F161_G10/segmentation_hf_sam2" \
    --grounding_model "IDEA-Research/grounding-dino-base" \
    --box_threshold 0.6 \
    --text_threshold 0.2 \
    --overlap_threshold 0.2 \
    --prompt_type box \
    --com 

# With center of mass analysis
python batch_video_segmentation_hf.py \
    --input_video_dir "/path/to/videos" \
    --prompts_file "/path/to/prompts.txt" \
    --output_dir "/path/to/output" \
    --com
```

## Mixed Image/Video Processing

The new `batch_mixed_segmentation.py` script automatically processes both images and videos in the same directory:

### Key Features
- **🖼️ Auto Image Detection**: Finds PNG/JPG files automatically
- **📝 Smart Prompts**: Reads prompts from `*_seg.txt` files (e.g., `image_001_seg.txt`)
- **🎬 Video Support**: Processes MP4 files with full tracking pipeline
- **⚡ Unified Output**: Images get `_seg` suffix, videos get `_segmented` suffix

### Directory Structure
```
input_directory/
├── image_001.png           # Image file
├── image_001_seg.txt       # Contains: "raccoon"
├── image_002.jpg           # Another image  
├── image_002_seg.txt       # Contains: "alligator"
├── video_001.mp4           # Video file (optional)
└── ...
```

### Command Line Interface
```bash
python batch_mixed_segmentation.py \
    --input_dir <path>                    # [REQUIRED] Directory with images/videos
    --output_dir <path>                   # [REQUIRED] Output directory
    --grounding_model <model_id>          # [OPTIONAL] GroundingDINO model (default: tiny)
    --box_threshold <float>               # [OPTIONAL] Detection threshold (default: 0.5)
    --text_threshold <float>              # [OPTIONAL] Text threshold (default: 0.3)
    --overlap_threshold <float>           # [OPTIONAL] IoU threshold (default: 0.9)
    --prompt_type <type>                  # [OPTIONAL] box|point|mask (default: box)
```

### Output Structure
```
output_directory/
├── image_001_seg.png       # Segmented image with masks/boxes
├── image_001_metadata.json # Detection info and parameters
├── image_002_seg.jpg       # Segmented image
├── image_002_metadata.json # Detection metadata  
├── video_001_segmented.mp4 # Segmented video (if videos present)
├── video_001_metadata.json # Video processing metadata
└── ...
```

### Example Usage
```bash
# High-quality processing
python batch_mixed_segmentation.py \
    --input_dir "/tmp/my_dataset" \
    --output_dir "/tmp/results" \
    --grounding_model "IDEA-Research/grounding-dino-base" \
    --box_threshold 0.05 \
    --text_threshold 0.05

# Fast processing  
python batch_mixed_segmentation.py \
    --input_dir "/tmp/my_dataset" \
    --output_dir "/tmp/results" \
    --grounding_model "IDEA-Research/grounding-dino-tiny" \
    --box_threshold 0.3 \
    --text_threshold 0.2
```

## Data Preparation

### 1. Video Directory Structure
```
your_videos/
├── video_001.mp4
├── video_002.mp4  
├── video_003.mp4
└── ...
```

### 2. Prompts File Format
Create a text file with one line per video (each line can contain multiple comma-separated objects):

```text
koala, parrot
hedgehog running, mouse
person walking, car driving
bird flying in the sky
dog playing in park
```

## Full Interface Reference

### Complete Command Line Interface
```bash
python batch_video_segmentation_hf.py \
    --input_video_dir <path>              # [REQUIRED] Directory containing MP4 videos to process
    --prompts_file <path>                 # [REQUIRED] Text file with prompts (one per line, corresponding to video index)
    --output_dir <path>                   # [REQUIRED] Output directory for segmentation results
    --grounding_model <model_id>          # [OPTIONAL] HuggingFace model ID for GroundingDINO (default: "IDEA-Research/grounding-dino-tiny")
    --box_threshold <float>               # [OPTIONAL] Minimum confidence threshold for GroundingDINO detections (default: 0.5)
    --text_threshold <float>              # [OPTIONAL] Text threshold for GroundingDINO (default: 0.3)
    --overlap_threshold <float>           # [OPTIONAL] IoU threshold for considering bounding boxes as overlapping (default: 0.9)
    --prompt_type <type>                  # [OPTIONAL] Prompt type for SAM2 video predictor (choices: point, box, mask; default: box)
    --sam2_checkpoint <path>              # [OPTIONAL] Path to SAM2 model checkpoint (default: "./checkpoints/sam2.1_hiera_large.pt")
    --sam2_config <path>                  # [OPTIONAL] Path to SAM2 model config (default: "configs/sam2.1/sam2.1_hiera_l.yaml")
    --com                                 # [OPTIONAL] Calculate and visualize center of mass for each segmented object (default: False)
```

### Quick Reference Flags
```bash
# Required
--input_video_dir     # Directory containing MP4 videos
--prompts_file        # Text file: one line per video, objects can be comma-separated
--output_dir          # Output directory for results

# Detection & Assignment (Optional)
--grounding_model     # Default: "IDEA-Research/grounding-dino-tiny"
--box_threshold       # Default: 0.5 (float 0.0-1.0)
--text_threshold      # Default: 0.3 (float 0.0-1.0)  
--overlap_threshold   # Default: 0.9 (float 0.0-1.0)

# SAM2 Segmentation (Optional)
--prompt_type         # Default: "box" (choices: box|point|mask)
--sam2_checkpoint     # Default: "./checkpoints/sam2.1_hiera_large.pt"
--sam2_config         # Default: "configs/sam2.1/sam2.1_hiera_l.yaml"

# Analysis Features (Optional)
--com                 # Enable center of mass calculation and visualization
```

### Required Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `--input_video_dir` | `str` | Directory with MP4 videos (processed alphabetically) |
| `--prompts_file` | `str` | Text file: one line per video, comma-separated objects per line |
| `--output_dir` | `str` | Output directory for segmented videos and metadata |

### 🔧 Algorithmic Parameters 

#### `--box_threshold` (default: `0.5`)
- **Range**: 0.0-1.0 (float)
- **Controls**: Minimum confidence for object detection 
- **Algorithm impact**: Higher → fewer but more confident detections; Lower → more detections including uncertain ones
- **Tuning**: Use `0.3` for sensitive detection, `0.7` for strict filtering

#### `--text_threshold` (default: `0.3`) 
- **Range**: 0.0-1.0 (float)
- **Controls**: Text-image similarity threshold
- **Algorithm impact**: Higher → stricter prompt matching; Lower → more flexible interpretation
- **Note**: Independent from `box_threshold` - both must be satisfied

#### `--overlap_threshold` (default: `0.9`)
- **Range**: 0.0-1.0 (float) 
- **Controls**: Maximum IoU between boxes assigned to different prompts
- **Algorithm**: Used by OptimalSumAssigner for non-overlapping assignment
- **Impact**: Higher → allows closer objects; Lower → forces separation
- **Purpose**: Prevents multiple prompts pointing to same object

### Model Parameters

#### `--grounding_model` (default: `"grounding-dino-tiny"`)
- **Options**: `"IDEA-Research/grounding-dino-tiny"` (fast), `"IDEA-Research/grounding-dino-base"` (accurate)
- **Impact**: Speed vs accuracy tradeoff, auto-downloads from HuggingFace

#### `--prompt_type` (default: `"box"`)  
- **Options**: 
  - `"box"`: Fastest, uses detection boxes directly
  - `"point"`: Moderate speed, samples interior points, more stable
  - `"mask"`: Slowest, highest precision, generates initial mask
- **Impact**: Speed vs segmentation quality tradeoff

#### `--sam2_checkpoint` & `--sam2_config`
- **Defaults**: `"./checkpoints/sam2.1_hiera_large.pt"` + `"configs/sam2.1/sam2.1_hiera_l.yaml"`
- **Purpose**: SAM2 model files (must match each other)
- **Note**: Download via `download_ckpts.sh`

### Analysis Features

#### `--com` (default: `False`)
- **Type**: Flag (no argument needed)
- **Purpose**: Calculate and visualize center of mass (centroid) for each segmented object
- **Outputs**:
  - **Video**: COM markers overlaid on segmentation video (different colors per object)
  - **Data**: JSON file with per-frame COM coordinates (`*_com_data.json`)
  - **Trajectory**: Static image showing COM path with white→black color gradient (`*_object_name_com.png`)
- **Impact**: Adds ~5-10% processing time, useful for motion analysis and object tracking

## Algorithm Flow & Parameter Impact

1. **Detection**: `grounding_model` finds objects → `box_threshold` filters confidence → `text_threshold` filters similarity
2. **Assignment**: OptimalSumAssigner assigns prompts to objects using `overlap_threshold` to prevent conflicts  
3. **Segmentation**: SAM2 generates masks using `prompt_type` (box/point/mask)

### Quick Tuning Guide
- **More objects**: Lower `box_threshold` (0.3) + `text_threshold` (0.2)  
- **Fewer false positives**: Higher `box_threshold` (0.7) + `text_threshold` (0.5)
- **Crowded scenes**: Lower `overlap_threshold` (0.7), use `point`/`mask` prompts
- **Speed priority**: `grounding-dino-tiny` + `prompt_type=box`
- **Quality priority**: `grounding-dino-base` + `prompt_type=mask`

## Example Commands

### Basic Usage
```bash
python batch_video_segmentation_hf.py \
    --input_video_dir "./videos" \
    --prompts_file "./prompts.txt" \
    --output_dir "./results"
```

### High Quality 
```bash
python batch_video_segmentation_hf.py \
    --input_video_dir "./videos" \
    --prompts_file "./prompts.txt" \
    --output_dir "./results" \
    --grounding_model "IDEA-Research/grounding-dino-base" \
    --prompt_type mask
```

### Sensitive Detection
```bash
python batch_video_segmentation_hf.py \
    --input_video_dir "./videos" \
    --prompts_file "./prompts.txt" \
    --output_dir "./results" \
    --box_threshold 0.3 \
    --text_threshold 0.2 \
    --overlap_threshold 0.7
```

## Output Structure

### Directory Layout
```
output_directory/
├── video_001_segmented.mp4     # Segmented video with overlays
├── video_001_metadata.json     # Detection and processing info
├── video_001_com_data.json     # Center of mass data (if --com enabled)
├── video_001_koala_com.png     # COM trajectory image for koala (if --com enabled)
├── video_001_parrot_com.png    # COM trajectory image for parrot (if --com enabled)
├── video_002_segmented.mp4     
├── video_002_metadata.json
└── ...
```

### Metadata JSON Format
```json
{
  "video_path": "/path/to/input/video_001.mp4",
  "prompts": "koala, parrot",
  "detections": [
    {
      "bbox": [100, 150, 200, 250],
      "label": "koala"
    },
    {
      "bbox": [300, 100, 400, 200], 
      "label": "parrot"
    }
  ],
  "video_info": {
    "fps": 30,
    "width": 1280,
    "height": 720,
    "frame_count": 150
  },
  "parameters": {
    "box_threshold": 0.5,
    "text_threshold": 0.3,
    "prompt_type": "box",
    "center_of_mass_enabled": true
  },
  "center_of_mass": {
    "com_data_file": "video_001_com_data.json",
    "trajectory_images": ["video_001_koala_com.png", "video_001_parrot_com.png"]
  }
}
```

### Center of Mass Data Format (when --com enabled)
```json
{
  "koala": [
    {"frame": 0, "x": 150.5, "y": 200.3},
    {"frame": 1, "x": 152.1, "y": 198.7},
    {"frame": 2, "x": null, "y": null},
    {"frame": 3, "x": 155.8, "y": 195.2}
  ],
  "parrot": [
    {"frame": 0, "x": 350.2, "y": 150.1},
    {"frame": 1, "x": 348.9, "y": 148.5},
    {"frame": 2, "x": 347.1, "y": 147.2},
    {"frame": 3, "x": null, "y": null}
  ]
}
```
**Note**: `null` values indicate frames where the object was not detected or segmented.

## Advanced Features

### 🧠 Smart Object Assignment Algorithm

The system uses **OptimalSumAssigner** for intelligent mapping:

1. **Detection**: Find all objects for each prompt in comma-separated list
2. **Assignment Options**: Generate all possible 1:1 prompt-to-bbox mappings  
3. **Overlap Filtering**: Eliminate assignments where boxes overlap above threshold
4. **Optimization**: Select assignment that maximizes total confidence score

**Example:**
```text
Input: "koala, parrot"
Found: 2 koalas (conf: 0.8, 0.6), 2 parrots (conf: 0.9, 0.7)
Algorithm: Tests all combinations, picks best non-overlapping pair
Result: koala@0.8 + parrot@0.9 = total confidence 1.7 (optimal)
```

### 📊 Processing Pipeline

1. **📁 Load Videos**: Scan directory for MP4 files
2. **📝 Load Prompts**: Read prompt file (one per video)
3. **🔍 Object Detection**: HuggingFace GroundingDINO finds objects
4. **🎯 Smart Assignment**: Optimal bbox-to-prompt mapping  
5. **🎭 Segmentation**: SAM2 generates precise masks
6. **📹 Tracking**: Follow objects across video frames
7. **💾 Save Results**: Output videos + metadata

## Troubleshooting

### Common Issues

**❌ "Found 0 video files"**
```bash
# Solution: Use absolute paths
--input_video_dir "/full/path/to/videos"  # not ~/videos
```

**❌ "No detections found"**
```bash
# Solution: Lower detection thresholds
--box_threshold 0.3 --text_threshold 0.2
```

**❌ CUDA out of memory**
```bash
# Solution: Use smaller model
--grounding_model "IDEA-Research/grounding-dino-tiny"
```

**❌ "Mismatch between videos and prompts"**
```bash
# Solution: Check prompt file has one line per video
wc -l prompts.txt
ls videos/*.mp4 | wc -l
```

### Performance Optimization

**For Large Datasets:**
- Process videos in smaller batches
- Use `grounding-dino-tiny` for speed
- Set `--prompt_type box` for fastest processing

**For Best Quality:**
- Use `grounding-dino-base` model
- Set higher thresholds: `--box_threshold 0.6 --text_threshold 0.4`
- Use `--prompt_type mask` for precision

**Memory Management:**
- Close other GPU applications
- Process videos sequentially (built-in)
- Use smaller SAM2 models if available

## Shell Script Usage

Create `run_batch_hf.sh`:
```bash
#!/bin/bash

python batch_video_segmentation_hf.py \
    --input_video_dir "/home/user/videos" \
    --prompts_file "/home/user/prompts.txt" \
    --output_dir "/home/user/output" \
    --grounding_model "IDEA-Research/grounding-dino-base" \
    --box_threshold 0.5 \
    --text_threshold 0.3 \
    --overlap_threshold 0.9 \
    --prompt_type box
```

```bash
chmod +x run_batch_hf.sh
./run_batch_hf.sh
```

## Key Advantages

✅ **No API Tokens**: Runs completely locally  
✅ **No Quotas**: Unlimited processing  
✅ **High Quality**: Same results as cloud APIs  
✅ **Flexible**: Full parameter control  
✅ **Reliable**: No network dependencies after model download  
✅ **Privacy**: Your videos never leave your machine  
✅ **Cost Effective**: Free to run (after initial setup)  
✅ **Scalable**: Process thousands of videos  

---

**Ready to start? Copy one of the example commands above and modify the paths for your data!**