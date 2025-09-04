# HuggingFace Batch Video Segmentation Guide

**🚀 Process multiple videos locally with text prompts using HuggingFace GroundingDINO + SAM2**

## Overview

This guide covers the **HuggingFace version** of batch video segmentation that runs completely locally without API tokens or quotas.

## Quick Start
```
conda activate grounded-sam2

python batch_video_segmentation_hf.py     \
--input_video_dir "/home/oshrihalimi/exp/motion_data/prompt4_i2v"     \
--prompts_file "/home/oshrihalimi/video-stack/syn_data/segmentation_prompts4.txt"     \
--output_dir "/home/oshrihalimi/exp/motion_data/prompt4_i2v/segmentation_hf_sam2"    \
 --grounding_model "IDEA-Research/grounding-dino-base"     \
 --box_threshold 0.6     \
 --text_threshold 0.5    \
 --overlap_threshold 0.8 \
--prompt_type box
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
    "prompt_type": "box"
  }
}
```

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