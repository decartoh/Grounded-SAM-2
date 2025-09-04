#!/usr/bin/env python3
"""
Batch Video Segmentation with DINO-X and SAM2
Based on: grounded_sam2_tracking_demo_custom_video_input_dinox.py

Process multiple MP4 videos with corresponding text prompts for object segmentation and tracking.
"""

# dds cloudapi for DINO-X - update to V2Task API
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task

import os
import cv2
import torch
import numpy as np
import supervision as sv
import argparse
import base64
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images


def parse_args():
    parser = argparse.ArgumentParser(description="Batch video segmentation with DINO-X and SAM2")
    parser.add_argument("--input_video_dir", type=str, required=True,
                       help="Directory containing MP4 videos to process")
    parser.add_argument("--prompts_file", type=str, required=True,
                       help="Text file with prompts (one per line, corresponding to video index)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for segmentation results")
    parser.add_argument("--box_threshold", type=float, default=0.5,
                       help="Minimum confidence threshold for DINO-X detections")
    parser.add_argument("--iou_threshold", type=float, default=0.8,
                       help="IOU threshold for detection filtering")
    parser.add_argument("--overlap_threshold", type=float, default=0.9,
                       help="IoU threshold for considering bounding boxes as overlapping")
    parser.add_argument("--assignment_algorithm", type=str, default="optimal_sum",
                       choices=["optimal_sum", "greedy_confidence", "single_best"],
                       help="Algorithm for assigning bboxes to prompts")
    parser.add_argument("--prompt_type", type=str, default="box", 
                       choices=["point", "box", "mask"],
                       help="Prompt type for SAM2 video predictor")
    parser.add_argument("--sam2_checkpoint", type=str, 
                       default="./checkpoints/sam2.1_hiera_large.pt",
                       help="Path to SAM2 model checkpoint")
    parser.add_argument("--sam2_config", type=str, 
                       default="configs/sam2.1/sam2.1_hiera_l.yaml",
                       help="Path to SAM2 model config")
    return parser.parse_args()


def load_prompts(prompts_file):
    """Load text prompts from file, one per line."""
    with open(prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def get_video_files(video_dir):
    """Get all MP4 files from directory, sorted by name."""
    video_dir = Path(video_dir)
    video_files = sorted([f for f in video_dir.glob("*.mp4")])
    return video_files


def initialize_sam2_models(sam2_checkpoint, model_cfg):
    """Initialize SAM2 models with optimizations."""
    # use bfloat16 for the entire processing
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # init sam image predictor and video predictor model
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)
    
    return video_predictor, image_predictor


def extract_video_frames(video_path, temp_frames_dir):
    """Extract frames from video and save as temporary JPEG files for SAM2 processing."""
    temp_frames_dir = Path(temp_frames_dir)
    temp_frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video info and frame generator
    video_info = sv.VideoInfo.from_video_path(str(video_path))
    frame_generator = sv.get_video_frames_generator(str(video_path))
    
    print(f"Video info: {video_info}")
    
    # Save video frames using OpenCV (temporary files for SAM2)
    for frame_idx, frame in tqdm(enumerate(frame_generator), 
                                desc=f"Extracting frames from {video_path.name}",
                                total=video_info.total_frames):
        frame_path = temp_frames_dir / f"{frame_idx:05d}.jpg"
        cv2.imwrite(str(frame_path), frame)
    
    return video_info


from prompt_bbox_assignment import get_assigner

def get_detection_boxes(img_path, text_prompt, api_token, box_threshold, iou_threshold, 
                       overlap_threshold=0.5, assignment_algorithm="optimal_sum"):
    """Use DINO-X cloud API to get detection boxes for objects.
    
    Uses modular assignment algorithms to map prompts to bounding boxes.
    """
    # Split the prompt by comma to get individual object prompts
    individual_prompts = [prompt.strip() for prompt in text_prompt.split(',') if prompt.strip()]
    
    if len(individual_prompts) == 0:
        return np.array([]), [], []
    
    print(f"Detecting objects for prompts: {individual_prompts}")
    print(f"Detection threshold: {box_threshold}, Assignment algorithm: {assignment_algorithm}")
    
    # Initialize API client
    config = Config(api_token)
    client = Client(config)
    
    # Convert image to base64 for V2 API
    with open(img_path, 'rb') as f:
        image_data = f.read()
        image_b64 = base64.b64encode(image_data).decode('utf-8')
    image_url = f"data:image/jpeg;base64,{image_b64}"
    
    # Step 1: Collect ALL detections for each prompt
    prompt_detections = {}  # {prompt: [(bbox, confidence), ...]}
    
    for prompt in individual_prompts:
        print(f"  - Running detection for: '{prompt}'")
        
        # Create and run detection task for this specific prompt
        task = V2Task(
            api_path="/v2/task/dinox/detection",
            api_body={
                "model": "DINO-X-1.0",
                "image": image_url,
                "prompt": {
                    "type": "text",
                    "text": prompt
                },
                "targets": ["bbox"],
                "bbox_threshold": box_threshold,
                "iou_threshold": iou_threshold,
            }
        )
        
        client.run_task(task)
        result = task.result
        
        objects = result["objects"]
        
        if objects:
            # Sort detections by confidence (descending)
            detections = [(obj["bbox"], obj["score"]) for obj in objects]
            detections.sort(key=lambda x: x[1], reverse=True)
            prompt_detections[prompt] = detections
            print(f"    Found {len(detections)} detections above threshold {box_threshold}")
        else:
            prompt_detections[prompt] = []
            print(f"    Found 0 instances of '{prompt}'")
    
    # Step 2: Use assignment algorithm to map prompts to bboxes
    assigner = get_assigner(assignment_algorithm)
    final_boxes, final_confidences, final_class_names = assigner.assign(
        prompt_detections, overlap_threshold
    )
    
    return np.array(final_boxes), final_confidences, final_class_names


def process_single_video(video_path, text_prompt, output_base_dir, video_predictor, 
                        image_predictor, args):
    """Process a single video with its corresponding text prompt."""
    
    video_name = video_path.stem
    print(f"\n=== Processing video: {video_name} with prompt: '{text_prompt}' ===")
    
    # Create output directory for this video (minimal structure)
    video_output_dir = Path(output_base_dir) / video_name
    temp_frames_dir = video_output_dir / "temp_frames"
    
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Extract video frames (temporary)
        video_info = extract_video_frames(video_path, temp_frames_dir)
        
        # Get frame names
        frame_names = [
            p for p in os.listdir(temp_frames_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        
        if not frame_names:
            print(f"No frames extracted for {video_name}")
            return False
        
        # Step 2: Initialize video predictor state
        inference_state = video_predictor.init_state(video_path=str(temp_frames_dir))
        ann_frame_idx = 0  # Use first frame for annotation
        
        # Step 3: Get detection boxes using DINO-X
        img_path = temp_frames_dir / frame_names[ann_frame_idx]
        image = Image.open(img_path)
        
        api_token = os.getenv('DDS_API_TOKEN')
        if not api_token:
            raise ValueError("DDS_API_TOKEN environment variable not set")
        
        input_boxes, confidences, class_names = get_detection_boxes(
            str(img_path), text_prompt, api_token, args.box_threshold, args.iou_threshold,
            args.overlap_threshold, args.assignment_algorithm
        )
        
        if len(input_boxes) == 0:
            print(f"No objects detected for {video_name} with prompt '{text_prompt}'")
            return False
        
        print(f"Detected {len(input_boxes)} objects: {class_names}")
        
        # Step 4: Get masks using SAM2 image predictor
        image_predictor.set_image(np.array(image.convert("RGB")))
        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        
        # Convert mask shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        # Step 5: Register objects to video predictor
        if args.prompt_type == "point":
            # Sample positive points from masks
            all_sample_points = sample_points_from_masks(masks=masks, num_points=10)
            for object_id, (label, points) in enumerate(zip(class_names, all_sample_points), start=1):
                labels = np.ones((points.shape[0]), dtype=np.int32)
                video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                )
        elif args.prompt_type == "box":
            for object_id, (label, box) in enumerate(zip(class_names, input_boxes), start=1):
                video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    box=box,
                )
        elif args.prompt_type == "mask":
            for object_id, (label, mask) in enumerate(zip(class_names, masks), start=1):
                video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    mask=mask
                )
        
        # Step 6: Propagate segmentation across video
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in tqdm(
            video_predictor.propagate_in_video(inference_state),
            desc=f"Tracking objects in {video_name}",
            total=len(frame_names)
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        # Step 7: Create visualizations in memory and output video
        ID_TO_OBJECTS = {i: obj for i, obj in enumerate(class_names, start=1)}
        
        # Create video writer for direct output
        output_video_path = video_output_dir / f"{video_name}_segmented.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = video_info.fps
        frame_size = (video_info.width, video_info.height)
        
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, frame_size)
        
        for frame_idx, segments in tqdm(video_segments.items(), 
                                       desc=f"Creating segmented video for {video_name}"):
            img = cv2.imread(str(temp_frames_dir / frame_names[frame_idx]))
            
            if len(segments) > 0:
                object_ids = list(segments.keys())
                masks = list(segments.values())
                masks = np.concatenate(masks, axis=0)
                
                detections = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks),
                    mask=masks,
                    class_id=np.array(object_ids, dtype=np.int32),
                )
                
                # Annotate with boxes and labels
                box_annotator = sv.BoxAnnotator()
                annotated_frame = box_annotator.annotate(
                    scene=img.copy(), 
                    detections=detections, 
                    labels=[ID_TO_OBJECTS[i] for i in object_ids]
                )
                
                # Annotate with masks
                mask_annotator = sv.MaskAnnotator()
                annotated_frame = mask_annotator.annotate(
                    scene=annotated_frame, 
                    detections=detections
                )
            else:
                annotated_frame = img.copy()
            
            # Write frame directly to output video (no individual frame saving)
            out.write(annotated_frame)
        
        out.release()
        
        # Step 8: Clean up temporary frames
        import shutil
        shutil.rmtree(temp_frames_dir)
        print(f"🧹 Cleaned up temporary frames")
        
        # Step 9: Save metadata
        metadata = {
            "video_name": video_name,
            "text_prompt": text_prompt,
            "detected_objects": class_names,
            "num_objects": len(class_names),
            "num_frames": len(frame_names),
            "video_info": {
                "fps": video_info.fps,
                "total_frames": video_info.total_frames,
                "resolution": f"{video_info.width}x{video_info.height}"
            }
        }
        
        import json
        with open(video_output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Successfully processed {video_name}")
        print(f"   - Objects detected: {class_names}")
        print(f"   - Output video: {output_video_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {video_name}: {str(e)}")
        return False


def main():
    args = parse_args()
    
    print("=== Batch Video Segmentation with DINO-X and SAM2 ===")
    print(f"Input video directory: {args.input_video_dir}")
    print(f"Prompts file: {args.prompts_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Box threshold: {args.box_threshold}")
    print(f"IOU threshold: {args.iou_threshold}")
    print(f"Prompt type: {args.prompt_type}")
    
    # Check API token
    api_token = os.getenv('DDS_API_TOKEN')
    if not api_token:
        print("❌ Error: DDS_API_TOKEN environment variable not set")
        print("Please set your API token: export DDS_API_TOKEN='your_token_here'")
        return
    
    # Load prompts
    try:
        prompts = load_prompts(args.prompts_file)
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    except Exception as e:
        print(f"❌ Error loading prompts: {e}")
        return
    
    # Get video files
    try:
        video_files = get_video_files(args.input_video_dir)
        print(f"Found {len(video_files)} video files in {args.input_video_dir}")
    except Exception as e:
        print(f"❌ Error loading video files: {e}")
        return
    
    # Check if number of videos matches number of prompts
    if len(video_files) != len(prompts):
        print(f"❌ Error: Number of videos ({len(video_files)}) doesn't match number of prompts ({len(prompts)})")
        print("Each video needs a corresponding prompt in the prompts file.")
        return
    
    # Initialize SAM2 models
    try:
        print("Initializing SAM2 models...")
        video_predictor, image_predictor = initialize_sam2_models(
            args.sam2_checkpoint, args.sam2_config
        )
        print("✅ SAM2 models initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing SAM2 models: {e}")
        return
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process videos
    successful_videos = 0
    failed_videos = 0
    
    for i, (video_file, prompt) in enumerate(zip(video_files, prompts)):
        print(f"\n--- Processing video {i+1}/{len(video_files)} ---")
        
        success = process_single_video(
            video_file, prompt, args.output_dir, 
            video_predictor, image_predictor, args
        )
        
        if success:
            successful_videos += 1
        else:
            failed_videos += 1
    
    # Final summary
    print(f"\n=== Processing Complete ===")
    print(f"✅ Successfully processed: {successful_videos} videos")
    print(f"❌ Failed to process: {failed_videos} videos")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
