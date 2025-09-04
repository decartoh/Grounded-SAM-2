#!/usr/bin/env python3
"""
Batch Video Segmentation with HuggingFace GroundingDINO and SAM2
Based on: batch_video_segmentation_dinox.py, but using HuggingFace transformers instead of cloud API

Process multiple MP4 videos with corresponding text prompts for object segmentation and tracking.
"""

import os
import cv2
import torch
import numpy as np
import supervision as sv
import argparse
import json
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from prompt_bbox_assignment import OptimalSumAssigner


def parse_args():
    parser = argparse.ArgumentParser(description="Batch video segmentation with HuggingFace GroundingDINO and SAM2")
    parser.add_argument("--input_video_dir", type=str, required=True,
                       help="Directory containing MP4 videos to process")
    parser.add_argument("--prompts_file", type=str, required=True,
                       help="Text file with prompts (one per line, corresponding to video index)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for segmentation results")
    parser.add_argument("--grounding_model", type=str, default="IDEA-Research/grounding-dino-tiny",
                       help="HuggingFace model ID for GroundingDINO")
    parser.add_argument("--box_threshold", type=float, default=0.5,
                       help="Minimum confidence threshold for GroundingDINO detections")
    parser.add_argument("--text_threshold", type=float, default=0.3,
                       help="Text threshold for GroundingDINO")
    parser.add_argument("--overlap_threshold", type=float, default=0.9,
                       help="IoU threshold for considering bounding boxes as overlapping")
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


def initialize_models(grounding_model, sam2_checkpoint, model_cfg):
    """Initialize HuggingFace GroundingDINO and SAM2 models with optimizations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # use bfloat16 for the entire processing
    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Initialize HuggingFace GroundingDINO
    print(f"Loading GroundingDINO model: {grounding_model}")
    processor = AutoProcessor.from_pretrained(grounding_model)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model).to(device)

    # init sam image predictor and video predictor model
    print(f"Loading SAM2 models...")
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)
    
    return processor, grounding_model, video_predictor, image_predictor, device


def extract_video_frames(video_path, temp_frames_dir):
    """Extract frames from video and save as temporary JPEG files for SAM2 processing."""
    temp_frames_dir = Path(temp_frames_dir)
    temp_frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video info and frame generator
    video_info = sv.VideoInfo.from_video_path(str(video_path))
    frame_generator = sv.get_video_frames_generator(str(video_path))
    
    print(f"Video info: {video_info}")
    
    # Save frames as JPEG files for SAM2
    frame_count = 0
    for frame in tqdm(frame_generator, desc="Extracting video frames"):
        frame_path = temp_frames_dir / f"{frame_count:05d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_count += 1
    
    print(f"Extracted {frame_count} frames to {temp_frames_dir}")
    return video_info, frame_count


def get_detection_boxes(processor, grounding_model, device, image, prompts_text, 
                       box_threshold, text_threshold, overlap_threshold):
    """
    Get object detection bounding boxes using HuggingFace GroundingDINO.
    
    Args:
        processor: HuggingFace processor
        grounding_model: HuggingFace GroundingDINO model
        device: Computing device
        image: PIL Image
        prompts_text: Comma-separated text prompts (e.g., "koala,parrot")
        box_threshold: Minimum confidence for detections
        text_threshold: Text confidence threshold
        overlap_threshold: IoU threshold for overlap resolution
        assignment_algorithm: Algorithm for bbox-to-prompt assignment
    
    Returns:
        List of (bbox, label) tuples for final assigned detections
    """
    
    # Split prompts by comma and clean whitespace
    individual_prompts = [prompt.strip() for prompt in prompts_text.split(',')]
    print(f"Processing {len(individual_prompts)} prompts: {individual_prompts}")
    
    # Collect detections organized by prompt for the assigner
    prompt_detections = {}
    
    for prompt in individual_prompts:
        # Format prompt (lowercase + dot)
        formatted_prompt = prompt.lower()
        if not formatted_prompt.endswith('.'):
            formatted_prompt += '.'
        
        print(f"  Running detection for: '{formatted_prompt}'")
        
        # Run HuggingFace GroundingDINO detection
        inputs = processor(images=image, text=formatted_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = grounding_model(**inputs)
        
        # Post-process results
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]  # [height, width]
        )
        
        # Extract bounding boxes and scores
        boxes = results[0]["boxes"].cpu().numpy()  # xyxy format
        scores = results[0]["scores"].cpu().numpy()
        
        # Sort detections by confidence (descending) for display and processing
        detection_scores = sorted(scores, reverse=True)
        
        print(f"    Found {len(boxes)} detections for '{prompt}' (threshold: {box_threshold})")
        if len(detection_scores) > 0:
            confidence_str = ', '.join([f"{score:.3f}" for score in detection_scores])
            print(f"    Confidences (descending): [{confidence_str}]")
        
        # Store detections for this prompt (sorted by confidence desc)
        detections_for_prompt = []
        for box, score in zip(boxes, scores):
            # Convert xyxy to xywh format expected by assigner
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            box_xywh = [x1, y1, w, h]
            detections_for_prompt.append((box_xywh, score))
        
        # Sort by confidence (descending)
        detections_for_prompt.sort(key=lambda x: x[1], reverse=True)
        prompt_detections[prompt] = detections_for_prompt
    
    if not any(prompt_detections.values()):
        print("No detections found for any prompt!")
        return []
    
    # Use optimal assignment algorithm to resolve overlaps and assign boxes to prompts
    assigner = OptimalSumAssigner()
    final_boxes, final_confidences, final_labels = assigner.assign(prompt_detections, overlap_threshold)
    
    print(f"Final assignments: {len(final_boxes)} boxes assigned to prompts")
    for label, confidence in zip(final_labels, final_confidences):
        print(f"  {label}: confidence={confidence:.3f}")
    
    # Convert xywh back to xyxy format expected by the rest of the code
    result_boxes = []
    for box_xywh in final_boxes:
        x, y, w, h = box_xywh
        box_xyxy = [x, y, x + w, y + h]
        result_boxes.append(np.array(box_xyxy))
    
    return list(zip(result_boxes, final_labels))


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two bounding boxes in xyxy format."""
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection_width = max(0, x2_min - x1_max)
    intersection_height = max(0, y2_min - y1_max)
    intersection_area = intersection_width * intersection_height
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - intersection_area
    
    # Calculate IoU
    if union_area == 0:
        return 0
    return intersection_area / union_area


def process_single_video(video_path, prompts_text, processor, grounding_model, 
                        video_predictor, image_predictor, device, output_base_dir, 
                        box_threshold, text_threshold, overlap_threshold, prompt_type):
    """Process a single video with given prompts."""
    
    video_name = video_path.stem
    print(f"\n🎬 Processing video: {video_name}")
    print(f"   Prompts: {prompts_text}")
    
    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_frames_dir:
        temp_frames_dir = Path(temp_frames_dir)
        
        # Step 1: Extract video frames
        video_info, frame_count = extract_video_frames(video_path, temp_frames_dir)
        
        # Step 2: Get first frame for object detection
        first_frame_path = temp_frames_dir / "00000.jpg"
        if not first_frame_path.exists():
            print(f"❌ No frames extracted from {video_path}")
            return
        
        # Load first frame
        first_frame_pil = Image.open(first_frame_path)
        first_frame_np = np.array(first_frame_pil)
        
        # Step 3: Run object detection on first frame
        print("🔍 Running object detection on first frame...")
        detection_results = get_detection_boxes(
            processor, grounding_model, device, first_frame_pil, prompts_text,
            box_threshold, text_threshold, overlap_threshold
        )
        
        if not detection_results:
            print("❌ No valid detections found, skipping video")
            return
        
        print(f"✅ Found {len(detection_results)} valid detections")
        
        # Step 4: Initialize SAM2 video predictor
        inference_state = video_predictor.init_state(video_path=str(temp_frames_dir))
        
        # Step 5: Set up tracking based on detections
        sam2_predictor = image_predictor
        sam2_predictor.set_image(first_frame_np)
        
        # Convert detection boxes and get initial masks
        input_boxes = []
        object_ids = []
        labels = []
        
        for idx, (bbox, label) in enumerate(detection_results):
            input_boxes.append(bbox)
            object_ids.append(idx)
            labels.append(label)
        
        input_boxes = np.array(input_boxes)
        
        # Get masks for the detected objects
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        
        # Add objects to video predictor
        if prompt_type == "box":
            prompts = input_boxes
        elif prompt_type == "point":
            # Sample points from masks
            prompts = sample_points_from_masks(masks)
        else:  # mask
            prompts = masks
        
        for i, (label, prompt) in enumerate(zip(labels, prompts)):
            if prompt_type == "box":
                _, _, _ = video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=i,
                    box=prompt,
                )
            elif prompt_type == "point":
                _, _, _ = video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=i,
                    points=prompt,
                    labels=np.array([1] * len(prompt)),
                )
            else:  # mask
                video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=i,
                    mask=prompt[0]
                )
        
        print(f"🎯 Added {len(labels)} objects for tracking")
        
        # Step 6: Propagate masks through video
        print("🚀 Propagating masks through video...")
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        # Step 7: Create output video with annotations
        print("🎨 Creating annotated output video...")
        
        # Create output video with OpenCV
        output_video_path = Path(output_base_dir) / f"{video_name}_segmented.mp4"
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(str(output_video_path), fourcc, video_info.fps, 
                                   (video_info.width, video_info.height))
        
        # Annotation setup
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        
        ID_TO_OBJECTS = {i: label for i, label in enumerate(labels)}
        
        # Process each frame
        frame_paths = sorted(temp_frames_dir.glob("*.jpg"))
        for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc="Processing frames")):
            # Load frame
            frame = cv2.imread(str(frame_path))
            
            # Get masks for this frame
            if frame_idx in video_segments:
                masks_data = video_segments[frame_idx]
                
                # Convert to supervision format
                mask_list = []
                box_list = []
                class_ids = []
                confidences = []
                
                for obj_id in sorted(masks_data.keys()):
                    mask = masks_data[obj_id]
                    
                    # Handle potential 3D mask (squeeze to 2D)
                    if mask.ndim > 2:
                        mask = mask.squeeze()
                    
                    if mask.any():  # Only process non-empty masks
                        mask_list.append(mask)
                        
                        # Get bounding box from mask
                        y_indices, x_indices = np.where(mask)
                        if len(x_indices) > 0 and len(y_indices) > 0:
                            x_min, x_max = x_indices.min(), x_indices.max()
                            y_min, y_max = y_indices.min(), y_indices.max()
                            box_list.append([x_min, y_min, x_max, y_max])
                            class_ids.append(obj_id)
                            confidences.append(1.0)  # Tracking confidence
                
                if mask_list:
                    # Create supervision Detections object
                    detections = sv.Detections(
                        xyxy=np.array(box_list),
                        mask=np.array(mask_list),
                        class_id=np.array(class_ids),
                        confidence=np.array(confidences)
                    )
                    
                    # Annotate frame
                    annotated_frame = mask_annotator.annotate(scene=frame.copy(), detections=detections)
                    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, 
                                                           labels=[ID_TO_OBJECTS[i] for i in class_ids])
                    
                    # Write annotated frame
                    out_video.write(annotated_frame)
                else:
                    # No masks, write original frame
                    out_video.write(frame)
            else:
                # No tracking data, write original frame
                out_video.write(frame)
        
        # Release video writer
        out_video.release()
        
        # Step 8: Save metadata
        metadata = {
            "video_name": video_name,
            "prompts": prompts_text,
            "detections": [{"bbox": bbox.tolist(), "label": label} for bbox, label in detection_results],
            "video_info": {
                "fps": video_info.fps,
                "width": video_info.width,
                "height": video_info.height,
                "frame_count": frame_count
            },
            "parameters": {
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "prompt_type": prompt_type
            }
        }
        
        metadata_path = Path(output_base_dir) / f"{video_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Video processing complete!")
        print(f"   Output video: {output_video_path}")
        print(f"   Metadata: {metadata_path}")


def main():
    args = parse_args()
    
    # Load prompts and get video files
    prompts = load_prompts(args.prompts_file)
    video_files = get_video_files(args.input_video_dir)
    
    print(f"Found {len(video_files)} video files and {len(prompts)} prompts")
    
    if len(video_files) != len(prompts):
        print(f"⚠️  Warning: Number of videos ({len(video_files)}) != number of prompts ({len(prompts)})")
        min_count = min(len(video_files), len(prompts))
        print(f"Processing first {min_count} videos/prompts")
        video_files = video_files[:min_count]
        prompts = prompts[:min_count]
    
    # Initialize models
    processor, grounding_model, video_predictor, image_predictor, device = initialize_models(
        args.grounding_model, args.sam2_checkpoint, args.sam2_config
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each video
    for video_file, prompt in zip(video_files, prompts):
        try:
            process_single_video(
                video_file, prompt, processor, grounding_model,
                video_predictor, image_predictor, device, args.output_dir,
                args.box_threshold, args.text_threshold, 
                args.overlap_threshold, args.prompt_type
            )
        except Exception as e:
            print(f"❌ Error processing {video_file}: {e}")
            continue
    
    print("\n🎉 Batch processing complete!")


if __name__ == "__main__":
    main()
