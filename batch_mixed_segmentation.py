#!/usr/bin/env python3
"""
Batch Mixed Segmentation with HuggingFace GroundingDINO and SAM2
Process both images and videos in the same directory with corresponding text prompts for object segmentation.

For images: Apply segmentation and save visualization with _seg suffix
For videos: Full video processing pipeline with tracking
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
    parser = argparse.ArgumentParser(description="Batch mixed image/video segmentation with HuggingFace GroundingDINO and SAM2")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing images and/or videos to process")
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
                       help="Prompt type for SAM2 predictor")
    parser.add_argument("--sam2_checkpoint", type=str, 
                       default="./checkpoints/sam2.1_hiera_large.pt",
                       help="Path to SAM2 model checkpoint")
    parser.add_argument("--sam2_config", type=str, 
                       default="configs/sam2.1/sam2.1_hiera_l.yaml",
                       help="Path to SAM2 model config")
    parser.add_argument("--port", type=int, default=8666,
                       help="Port number (for reference, not used in processing)")
    return parser.parse_args()


def get_image_files(input_dir):
    """Get all image files from directory, sorted by name."""
    input_dir = Path(input_dir)
    image_extensions = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
    return sorted(image_files)


def get_video_files(input_dir):
    """Get all MP4 files from directory, sorted by name."""
    input_dir = Path(input_dir)
    video_files = sorted([f for f in input_dir.glob("*.mp4")])
    return video_files


def get_prompts_for_image(image_path):
    """Get prompts from corresponding _seg.txt file."""
    image_stem = image_path.stem
    seg_file = image_path.parent / f"{image_stem}_seg.txt"
    
    if seg_file.exists():
        with open(seg_file, 'r') as f:
            prompts = f.read().strip()
        return prompts
    return None


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


def get_detection_boxes(processor, grounding_model, device, image, prompts_text, 
                       box_threshold, text_threshold, overlap_threshold):
    """Use HuggingFace GroundingDINO to get detection boxes for objects."""
    # Split the prompt by comma to get individual object prompts
    individual_prompts = [prompt.strip() for prompt in prompts_text.split(',') if prompt.strip()]
    
    if len(individual_prompts) == 0:
        return []
    
    print(f"🔍 Detecting objects for prompts: {individual_prompts}")
    
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
        
        print(f"    Found {len(boxes)} detections for '{prompt}' (threshold: {box_threshold})")
        if len(scores) > 0:
            confidence_str = ', '.join([f"{score:.3f}" for score in sorted(scores, reverse=True)])
            print(f"    Confidences: [{confidence_str}]")
        
        # Store detections for this prompt
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
        print("❌ No detections found for any prompt")
        return []
    
    # Use OptimalSumAssigner to assign prompts to boxes
    assigner = OptimalSumAssigner()
    bboxes, confidences, prompt_labels = assigner.assign(prompt_detections, overlap_threshold)
    
    if not bboxes:
        print("❌ No valid assignments found")
        return []
    
    print(f"✅ Assignment found {len(bboxes)} valid prompt-box pairs")
    
    # Convert assignments to final format
    detection_results = []
    for bbox_xywh, confidence, prompt in zip(bboxes, confidences, prompt_labels):
        # Convert back to xyxy format for SAM2
        x, y, w, h = bbox_xywh
        bbox_xyxy = [x, y, x + w, y + h]
        detection_results.append({
            'bbox': bbox_xyxy,
            'label': prompt,
            'confidence': float(confidence)
        })
    
    return detection_results


def process_single_image(image_path, prompts_text, processor, grounding_model, 
                        image_predictor, device, output_dir, 
                        box_threshold, text_threshold, overlap_threshold, prompt_type):
    """Process a single image with given prompts and save segmentation visualization."""
    
    image_name = image_path.stem
    print(f"\n🖼️  Processing image: {image_name}")
    print(f"   Prompts: {prompts_text}")
    
    # Load image
    image_pil = Image.open(image_path)
    image_np = np.array(image_pil)
    
    # Run object detection
    print("🔍 Running object detection...")
    detection_results = get_detection_boxes(
        processor, grounding_model, device, image_pil, prompts_text,
        box_threshold, text_threshold, overlap_threshold
    )
    
    if not detection_results:
        print("❌ No valid detections found, skipping image")
        return
    
    print(f"✅ Found {len(detection_results)} valid detections")
    
    # Set up SAM2 image predictor
    image_predictor.set_image(image_np)
    
    # Prepare prompts for SAM2 based on detection results
    input_boxes = []
    labels = []
    
    for detection in detection_results:
        bbox = detection['bbox']
        label = detection['label']
        
        input_boxes.append(bbox)
        labels.append(label)
    
    input_boxes = np.array(input_boxes)
    
    # Generate SAM2 prompts based on prompt_type
    if prompt_type == "box":
        # Use bounding boxes directly
        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
    elif prompt_type == "point":
        # Sample points from boxes
        point_coords = []
        point_labels = []
        for bbox in input_boxes:
            x1, y1, x2, y2 = bbox
            # Sample center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            point_coords.append([center_x, center_y])
            point_labels.append(1)  # foreground point
        
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)
        
        masks, scores, logits = image_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=None,
            multimask_output=False,
        )
    else:  # mask - use boxes to generate initial masks
        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
    
    # Create visualization
    print("🎨 Creating segmentation visualization...")
    
    # Convert masks to supervision format
    mask_list = []
    box_list = []
    class_ids = []
    confidences = []
    
    for i, (mask, detection) in enumerate(zip(masks, detection_results)):
        if mask.any():  # Only process non-empty masks
            # Handle potential 3D mask (squeeze to 2D)
            if mask.ndim > 2:
                mask = mask.squeeze()
            
            mask_list.append(mask)
            box_list.append(detection['bbox'])
            class_ids.append(i)
            confidences.append(detection['confidence'])
    
    if mask_list:
        # Create supervision Detections object
        detections = sv.Detections(
            xyxy=np.array(box_list),
            mask=np.array(mask_list),
            class_id=np.array(class_ids),
            confidence=np.array(confidences)
        )
        
        # Annotate image
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        
        annotated_image = mask_annotator.annotate(scene=image_np.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, 
                                                labels=[labels[i] for i in class_ids])
        
        # Save segmented image with _seg suffix
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_image_path = output_dir / f"{image_name}_seg{image_path.suffix}"
        cv2.imwrite(str(output_image_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
        # Save metadata
        metadata = {
            "image_path": str(image_path),
            "prompts": prompts_text,
            "detections": [
                {
                    "bbox": [float(x) for x in detection['bbox']],
                    "label": detection['label'],
                    "confidence": float(detection['confidence'])
                }
                for detection in detection_results
            ],
            "parameters": {
                "box_threshold": float(box_threshold),
                "text_threshold": float(text_threshold),
                "prompt_type": prompt_type
            }
        }
        
        metadata_path = output_dir / f"{image_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Segmented image saved: {output_image_path}")
        print(f"   Metadata: {metadata_path}")
    else:
        print("❌ No valid masks generated")


def extract_video_frames(video_path, temp_frames_dir):
    """Extract frames from video and save as temporary JPEG files for SAM2 processing."""
    temp_frames_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create a simple namespace for video info
    class VideoInfo:
        def __init__(self, fps, width, height, frame_count):
            self.fps = fps
            self.width = width
            self.height = height
            self.frame_count = frame_count
    
    video_info = VideoInfo(fps, width, height, frame_count)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_path = temp_frames_dir / f"{frame_idx:05d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_idx += 1
    
    cap.release()
    print(f"✅ Extracted {frame_idx} frames from video")
    
    return video_info, frame_idx


def process_single_video(video_path, prompts_text, processor, grounding_model, 
                        video_predictor, image_predictor, device, output_base_dir, 
                        box_threshold, text_threshold, overlap_threshold, prompt_type):
    """Process a single video with given prompts (full video processing pipeline)."""
    
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
        
        input_boxes = []
        labels = []
        
        for detection in detection_results:
            input_boxes.append(detection['bbox'])
            labels.append(detection['label'])
        
        input_boxes = np.array(input_boxes)
        
        # Generate prompts for SAM2 based on prompt_type
        if prompt_type == "box":
            prompts = input_boxes
        elif prompt_type == "point":
            # Sample points from masks generated from boxes
            masks, _, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            prompts = sample_points_from_masks(masks)
        else:  # mask
            # Generate masks from boxes
            masks, _, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            prompts = masks
        
        # Add objects to video predictor
        for i, prompt in enumerate(prompts):
            if prompt_type == "box":
                video_predictor.add_new_bbox(
                    inference_state=inference_state,
                    bbox=prompt,
                    frame_idx=0,
                    obj_id=i,
                )
            elif prompt_type == "point":
                video_predictor.add_new_points_or_box(
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
                
                # Process each tracked object
                for obj_id in range(len(labels)):
                    if obj_id in masks_data:
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
            "video_path": str(video_path),
            "prompts": prompts_text,
            "detections": [
                {
                    "bbox": [float(x) for x in detection['bbox']],
                    "label": detection['label'],
                    "confidence": float(detection['confidence'])
                }
                for detection in detection_results
            ],
            "video_info": {
                "fps": float(video_info.fps),
                "width": int(video_info.width),
                "height": int(video_info.height),
                "frame_count": int(video_info.frame_count)
            },
            "parameters": {
                "box_threshold": float(box_threshold),
                "text_threshold": float(text_threshold),
                "prompt_type": prompt_type
            }
        }
        
        metadata_path = Path(output_base_dir) / f"{video_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Segmented video saved: {output_video_path}")
        print(f"   Metadata: {metadata_path}")


def main():
    args = parse_args()
    
    print(f"🚀 Starting batch mixed segmentation processing")
    print(f"   Input directory: {args.input_dir}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Port (reference): {args.port}")
    
    # Get image and video files
    image_files = get_image_files(args.input_dir)
    video_files = get_video_files(args.input_dir)
    
    print(f"\n📊 Found {len(image_files)} image files and {len(video_files)} video files")
    
    if len(image_files) == 0 and len(video_files) == 0:
        print("❌ No image or video files found in input directory")
        return
    
    # Initialize models
    processor, grounding_model, video_predictor, image_predictor, device = initialize_models(
        args.grounding_model, args.sam2_checkpoint, args.sam2_config
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images first
    if image_files:
        print(f"\n🖼️  Processing {len(image_files)} images...")
        
        for image_file in image_files:
            try:
                # Get prompts from corresponding _seg.txt file
                prompts = get_prompts_for_image(image_file)
                
                if prompts:
                    process_single_image(
                        image_file, prompts, processor, grounding_model,
                        image_predictor, device, args.output_dir,
                        args.box_threshold, args.text_threshold, 
                        args.overlap_threshold, args.prompt_type
                    )
                else:
                    print(f"⚠️  No prompts found for image {image_file.name} (looking for {image_file.stem}_seg.txt)")
                    
            except Exception as e:
                print(f"❌ Error processing image {image_file}: {e}")
                continue
    
    # Process videos if present
    if video_files:
        print(f"\n🎬 Processing {len(video_files)} videos...")
        
        for video_file in video_files:
            try:
                # For videos, we'd need a different way to get prompts
                # For now, let's assume there's a prompts.txt file or similar
                # This is a placeholder - you may need to modify based on your video prompt structure
                prompts = "person, car"  # Default prompts - modify as needed
                
                process_single_video(
                    video_file, prompts, processor, grounding_model,
                    video_predictor, image_predictor, device, args.output_dir,
                    args.box_threshold, args.text_threshold, 
                    args.overlap_threshold, args.prompt_type
                )
                    
            except Exception as e:
                print(f"❌ Error processing video {video_file}: {e}")
                continue
    
    print("\n🎉 Batch processing complete!")


if __name__ == "__main__":
    main()
