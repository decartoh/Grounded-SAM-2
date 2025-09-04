#!/usr/bin/env python3
"""
Prompt-to-BoundingBox Assignment Algorithm

Optimal assignment strategy for assigning detected bounding boxes to text prompts,
maximizing total confidence while avoiding overlaps above IoU threshold.
"""

import numpy as np
from itertools import product
from typing import List, Tuple, Dict, Optional


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: Bounding boxes in format [x, y, width, height]
        
    Returns:
        IoU value between 0.0 and 1.0
    """
    # Convert to (x1, y1, x2, y2) format
    x1_1, y1_1, w1, h1 = box1[0], box1[1], box1[2], box1[3]
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    
    x1_2, y1_2, w2, h2 = box2[0], box2[1], box2[2], box2[3] 
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    # Calculate intersection coordinates
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # No intersection
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    # Calculate areas
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


class OptimalSumAssigner:
    """Assigns bboxes to maximize total confidence sum with no overlaps."""
    
    def assign(self, prompt_detections: Dict[str, List[Tuple[List[float], float]]], 
               overlap_threshold: float = 0.9) -> Tuple[List[List[float]], List[float], List[str]]:
        """Find optimal 1:1 mapping that maximizes sum of confidence scores.
        
        Args:
            prompt_detections: Dict mapping prompt -> [(bbox, confidence), ...]
            overlap_threshold: IoU threshold for considering boxes as overlapping (default: 0.9)
            
        Returns:
            Tuple of (bboxes, confidences, prompt_labels)
        """
        
        prompts = list(prompt_detections.keys())
        
        # Create all possible combinations (Cartesian product)
        prompt_detection_options = []
        for prompt in prompts:
            if prompt_detections[prompt]:
                # Add all detections for this prompt with their prompt label
                options = [(prompt, bbox, conf) for bbox, conf in prompt_detections[prompt]]
                prompt_detection_options.append(options)
            else:
                # No detections for this prompt
                prompt_detection_options.append([])
        
        # If any prompt has no detections, return partial mapping
        if any(len(options) == 0 for options in prompt_detection_options):
            missing_prompts = [prompts[i] for i, options in enumerate(prompt_detection_options) if len(options) == 0]
            print(f"Cannot create complete mapping - no detections for: {missing_prompts}")
            
            # Return best available detections (partial mapping)
            final_boxes = []
            final_confidences = []
            final_class_names = []
            
            for prompt in prompts:
                if prompt_detections[prompt]:
                    bbox, conf = prompt_detections[prompt][0]  # Best detection
                    final_boxes.append(bbox)
                    final_confidences.append(conf)
                    final_class_names.append(prompt)
            
            return final_boxes, final_confidences, final_class_names
        
        # Find all possible 1:1 mappings
        all_combinations = list(product(*prompt_detection_options))
        
        best_mapping = None
        best_score = -1
        
        print(f"Evaluating {len(all_combinations)} possible 1:1 mappings...")
        
        eliminated_by_iou = 0
        valid_combinations = 0
        
        for combination in all_combinations:
            # Extract bounding boxes from this combination
            boxes = [item[1] for item in combination]  # item = (prompt, bbox, conf)
            confidences = [item[2] for item in combination]
            
            # Check if any boxes overlap above threshold
            has_overlap = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    iou = calculate_iou(boxes[i], boxes[j])
                    if iou > overlap_threshold:
                        has_overlap = True
                        break
                if has_overlap:
                    break
            
            # Count eliminations and valid combinations
            if has_overlap:
                eliminated_by_iou += 1
            else:
                valid_combinations += 1
                # Calculate total confidence for valid combinations
                total_confidence = sum(confidences)
                if total_confidence > best_score:
                    best_score = total_confidence
                    best_mapping = combination
        
        # Print assignment statistics
        print(f"Assignment statistics:")
        print(f"  • Total combinations evaluated: {len(all_combinations)}")
        print(f"  • Eliminated due to IoU > {overlap_threshold}: {eliminated_by_iou}")
        print(f"  • Valid (non-overlapping) combinations: {valid_combinations}")
        
        if best_mapping is None:
            print(f"No non-overlapping 1:1 mapping found with IoU threshold {overlap_threshold}")
            return [], [], []
        
        # Extract final results from best mapping
        final_boxes = [item[1] for item in best_mapping]
        final_confidences = [item[2] for item in best_mapping]
        final_class_names = [item[0] for item in best_mapping]
        
        print(f"Optimal mapping found with total confidence: {best_score:.3f}")
        for prompt, bbox, conf in best_mapping:
            print(f"  {prompt} → confidence {conf:.3f}")
        
        return final_boxes, final_confidences, final_class_names