#!/usr/bin/env python3
"""
Prompt-to-BoundingBox Assignment Algorithms

This module provides different strategies for assigning detected bounding boxes
to text prompts, handling overlaps and optimizing for various criteria.
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


class PromptBBoxAssigner:
    """Base class for prompt-to-bbox assignment algorithms."""
    
    def assign(self, prompt_detections: Dict[str, List[Tuple[List[float], float]]], 
               overlap_threshold: float = 0.5) -> Tuple[List[List[float]], List[float], List[str]]:
        """Assign bounding boxes to prompts.
        
        Args:
            prompt_detections: Dict mapping prompt -> [(bbox, confidence), ...]
            overlap_threshold: IoU threshold for considering boxes as overlapping
            
        Returns:
            Tuple of (bboxes, confidences, prompt_labels)
        """
        raise NotImplementedError


class OptimalSumAssigner(PromptBBoxAssigner):
    """Assigns bboxes to maximize total confidence sum with no overlaps."""
    
    def assign(self, prompt_detections: Dict[str, List[Tuple[List[float], float]]], 
               overlap_threshold: float = 0.5) -> Tuple[List[List[float]], List[float], List[str]]:
        """Find optimal 1:1 mapping that maximizes sum of confidence scores."""
        
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
        
        for combination in all_combinations:
            # Extract bounding boxes from this combination
            boxes = [item[1] for item in combination]  # item = (prompt, bbox, conf)
            confidences = [item[2] for item in combination]
            
            # Check if any boxes overlap
            has_overlap = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    iou = calculate_iou(boxes[i], boxes[j])
                    if iou > overlap_threshold:
                        has_overlap = True
                        break
                if has_overlap:
                    break
            
            # If no overlaps, calculate total confidence
            if not has_overlap:
                total_confidence = sum(confidences)
                if total_confidence > best_score:
                    best_score = total_confidence
                    best_mapping = combination
        
        if best_mapping is None:
            print("No non-overlapping 1:1 mapping found")
            return [], [], []
        
        # Extract final results from best mapping
        final_boxes = [item[1] for item in best_mapping]
        final_confidences = [item[2] for item in best_mapping]
        final_class_names = [item[0] for item in best_mapping]
        
        print(f"Optimal mapping found with total confidence: {best_score:.3f}")
        for prompt, bbox, conf in best_mapping:
            print(f"  {prompt} → confidence {conf:.3f}")
        
        return final_boxes, final_confidences, final_class_names


class GreedyConfidenceAssigner(PromptBBoxAssigner):
    """Assigns bboxes greedily by confidence, resolving overlaps iteratively."""
    
    def assign(self, prompt_detections: Dict[str, List[Tuple[List[float], float]]], 
               overlap_threshold: float = 0.5) -> Tuple[List[List[float]], List[float], List[str]]:
        """Greedy assignment starting with highest confidence prompts."""
        
        prompts = list(prompt_detections.keys())
        
        # Sort prompts by their best detection confidence
        prompts_by_confidence = []
        for prompt in prompts:
            if prompt_detections[prompt]:
                best_confidence = prompt_detections[prompt][0][1]  # Assume sorted by confidence
                prompts_by_confidence.append((prompt, best_confidence))
            else:
                prompts_by_confidence.append((prompt, 0.0))
        
        prompts_by_confidence.sort(key=lambda x: x[1], reverse=True)
        print(f"Processing prompts by confidence: {[f'{p}({c:.3f})' for p, c in prompts_by_confidence]}")
        
        # Greedy assignment
        final_boxes = []
        final_confidences = []
        final_class_names = []
        
        for i, (prompt, best_conf) in enumerate(prompts_by_confidence):
            detections = prompt_detections[prompt]
            if not detections:
                print(f"  {prompt}: No valid detections")
                continue
            
            if i == 0:
                # First prompt - take the best detection
                bbox, confidence = detections[0]
                final_boxes.append(bbox)
                final_confidences.append(confidence)
                final_class_names.append(prompt)
                print(f"  {prompt}: ANCHOR → confidence {confidence:.3f}")
            else:
                # Subsequent prompts - find first non-overlapping detection
                selected_detection = None
                
                for rank, (bbox, confidence) in enumerate(detections):
                    # Check overlap with all previously selected boxes
                    has_overlap = False
                    for existing_bbox in final_boxes:
                        iou = calculate_iou(bbox, existing_bbox)
                        if iou > overlap_threshold:
                            has_overlap = True
                            print(f"    {prompt}: #{rank+1} detection overlaps (IoU: {iou:.3f})")
                            break
                    
                    if not has_overlap:
                        selected_detection = (bbox, confidence, rank + 1)
                        break
                
                if selected_detection:
                    bbox, confidence, rank = selected_detection
                    final_boxes.append(bbox)
                    final_confidences.append(confidence)
                    final_class_names.append(prompt)
                    print(f"  {prompt}: Using #{rank} detection → confidence {confidence:.3f}")
                else:
                    print(f"  {prompt}: No non-overlapping detection found")
        
        print(f"Final result: {len(final_boxes)} non-overlapping detections")
        return final_boxes, final_confidences, final_class_names


class SingleBestAssigner(PromptBBoxAssigner):
    """Simple assigner that takes only the single best detection per prompt."""
    
    def assign(self, prompt_detections: Dict[str, List[Tuple[List[float], float]]], 
               overlap_threshold: float = 0.5) -> Tuple[List[List[float]], List[float], List[str]]:
        """Take single best detection per prompt, ignoring overlaps."""
        
        final_boxes = []
        final_confidences = []
        final_class_names = []
        
        for prompt, detections in prompt_detections.items():
            if detections:
                bbox, confidence = detections[0]  # Best detection
                final_boxes.append(bbox)
                final_confidences.append(confidence)
                final_class_names.append(prompt)
                print(f"  {prompt} → confidence {confidence:.3f}")
            else:
                print(f"  {prompt}: No detections")
        
        return final_boxes, final_confidences, final_class_names


# Factory function to get assigners
def get_assigner(algorithm: str = "optimal_sum") -> PromptBBoxAssigner:
    """Get assignment algorithm by name.
    
    Args:
        algorithm: One of 'optimal_sum', 'greedy_confidence', 'single_best'
        
    Returns:
        PromptBBoxAssigner instance
    """
    algorithms = {
        "optimal_sum": OptimalSumAssigner,
        "greedy_confidence": GreedyConfidenceAssigner, 
        "single_best": SingleBestAssigner,
    }
    
    if algorithm not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(algorithms.keys())}")
    
    return algorithms[algorithm]()

