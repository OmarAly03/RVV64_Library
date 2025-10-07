#!/usr/bin/env python3

import numpy as np
import struct
import matplotlib.pyplot as plt
from typing import Tuple, List

def load_binary_data(filename: str, dtype=np.float32) -> np.ndarray:
    """
    Load binary data from file
    
    Args:
        filename: Path to binary file
        dtype: Data type to load
        
    Returns:
        Loaded data as numpy array
    """
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=dtype)
    return data

def compute_iou_numpy(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes using NumPy
    
    Args:
        box1: First box [y1, x1, y2, x2]
        box2: Second box [y1, x1, y2, x2]
        
    Returns:
        IoU score
    """
    # Extract coordinates
    y1_1, x1_1, y2_1, x2_1 = box1
    y1_2, x1_2, y2_2, x2_2 = box2
    
    # Compute intersection coordinates
    inter_y1 = max(y1_1, y1_2)
    inter_x1 = max(x1_1, x1_2)
    inter_y2 = min(y2_1, y2_2)
    inter_x2 = min(x2_1, x2_2)
    
    # Compute intersection area
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height
    
    # Compute areas of both boxes
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Compute union area
    union_area = area1 + area2 - inter_area
    
    # Compute IoU
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def nms_reference_numpy(boxes: np.ndarray, scores: np.ndarray, 
                       max_output_boxes_per_class: int = 50,
                       iou_threshold: float = 0.5, 
                       score_threshold: float = 0.1) -> List[Tuple[int, int, int]]:
    """
    Reference NMS implementation using pure NumPy
    
    Args:
        boxes: Box array [batch_size, spatial_dimension, 4]
        scores: Score array [batch_size, num_classes, spatial_dimension]
        max_output_boxes_per_class: Maximum boxes per class
        iou_threshold: IoU threshold
        score_threshold: Score threshold
        
    Returns:
        List of selected indices (batch_idx, class_idx, box_idx)
    """
    selected_indices = []
    num_batches, spatial_dimension, _ = boxes.shape
    _, num_classes, _ = scores.shape
    
    for batch in range(num_batches):
        for cls in range(num_classes):
            # Get scores for current batch and class
            class_scores = scores[batch, cls, :]
            
            # Filter by score threshold
            valid_indices = np.where(class_scores >= score_threshold)[0]
            if len(valid_indices) == 0:
                continue
                
            valid_scores = class_scores[valid_indices]
            
            # Sort by score in descending order
            sorted_indices = valid_indices[np.argsort(valid_scores)[::-1]]
            
            # Apply NMS
            suppressed = set()
            selected_count = 0
            
            for i, box_idx in enumerate(sorted_indices):
                if box_idx in suppressed or selected_count >= max_output_boxes_per_class:
                    continue
                    
                selected_indices.append((batch, cls, box_idx))
                selected_count += 1
                
                # Suppress overlapping boxes
                current_box = boxes[batch, box_idx, :]
                
                for j in range(i + 1, len(sorted_indices)):
                    other_box_idx = sorted_indices[j]
                    if other_box_idx in suppressed:
                        continue
                        
                    other_box = boxes[batch, other_box_idx, :]
                    iou = compute_iou_numpy(current_box, other_box)
                    
                    if iou > iou_threshold:
                        suppressed.add(other_box_idx)
    
    return selected_indices

def compute_metrics(reference_results: List, test_results: List) -> dict:
    """
    Compute comparison metrics between two NMS result sets
    
    Args:
        reference_results: Reference NMS results
        test_results: Test NMS results
        
    Returns:
        Dictionary with metrics
    """
    ref_set = set(map(tuple, reference_results))
    test_set = set(map(tuple, test_results))
    
    intersection = ref_set.intersection(test_set)
    union = ref_set.union(test_set)
    
    precision = len(intersection) / len(test_set) if len(test_set) > 0 else 0.0
    recall = len(intersection) / len(ref_set) if len(ref_set) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou_metric = len(intersection) / len(union) if len(union) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'iou_metric': iou_metric,
        'reference_count': len(ref_set),
        'test_count': len(test_set),
        'matches': len(intersection),
        'exact_match': ref_set == test_set
    }

def visualize_results(boxes: np.ndarray, scores: np.ndarray, 
                     selected_indices: List[Tuple[int, int, int]], 
                     batch_idx: int = 0, class_idx: int = 0,
                     save_path: str = None, show_plot: bool = False):
    """
    Visualize NMS results for a specific batch and class
    
    Args:
        boxes: Box coordinates
        scores: Score array
        selected_indices: Selected box indices
        batch_idx: Batch to visualize
        class_idx: Class to visualize
        save_path: Path to save plot
        show_plot: Whether to display plot
    """
    # Filter selected indices for the specified batch and class
    selected_boxes = [
        box_idx for (b, c, box_idx) in selected_indices 
        if b == batch_idx and c == class_idx
    ]
    
    if len(selected_boxes) == 0:
        print(f"No selected boxes for batch {batch_idx}, class {class_idx}")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Get all boxes for this batch
    batch_boxes = boxes[batch_idx]
    batch_scores = scores[batch_idx, class_idx]
    
    # Plot all boxes with low opacity
    for i, box in enumerate(batch_boxes):
        y1, x1, y2, x2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Color based on score
        color = 'lightgray' if batch_scores[i] < 0.1 else 'lightblue'
        alpha = 0.3
        
        rect = plt.Rectangle((x1, y1), width, height, 
                           linewidth=1, edgecolor=color, facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        
        # Add score text
        ax.text(x1, y1, f'{batch_scores[i]:.2f}', fontsize=8, alpha=0.7)
    
    # Highlight selected boxes
    for i, box_idx in enumerate(selected_boxes):
        box = batch_boxes[box_idx]
        y1, x1, y2, x2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Use different colors for different selected boxes
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
        color = colors[i % len(colors)]
        
        rect = plt.Rectangle((x1, y1), width, height, 
                           linewidth=3, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add selection number
        ax.text(x1, y2, f'#{i+1}', fontsize=12, color=color, fontweight='bold')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(f'NMS Results - Batch {batch_idx}, Class {class_idx}\n'
                f'Selected {len(selected_boxes)} boxes')
    ax.grid(True, alpha=0.3)
    
    # Invert y-axis to match image coordinates
    ax.invert_yaxis()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def generate_test_data(num_batches: int = 1, num_classes: int = 2, 
                      spatial_dimension: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate test data for NMS
    
    Args:
        num_batches: Number of batches
        num_classes: Number of classes
        spatial_dimension: Number of boxes per batch
        seed: Random seed
        
    Returns:
        Tuple of (boxes, scores)
    """
    np.random.seed(seed)
    
    # Generate random boxes in corner format [y1, x1, y2, x2]
    boxes = np.zeros((num_batches, spatial_dimension, 4), dtype=np.float32)
    
    for batch in range(num_batches):
        for i in range(spatial_dimension):
            # Generate random center and size
            center_y = np.random.uniform(10, 90)
            center_x = np.random.uniform(10, 90)
            height = np.random.uniform(2, 20)
            width = np.random.uniform(2, 20)
            
            # Convert to corner format
            y1 = center_y - height / 2
            x1 = center_x - width / 2
            y2 = center_y + height / 2
            x2 = center_x + width / 2
            
            # Clamp to valid range
            boxes[batch, i] = [
                max(0, y1), max(0, x1),
                min(100, y2), min(100, x2)
            ]
    
    # Generate random scores
    scores = np.random.uniform(0, 1, (num_batches, num_classes, spatial_dimension)).astype(np.float32)
    
    return boxes, scores

if __name__ == "__main__":
    # Test utility functions
    print("Testing NMS utility functions...")
    
    # Generate test data
    boxes, scores = generate_test_data()
    print(f"Generated test data: boxes={boxes.shape}, scores={scores.shape}")
    
    # Test reference implementation
    selected = nms_reference_numpy(boxes, scores)
    print(f"Reference NMS selected {len(selected)} boxes")
    
    # Test visualization
    try:
        visualize_results(boxes, scores, selected, save_path="test_nms_viz.png")
        print("Visualization test completed")
    except Exception as e:
        print(f"Visualization test failed: {e}")