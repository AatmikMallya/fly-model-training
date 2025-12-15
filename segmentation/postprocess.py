# postprocess.py
# ================== Imports ==================
import numpy as np
import argparse
import os
from scipy.spatial import KDTree
from skimage.morphology import skeletonize_3d
import networkx as nx
from typing import Set, Tuple

np.set_printoptions(precision=5, suppress=True)

def create_weight_map(box_size: int = 96, center_size: int = 60, min_weight: float = 0.1) -> np.ndarray:
    """Create weight map for volume merging."""
    start_falloff = (box_size - center_size) // 2
    falloff = np.linspace(min_weight, 1.0, start_falloff, dtype=np.float32)
    dim_mask = np.ones(box_size, dtype=np.float32)
    dim_mask[:start_falloff] = falloff
    dim_mask[-start_falloff:] = falloff[::-1]
    return dim_mask[:, None, None] * dim_mask[None, :, None] * dim_mask[None, None, :]

class MicrotubuleProcessor:
    def __init__(self, boxes: np.ndarray, predictions: np.ndarray):
        """Initialize processor with boxes and predictions."""
        self.boxes = boxes
        self.predictions = predictions
        self.centers = (self.boxes[:,0] + self.boxes[:,1]) / 2
        self.weight_map = create_weight_map()
        
        # Build KDTree for neighbor finding
        self.tree = KDTree(self.centers)
        self.radius = 96 * np.sqrt(3)
    
    def find_overlapping_boxes(self, active_set: Set[int], processed: Set[int], 
                             unprocessed: Set[int]) -> Set[int]:
        """Find unprocessed boxes that overlap with active set."""
        new_overlaps = set()
        active_list = list(active_set)
        active_boxes = self.boxes[active_list]
        
        # Get potential neighbors using KDTree
        neighbors = set()
        for idx in active_list:
            neighbors.update(self.tree.query_ball_point(self.centers[idx], self.radius))
        
        candidates = neighbors & unprocessed - processed - active_set
        if not candidates:
            return new_overlaps
            
        # Check overlap for remaining candidates
        candidate_boxes = self.boxes[list(candidates)]
        for i, n in enumerate(candidates):
            box = candidate_boxes[i]
            if np.any((active_boxes[:,1] > box[0]).all(axis=1) & 
                     (box[1] > active_boxes[:,0]).all(axis=1)):
                new_overlaps.add(n)
        
        return new_overlaps
    
    def find_disjoint_sets(self, box_indices: List[int]) -> List[Set[int]]:
        """Split boxes into disjoint sets based on overlap."""
        if len(box_indices) <= 1:
            return [set(box_indices)]
            
        G = nx.Graph()
        G.add_nodes_from(box_indices)
        boxes = self.boxes[box_indices]
        
        for i, idx1 in enumerate(box_indices[:-1]):
            box1 = boxes[i]
            remaining_boxes = boxes[i+1:]
            overlaps = ((remaining_boxes[:,1] > box1[0]).all(axis=1) & 
                       (box1[1] > remaining_boxes[:,0]).all(axis=1))
            
            for j, overlaps in enumerate(overlaps, i+1):
                if overlaps:
                    G.add_edge(idx1, box_indices[j])
        
        return list(nx.connected_components(G))
    
    def merge_volumes(self, box_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Merge volumes from multiple boxes."""
        box_subset = self.boxes[box_indices]
        min_coords = np.min(box_subset[:,0], axis=0)
        max_coords = np.max(box_subset[:,1], axis=0)
        shape = max_coords - min_coords
        
        weighted_sum = np.zeros(shape, dtype=np.float32)
        weight_sum = np.zeros(shape, dtype=np.float32)
        
        for idx in box_indices:
            box = self.boxes[idx]
            local_min = box[0] - min_coords
            local_max = box[1] - min_coords
            sl = tuple(slice(mn, mx) for mn, mx in zip(local_min, local_max))
            
            weighted_sum[sl] += self.weight_map * self.predictions[idx]
            weight_sum[sl] += self.weight_map
        
        valid_mask = weight_sum > 0
        merged = np.zeros_like(weighted_sum)
        merged[valid_mask] = weighted_sum[valid_mask] / weight_sum[valid_mask]
        
        return merged, min_coords
    
    def process_volume(self, probability_threshold: float = 0.5) -> np.ndarray:
        """Process full volume using wavefront propagation."""
        processed = set()
        all_points = []
        unprocessed = set(range(len(self.boxes)))
        
        print(f"Processing {len(self.boxes)} boxes...")
        component = 0
        
        while unprocessed:
            component += 1
            start_idx = unprocessed.pop()
            active_set = {start_idx}
            
            while active_set:
                new_overlaps = self.find_overlapping_boxes(active_set, processed, unprocessed)
                
                if active_set:
                    disjoint_sets = self.find_disjoint_sets(list(active_set))
                    
                    for box_set in disjoint_sets:
                        merged, origin = self.merge_volumes(list(box_set))
                        binary = merged > probability_threshold
                        skeleton = skeletonize_3d(binary)
                        
                        if np.any(skeleton):
                            points = np.argwhere(skeleton) + origin
                            all_points.append(points)
                
                processed.update(active_set)
                unprocessed.difference_update(active_set)
                active_set = new_overlaps
        
        print(f"Processed {component} components")
        
        if all_points:
            points = np.unique(np.vstack(all_points), axis=0)
        else:
            points = np.array([])
        
        return points

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing UNET predictions')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save skeletonized results')
    parser.add_argument('--bodyId', type=int, required=True,
                        help='Body ID of the neuron being processed')
    parser.add_argument('--probability_threshold', type=float, default=0.5,
                        help='Threshold for binary prediction')
    args = parser.parse_args()
    
    try:
        # Load data
        boxes = np.load(f"{args.input_dir}/boxes_{args.bodyId}.npy")
        predictions = np.load(f"{args.input_dir}/raw_predictions_{args.bodyId}.npy")
        
        print(f"Loaded {len(boxes)} boxes for bodyId {args.bodyId}")
        
        # Process microtubules
        processor = MicrotubuleProcessor(boxes, predictions)
        skeleton_points = processor.process_volume(args.probability_threshold)
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(f"{args.output_dir}/skeleton_points_{args.bodyId}.npy", skeleton_points)
        print(f"Saved {len(skeleton_points)} skeleton points for bodyId {args.bodyId}")
        
    except Exception as e:
        print(f"Error processing bodyId {args.bodyId}: {str(e)}")
        raise

if __name__ == "__main__":
    main()




