# preprocess.py
# ================== Imports ==================
import numpy as np
import argparse
import os
import random
import platform
import importlib
import pandas as pd
from scipy.spatial import KDTree
from neuprint import fetch_skeleton
from typing import List, Dict, Tuple, Set
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
rc('animation', embed_limit=500)

# Local imports
def import_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

home_dir = '/Users/aatmikmallya/Desktop/research/fly/segmentation'
config = import_module('config', f'{home_dir}/util_files/config.py')
voxel_utils = import_module('voxel_utils', f'{home_dir}/util_files/voxel_utils.py')

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '~/.config/gcloud/application_default_credentials.json'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def clip_box(box_min: np.ndarray, box_max: np.ndarray, tile_size: int = 96) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clip box coordinates to dataset bounds while maintaining fixed box size.
    Vectorized implementation.
    """
    min_bounds = np.array([0, 0, 0], dtype=np.int32)
    max_bounds = np.array([41344, 37888, 34367], dtype=np.int32)
    
    # Convert to integers and handle all dimensions at once
    new_min = np.floor(box_min).astype(np.int32)
    new_max = np.floor(box_max).astype(np.int32)
    
    # Vectorized max bound check
    max_exceeded = new_max > max_bounds
    new_max[max_exceeded] = max_bounds[max_exceeded]
    new_min[max_exceeded] = new_max[max_exceeded] - tile_size
    
    # Vectorized min bound check
    min_exceeded = new_min < min_bounds
    new_min[min_exceeded] = min_bounds[min_exceeded]
    new_max[min_exceeded] = new_min[min_exceeded] + tile_size
    
    # Fast size check using sum instead of abs
    if not (new_max - new_min == tile_size).all():
        raise ValueError(f"Box size error: {new_max - new_min}")
        
    return new_min, new_max

def is_valid_box(box_min: np.ndarray, box_max: np.ndarray, min_size: int = 20) -> bool:
    """Check if box is valid after clipping."""
    size = box_max - box_min
    return np.all(size >= min_size)

def get_boxes_for_point(center: np.ndarray, radius: float, tile_size: int = 96, min_overlap: int = 15) -> List[List[List[int]]]:
    """Adaptively place minimum number of boxes needed to cover area around point"""
    boxes = []
    half_size = tile_size // 2
    
    # Always place center box
    box_min = center - half_size
    box_max = box_min + tile_size
    boxes.append([box_min, box_max])
    
    if radius > tile_size//3:
        coverage_radius = radius * 1.25  # Changed from 1.4 to 1.25
        max_rings = int(np.ceil(coverage_radius / (tile_size - min_overlap)))
        
        points_to_check = []
        # Pre-compute angles for all rings
        ring_points = {ring: [] for ring in range(1, max_rings + 1)}
        
        for ring in range(1, max_rings + 1):
            ring_radius = ring * (tile_size - min_overlap)
            num_points = max(8, int(2 * np.pi * ring_radius / (tile_size - min_overlap)))
            angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
            
            # Vectorized point generation for cardinal directions
            cardinal_mask = np.zeros(num_points, dtype=bool)
            cardinal_mask[::num_points//4] = True
            cardinal_angles = angles[cardinal_mask]
            
            # Generate all points at once using broadcasting
            cos_sin = np.column_stack((np.cos(cardinal_angles), np.sin(cardinal_angles), np.zeros_like(cardinal_angles)))
            cardinal_points = center + ring_radius * cos_sin
            points_to_check.extend(cardinal_points)
            
            # Store intermediate angles for potential later use
            if ring > 1:
                intermediate_angles = angles[~cardinal_mask]
                cos_sin = np.column_stack((np.cos(intermediate_angles), np.sin(intermediate_angles), 
                                         np.zeros_like(intermediate_angles)))
                ring_points[ring] = center + ring_radius * cos_sin
        
        # Use KDTree for efficient coverage checking
        if boxes:
            box_centers = np.array([(b[0] + b[1])/2 for b in boxes])
            tree = KDTree(box_centers)
            
            # Process cardinal points first
            points_array = np.array(points_to_check)
            if len(points_array) > 0:
                dists, _ = tree.query(points_array, k=1)
                needs_coverage = dists > tile_size/2
                
                for point in points_array[needs_coverage]:
                    box_min = point - half_size
                    box_max = box_min + tile_size
                    boxes.append([box_min, box_max])
                    # Update KDTree efficiently by stacking arrays
                    box_centers = np.vstack([box_centers, [(box_min + box_max)/2]])
                    tree = KDTree(box_centers)
            
            # Process stored intermediate points if needed
            for ring in range(2, max_rings + 1):
                if len(ring_points[ring]) > 0:
                    dists, _ = tree.query(ring_points[ring], k=1)
                    needs_coverage = dists > tile_size/2
                    
                    for point in ring_points[ring][needs_coverage]:
                        box_min = point - half_size
                        box_max = box_min + tile_size
                        boxes.append([box_min, box_max])
                        box_centers = np.vstack([box_centers, [(box_min + box_max)/2]])
                        tree = KDTree(box_centers)
    
    return boxes

def generate_optimized_boxes(df: pd.DataFrame, tile_size: int = 96, min_overlap: int = 15) -> List[List[List[int]]]:
    """Generate minimal set of boxes ensuring complete coverage"""
    boxes = set()
    half_size = tile_size // 2
    
    # Pre-compute necessary arrays
    points = df[['z', 'y', 'x']].values
    radii = df['radius'].values
    links = df['link'].values
    rowid_to_idx = {row.rowId: i for i, row in df.iterrows()}
    
    print(f"Processing {len(points)} points...")
    
    # More efficient next point counting
    next_counts = np.zeros(len(points))
    valid_links = links >= 0
    next_indices = np.array([rowid_to_idx[link] for link in links[valid_links]])
    np.add.at(next_counts, next_indices, 1)
    
    # Find start points efficiently
    start_points = np.where((valid_links & (next_counts == 0)) | (next_counts != 1))[0]
    print(f"Found {len(start_points)} start points")
    
    processed_points = set()
    
    # Process paths
    for start_idx in start_points:
        if df.iloc[start_idx].rowId in processed_points:
            continue
            
        current_idx = start_idx
        path_points = []
        path_radii = []
        
        while True:
            if df.iloc[current_idx].rowId in processed_points:
                break
                
            path_points.append(points[current_idx])
            path_radii.append(radii[current_idx])
            processed_points.add(df.iloc[current_idx].rowId)
            
            link = links[current_idx]
            next_idx = rowid_to_idx[link] if link >= 0 else -1
            if next_idx >= 0 and next_idx not in processed_points:
                current_idx = next_idx
            else:
                break
        
        if not path_points:
            continue
            
        path_points = np.array(path_points)
        path_radii = np.array(path_radii)
        
        # Vectorized angle computation
        if len(path_points) > 2:
            vectors = np.diff(path_points, axis=0)
            norms = np.linalg.norm(vectors, axis=1)
            cos_angles = np.sum(vectors[1:] * vectors[:-1], axis=1) / (norms[1:] * norms[:-1])
            angles = np.arccos(np.clip(cos_angles, -1, 1))
            bend_points = np.where(angles > np.pi/4)[0]
        else:
            bend_points = []
        
        # Place boxes along path
        path_length = np.sum(np.linalg.norm(np.diff(path_points, axis=0), axis=1))
        num_points = max(1, int(np.ceil(path_length / (tile_size - min_overlap))))
        
        for idx in range(num_points):
            t = idx / (num_points - 1) if num_points > 1 else 0.5
            
            if num_points == 1:
                pos = path_points[len(path_points)//2]
                radius = path_radii[len(path_points)//2]
            else:
                idx_float = t * (len(path_points) - 1)
                idx_low = int(idx_float)
                idx_high = min(idx_low + 1, len(path_points) - 1)
                t_interp = idx_float - idx_low
                
                pos = path_points[idx_low] * (1 - t_interp) + path_points[idx_high] * t_interp
                radius = path_radii[idx_low] * (1 - t_interp) + path_radii[idx_high] * t_interp
                
                if idx_low in bend_points:
                    radius *= 1.25
            
            new_boxes = get_boxes_for_point(pos, radius, tile_size, min_overlap)
            
            for box in new_boxes:
                box_min_clipped, box_max_clipped = clip_box(np.array(box[0]), np.array(box[1]))
                if is_valid_box(box_min_clipped, box_max_clipped):
                    boxes.add((tuple(box_min_clipped), tuple(box_max_clipped)))
    
    # Final gap check using KDTree
    print("Checking for gaps...")
    box_list = list(boxes)
    box_centers = np.array([(np.array(b[0]) + np.array(b[1]))/2 for b in box_list])
    tree = KDTree(box_centers)
    
    # Vectorized gap checking
    margin = tile_size/2 - min_overlap/2
    dists, _ = tree.query(points, k=1)
    gap_points = points[dists > margin]
    
    if len(gap_points) > 0:
        print(f"Adding boxes for {len(gap_points)} uncovered points")
        for point in gap_points:
            box_min = point - half_size
            box_max = box_min + tile_size
            box_min_clipped, box_max_clipped = clip_box(box_min, box_max)
            if is_valid_box(box_min_clipped, box_max_clipped):
                boxes.add((tuple(box_min_clipped), tuple(box_max_clipped)))
    
    print(f"Generated {len(boxes)} total boxes")
    return [list(map(list, box)) for box in boxes]


def filter_empty_boxes(boxes, seg_data):
    """Optimize memory access pattern"""
    keep_mask = []
    for seg in seg_data:
        if not seg.any():
            keep_mask.append(False)
            continue
        
        # Make each axis the fast axis in turn
        seg_y = np.transpose(seg, (1,0,2))  # y becomes first axis
        profile = seg_y.sum(axis=0)
        if (profile >= 8).any():
            keep_mask.append(True)
            continue
            
        seg_x = np.transpose(seg, (2,0,1))  # x becomes first axis
        profile = seg_x.sum(axis=0)
        if (profile >= 8).any():
            keep_mask.append(True)
            continue
            
        # Original z orientation
        profile = seg.sum(axis=0)
        if (profile >= 8).any():
            keep_mask.append(True)
            continue
            
        keep_mask.append(False)
    
    keep_mask = np.array(keep_mask)
    filtered_boxes = [box for box, keep in zip(boxes, keep_mask) if keep]
    filtered_seg_data = seg_data[keep_mask]
    print(f"Removed {len(boxes) - len(filtered_boxes)} empty boxes")
    return np.array(filtered_boxes), filtered_seg_data

def find_uncovered_boundaries(filtered_boxes: np.ndarray, 
                            filtered_seg_data: np.ndarray,
                            tile_size: int = 96,
                            min_overlap: int = 15,
                            boundary_threshold: float = 0.2,
                            max_iterations: int = 5) -> np.ndarray:
    min_bounds = np.array([0, 0, 0])
    max_bounds = np.array([41344, 37888, 34367])
    
    all_boxes = filtered_boxes
    all_seg_data = filtered_seg_data
    
    total_new_boxes = []
    total_new_seg_data = []
    
    # Pre-compute directions once
    directions = np.array([
        [-1,0,0], [1,0,0],  # X
        [0,-1,0], [0,1,0],  # Y
        [0,0,-1], [0,0,1]   # Z
    ])

    for iteration in range(max_iterations):
        # Vectorized face content check - already efficient
        face_means = np.array([
            all_seg_data[:,0,:,:].mean(axis=(1,2)),   # -X
            all_seg_data[:,-1,:,:].mean(axis=(1,2)),  # +X  
            all_seg_data[:,:,0,:].mean(axis=(1,2)),   # -Y
            all_seg_data[:,:,-1,:].mean(axis=(1,2)),  # +Y
            all_seg_data[:,:,:,0].mean(axis=(1,2)),   # -Z
            all_seg_data[:,:,:,-1].mean(axis=(1,2))   # +Z
        ])

        significant_faces = face_means > boundary_threshold
        if not significant_faces.any():
            break

        # Pre-compute box centers and tree once per iteration
        box_centers = (all_boxes[:, 0, :] + all_boxes[:, 1, :]) / 2
        tree = KDTree(box_centers)
        new_boxes = []

        # Process boxes with any significant faces
        active_boxes = np.where(significant_faces.any(axis=0))[0]
        
        for i in active_boxes:
            box_min = all_boxes[i, 0]
            box_faces = np.where(significant_faces[:,i])[0]
            
            for face_idx in box_faces:
                new_min = box_min + directions[face_idx] * (tile_size - min_overlap)
                new_max = new_min + tile_size
                
                if not (np.all(new_min >= min_bounds) and 
                       np.all(new_max <= max_bounds) and 
                       np.all(new_max - new_min >= 20)):
                    continue

                new_center = (new_min + new_max) / 2
                total_vol = np.prod(new_max - new_min)
                
                # Get neighbors efficiently
                neighbors = tree.query_ball_point(new_center, tile_size * 1.1)
                if len(neighbors) <= 1:
                    new_boxes.append([new_min, new_max])
                    continue

                all_intersections = []
                
                # Check intersections with existing boxes
                neighbor_boxes = all_boxes[neighbors]
                intersections = np.minimum(new_max, neighbor_boxes[:,1]) - \
                              np.maximum(new_min, neighbor_boxes[:,0])
                valid_intersections = np.all(intersections > 0, axis=1)
                valid_intersections[neighbors.index(i)] = False
                
                if valid_intersections.any():
                    volumes = np.prod(intersections[valid_intersections], axis=1)
                    all_intersections.extend(volumes)

                # Check new boxes if any found
                if new_boxes:
                    new_boxes_array = np.array(new_boxes)
                    intersections = np.minimum(new_max, new_boxes_array[:,1]) - \
                                  np.maximum(new_min, new_boxes_array[:,0])
                    valid_intersections = np.all(intersections > 0, axis=1)
                    
                    if valid_intersections.any():
                        volumes = np.prod(intersections[valid_intersections], axis=1)
                        all_intersections.extend(volumes)

                # Check coverage exactly as in original
                if all_intersections:
                    all_intersections.sort(reverse=True)
                    covered_vol = 0
                    unique_vol = total_vol

                    for vol in all_intersections:
                        new_coverage = min(vol, unique_vol)
                        covered_vol += new_coverage
                        unique_vol -= new_coverage
                        
                        if covered_vol >= total_vol * 0.9999:
                            break
                    
                    if covered_vol >= total_vol * 0.9999:
                        continue

                new_boxes.append([new_min, new_max])

        if len(new_boxes) == 0:
            break
            
        print(f"Iteration {iteration + 1}: Adding {len(new_boxes)} boxes")
        new_boxes = np.array(new_boxes)
        new_seg_data = voxel_utils.get_subvols_batched(new_boxes, 'segmentation') == bodyId
        
        total_new_boxes.append(new_boxes)
        total_new_seg_data.append(new_seg_data)
        
        all_boxes = np.concatenate([all_boxes, new_boxes])
        all_seg_data = np.concatenate([all_seg_data, new_seg_data])

    if total_new_boxes:
        return all_boxes, all_seg_data
    else:
        return (filtered_boxes, filtered_seg_data,
                np.zeros((0, 2, 3), dtype=filtered_boxes.dtype),
                np.zeros((0, tile_size, tile_size, tile_size), dtype=bool))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bodyId', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    bodyId = args.bodyId
    output_dir = args.output_dir
    
    try:
        print(f"Processing bodyId: {bodyId}")
        
        # Get neuron data
        neuron = fetch_skeleton(bodyId)
        neuron = neuron.astype({'x': int, 'y': int, 'z': int})
        
        # Generate boxes
        boxes = generate_optimized_boxes(neuron)
        seg_data = voxel_utils.get_subvols_batched(boxes, 'segmentation') == bodyId
        boxes, seg_data = filter_empty_boxes(boxes, seg_data)
    
        boxes, seg_data = find_uncovered_boundaries(boxes, seg_data)
    
        # Save result
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(f"{output_dir}/boxes_{bodyId}.npy", boxes)
        np.save(f"{output_dir}/seg_data_{bodyId}.npy", seg_data)
        print(f"Saved {len(boxes)} boxes for bodyId {bodyId}")
        
    except Exception as e:
        print(f"Error processing bodyId {bodyId}: {str(e)}")
        raise





