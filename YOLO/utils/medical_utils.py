"""
Medical image processing utilities for vertebrae YOLO implementation.
"""

import numpy as np
import nibabel as nib
import cv2
from typing import Tuple, List, Optional
import os
from pathlib import Path


def load_nifti(file_path: str) -> Tuple[np.ndarray, nib.Nifti1Header]:
    """
    Load NIfTI file and return image data and header.
    
    Args:
        file_path: Path to NIfTI file
        
    Returns:
        Tuple of (image_data, header)
    """
    nii = nib.load(file_path)
    return nii.get_fdata(), nii.header


def load_cut_coordinates(cut_file: str) -> List[Tuple[int, int, int, int, int, int]]:
    """
    Load vertebrae cut coordinates from text file.
    
    Args:
        cut_file: Path to cut_li*.txt file
        
    Returns:
        List of (x_min, x_max, y_min, y_max, z_min, z_max) tuples
    """
    coordinates = []
    with open(cut_file, 'r') as f:
        for line in f:
            if line.strip():
                coords = list(map(int, line.strip().split()))
                if len(coords) == 6:
                    coordinates.append(tuple(coords))
    return coordinates


def extract_vertebrae_region(volume: np.ndarray, 
                           coordinates: Tuple[int, int, int, int, int, int]) -> np.ndarray:
    """
    Extract vertebrae region from 3D volume using coordinates.
    
    Args:
        volume: 3D CT volume
        coordinates: (x_min, x_max, y_min, y_max, z_min, z_max)
        
    Returns:
        Extracted vertebrae region
    """
    x_min, x_max, y_min, y_max, z_min, z_max = coordinates
    return volume[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]


def normalize_hounsfield(image: np.ndarray, 
                        window_center: int = 40, 
                        window_width: int = 400) -> np.ndarray:
    """
    Normalize Hounsfield units to 0-255 range with windowing.
    
    Args:
        image: Input image in Hounsfield units
        window_center: Window center value
        window_width: Window width value
        
    Returns:
        Normalized image in 0-255 range
    """
    window_min = window_center - window_width // 2
    window_max = window_center + window_width // 2
    
    # Apply windowing
    image_windowed = np.clip(image, window_min, window_max)
    
    # Normalize to 0-255
    image_normalized = (image_windowed - window_min) / (window_max - window_min) * 255
    
    return image_normalized.astype(np.uint8)


def create_axial_slices(volume: np.ndarray, 
                       slice_thickness: int = 1) -> List[np.ndarray]:
    """
    Create axial slices from 3D volume.
    
    Args:
        volume: 3D CT volume
        slice_thickness: Thickness of each slice
        
    Returns:
        List of 2D axial slices
    """
    slices = []
    for z in range(0, volume.shape[2], slice_thickness):
        if z + slice_thickness <= volume.shape[2]:
            if slice_thickness == 1:
                slice_2d = volume[:, :, z]
            else:
                slice_2d = np.mean(volume[:, :, z:z+slice_thickness], axis=2)
            slices.append(slice_2d)
    return slices


def resize_slice(slice_2d: np.ndarray, 
                target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Resize 2D slice to target size.
    
    Args:
        slice_2d: 2D image slice
        target_size: Target size (width, height)
        
    Returns:
        Resized slice
    """
    return cv2.resize(slice_2d, target_size, interpolation=cv2.INTER_LINEAR)


def get_patient_files(input_dir: str, patient_id: str) -> dict:
    """
    Get all files for a specific patient.
    
    Args:
        input_dir: Directory containing NIfTI files
        patient_id: Patient ID (e.g., "1003")
        
    Returns:
        Dictionary with file paths
    """
    files = {}
    input_path = Path(input_dir)
    
    # Find input CT file
    inp_files = list(input_path.glob(f"inp{patient_id}.nii*"))
    if inp_files:
        files['input'] = str(inp_files[0])
    
    # Find segmentation file
    seg_files = list(input_path.glob(f"seg{patient_id}.nii*"))
    if seg_files:
        files['segmentation'] = str(seg_files[0])
    
    # Find answer file
    ans_files = list(input_path.glob(f"ans{patient_id}.nii*"))
    if ans_files:
        files['answer'] = str(ans_files[0])
    
    # Find cut coordinates file
    cut_files = list(input_path.glob(f"cut_li{patient_id}.txt"))
    if cut_files:
        files['cut_coordinates'] = str(cut_files[0])
    
    return files


def create_yolo_annotation(bbox: Tuple[int, int, int, int], 
                          class_id: int, 
                          img_width: int, 
                          img_height: int) -> str:
    """
    Create YOLO format annotation string.
    
    Args:
        bbox: Bounding box (x_min, y_min, x_max, y_max)
        class_id: Class ID
        img_width: Image width
        img_height: Image height
        
    Returns:
        YOLO format annotation string
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Convert to YOLO format (center_x, center_y, width, height) normalized
    center_x = (x_min + x_max) / 2 / img_width
    center_y = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"