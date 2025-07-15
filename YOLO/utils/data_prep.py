"""
Data preprocessing utilities for YOLO training.
"""

import os
import shutil
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .medical_utils import (
    load_nifti, load_cut_coordinates, extract_vertebrae_region,
    normalize_hounsfield, create_axial_slices, resize_slice,
    get_patient_files, create_yolo_annotation
)


class YOLODatasetPreprocessor:
    """
    Preprocessor for converting medical images to YOLO format.
    """
    
    def __init__(self, 
                 input_dir: str,
                 output_dir: str,
                 target_size: Tuple[int, int] = (640, 640),
                 window_center: int = 40,
                 window_width: int = 400):
        """
        Initialize preprocessor.
        
        Args:
            input_dir: Directory containing NIfTI files
            output_dir: Output directory for YOLO dataset
            target_size: Target image size (width, height)
            window_center: CT windowing center
            window_width: CT windowing width
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.window_center = window_center
        self.window_width = window_width
        
        # Create output directories
        self.setup_output_directories()
    
    def setup_output_directories(self):
        """Create YOLO dataset directory structure."""
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def get_patient_split(self, patient_id: str) -> str:
        """
        Determine which split a patient belongs to based on existing splits.
        
        Args:
            patient_id: Patient ID
            
        Returns:
            Split name ('train', 'val', or 'test')
        """
        # Check existing split directories
        base_dir = self.input_dir.parent
        
        if (base_dir / 'Sakaguchi_file' / 'S_train').exists():
            train_files = list((base_dir / 'Sakaguchi_file' / 'S_train').glob(f"inp{patient_id}.nii*"))
            if train_files:
                return 'train'
        
        if (base_dir / 'Sakaguchi_file' / 'S_val').exists():
            val_files = list((base_dir / 'Sakaguchi_file' / 'S_val').glob(f"inp{patient_id}.nii*"))
            if val_files:
                return 'val'
        
        if (base_dir / 'Sakaguchi_file' / 'S_test').exists():
            test_files = list((base_dir / 'Sakaguchi_file' / 'S_test').glob(f"inp{patient_id}.nii*"))
            if test_files:
                return 'test'
        
        # Default to train if not found
        return 'train'
    
    def process_patient(self, patient_id: str) -> Dict:
        """
        Process a single patient's data.
        
        Args:
            patient_id: Patient ID
            
        Returns:
            Processing statistics
        """
        # Get patient files
        files = get_patient_files(str(self.input_dir), patient_id)
        
        if not all(key in files for key in ['input', 'answer', 'cut_coordinates']):
            return {'status': 'error', 'message': 'Missing required files'}
        
        # Load data
        ct_volume, _ = load_nifti(files['input'])
        ans_volume, _ = load_nifti(files['answer'])
        coordinates = load_cut_coordinates(files['cut_coordinates'])
        
        # Determine split
        split = self.get_patient_split(patient_id)
        
        processed_count = 0
        
        # Process each vertebrae
        for i, coords in enumerate(coordinates):
            try:
                # Extract vertebrae region
                vertebrae_ct = extract_vertebrae_region(ct_volume, coords)
                vertebrae_ans = extract_vertebrae_region(ans_volume, coords)
                
                # Create axial slices
                ct_slices = create_axial_slices(vertebrae_ct)
                ans_slices = create_axial_slices(vertebrae_ans)
                
                # Process each slice
                for j, (ct_slice, ans_slice) in enumerate(zip(ct_slices, ans_slices)):
                    # Normalize and resize
                    ct_normalized = normalize_hounsfield(ct_slice, 
                                                       self.window_center, 
                                                       self.window_width)
                    ct_resized = resize_slice(ct_normalized, self.target_size)
                    
                    # Check if slice has fracture
                    has_fracture = np.any(ans_slice > 0)
                    
                    # Create image filename
                    img_filename = f"{patient_id}_vertebrae_{i:02d}_slice_{j:03d}.jpg"
                    img_path = self.output_dir / split / 'images' / img_filename
                    
                    # Save image
                    cv2.imwrite(str(img_path), ct_resized)
                    
                    # Create annotation if fracture exists
                    if has_fracture:
                        # For now, create a simple bounding box around the entire image
                        # This should be improved with actual fracture localization
                        bbox = (50, 50, self.target_size[0]-50, self.target_size[1]-50)
                        class_id = 1  # fracture class
                        
                        annotation = create_yolo_annotation(bbox, class_id, 
                                                          self.target_size[0], 
                                                          self.target_size[1])
                        
                        # Save annotation
                        label_filename = f"{patient_id}_vertebrae_{i:02d}_slice_{j:03d}.txt"
                        label_path = self.output_dir / split / 'labels' / label_filename
                        
                        with open(label_path, 'w') as f:
                            f.write(annotation + '\n')
                    else:
                        # Create empty annotation file for non-fracture slices
                        label_filename = f"{patient_id}_vertebrae_{i:02d}_slice_{j:03d}.txt"
                        label_path = self.output_dir / split / 'labels' / label_filename
                        
                        with open(label_path, 'w') as f:
                            pass  # Empty file
                    
                    processed_count += 1
                    
            except Exception as e:
                print(f"Error processing vertebrae {i} for patient {patient_id}: {e}")
                continue
        
        return {
            'status': 'success',
            'patient_id': patient_id,
            'split': split,
            'processed_slices': processed_count
        }
    
    def create_dataset_yaml(self):
        """Create YOLO dataset configuration file."""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 2,  # number of classes
            'names': ['normal', 'fracture']
        }
        
        with open(self.output_dir / 'dataset.yaml', 'w') as f:
            json.dump(config, f, indent=2)
    
    def process_all_patients(self) -> Dict:
        """
        Process all patients in the input directory.
        
        Returns:
            Processing statistics
        """
        # Find all patient IDs
        patient_ids = set()
        for file_path in self.input_dir.glob("inp*.nii*"):
            # Extract patient ID from filename
            filename = file_path.stem.replace('.nii', '')
            patient_id = filename.replace('inp', '')
            patient_ids.add(patient_id)
        
        results = []
        stats = {'train': 0, 'val': 0, 'test': 0, 'errors': 0}
        
        print(f"Processing {len(patient_ids)} patients...")
        
        for patient_id in tqdm(sorted(patient_ids)):
            result = self.process_patient(patient_id)
            results.append(result)
            
            if result['status'] == 'success':
                stats[result['split']] += result['processed_slices']
            else:
                stats['errors'] += 1
        
        # Create dataset configuration
        self.create_dataset_yaml()
        
        return {
            'results': results,
            'statistics': stats,
            'total_patients': len(patient_ids)
        }


def get_training_transforms(image_size: int = 640) -> A.Compose:
    """
    Get training data augmentation transforms.
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussianBlur(p=0.2),
        A.RandomGamma(p=0.2),
        A.OneOf([
            A.ElasticTransform(p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=0.5),
        ], p=0.3),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))


def get_validation_transforms(image_size: int = 640) -> A.Compose:
    """
    Get validation data transforms.
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))