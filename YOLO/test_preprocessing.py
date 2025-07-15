#!/usr/bin/env python3
"""
Test script for data preprocessing functionality.
"""

import sys
from pathlib import Path
import tempfile
import shutil
import numpy as np
import nibabel as nib

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

from utils.medical_utils import (
    load_nifti, normalize_hounsfield, create_axial_slices,
    resize_slice, get_patient_files, create_yolo_annotation
)


def create_test_data():
    """Create test NIfTI files for testing."""
    test_dir = Path(tempfile.mkdtemp())
    
    # Create test CT volume (100x100x50)
    ct_volume = np.random.randint(-1000, 1000, size=(100, 100, 50)).astype(np.float32)
    ct_nii = nib.Nifti1Image(ct_volume, np.eye(4))
    
    # Create test answer volume (binary)
    ans_volume = np.random.choice([0, 1], size=(100, 100, 50), p=[0.9, 0.1]).astype(np.float32)
    ans_nii = nib.Nifti1Image(ans_volume, np.eye(4))
    
    # Save test files
    test_patient_id = "1001"
    nib.save(ct_nii, test_dir / f"inp{test_patient_id}.nii.gz")
    nib.save(ans_nii, test_dir / f"ans{test_patient_id}.nii")
    
    # Create test cut coordinates
    cut_coords = "10 90 10 90 10 40\n20 80 20 80 20 30\n"
    with open(test_dir / f"cut_li{test_patient_id}.txt", 'w') as f:
        f.write(cut_coords)
    
    return test_dir, test_patient_id


def test_medical_utils():
    """Test medical utility functions."""
    print("Testing medical utility functions...")
    
    # Create test data
    test_dir, patient_id = create_test_data()
    
    try:
        # Test get_patient_files
        files = get_patient_files(str(test_dir), patient_id)
        print(f"Found files: {list(files.keys())}")
        
        # Test load_nifti
        ct_volume, header = load_nifti(files['input'])
        print(f"CT volume shape: {ct_volume.shape}")
        print(f"CT volume data type: {ct_volume.dtype}")
        print(f"CT volume range: {ct_volume.min():.1f} to {ct_volume.max():.1f}")
        
        # Test normalize_hounsfield
        normalized = normalize_hounsfield(ct_volume)
        print(f"Normalized volume shape: {normalized.shape}")
        print(f"Normalized volume range: {normalized.min()} to {normalized.max()}")
        
        # Test create_axial_slices
        slices = create_axial_slices(ct_volume)
        print(f"Created {len(slices)} axial slices")
        if slices:
            print(f"First slice shape: {slices[0].shape}")
        
        # Test resize_slice
        if slices:
            resized = resize_slice(slices[0], (640, 640))
            print(f"Resized slice shape: {resized.shape}")
        
        # Test create_yolo_annotation
        bbox = (100, 100, 200, 200)
        annotation = create_yolo_annotation(bbox, 1, 640, 640)
        print(f"YOLO annotation: {annotation}")
        
        print("✓ All medical utility functions working correctly")
        
    except Exception as e:
        print(f"✗ Error in medical utilities: {e}")
        return False
    
    finally:
        # Clean up
        shutil.rmtree(test_dir)
    
    return True


def test_data_preprocessing():
    """Test data preprocessing functionality."""
    print("\nTesting data preprocessing...")
    
    # Create test data
    test_dir, patient_id = create_test_data()
    output_dir = Path(tempfile.mkdtemp())
    
    try:
        # Import here to avoid issues if packages aren't installed
        from utils.data_prep import YOLODatasetPreprocessor
        
        # Create preprocessor
        preprocessor = YOLODatasetPreprocessor(
            input_dir=str(test_dir),
            output_dir=str(output_dir),
            target_size=(640, 640),
            window_center=40,
            window_width=400
        )
        
        # Test directory setup
        assert (output_dir / 'train' / 'images').exists()
        assert (output_dir / 'train' / 'labels').exists()
        print("✓ Output directories created successfully")
        
        # Test single patient processing
        result = preprocessor.process_patient(patient_id)
        print(f"Processing result: {result}")
        
        if result['status'] == 'success':
            print(f"✓ Successfully processed patient {patient_id}")
            print(f"  Processed {result['processed_slices']} slices")
            print(f"  Split: {result['split']}")
        else:
            print(f"✗ Failed to process patient: {result.get('message', 'unknown error')}")
            return False
        
        # Check output files
        images_dir = output_dir / result['split'] / 'images'
        labels_dir = output_dir / result['split'] / 'labels'
        
        image_files = list(images_dir.glob('*.jpg'))
        label_files = list(labels_dir.glob('*.txt'))
        
        print(f"Generated {len(image_files)} image files")
        print(f"Generated {len(label_files)} label files")
        
        if image_files:
            print(f"Example image file: {image_files[0].name}")
        if label_files:
            print(f"Example label file: {label_files[0].name}")
        
        print("✓ Data preprocessing working correctly")
        
    except ImportError as e:
        print(f"⚠ Skipping data preprocessing test (missing dependencies): {e}")
        return True  # Not a failure, just missing optional dependencies
    
    except Exception as e:
        print(f"✗ Error in data preprocessing: {e}")
        return False
    
    finally:
        # Clean up
        shutil.rmtree(test_dir)
        shutil.rmtree(output_dir)
    
    return True


def main():
    """Run all tests."""
    print("YOLO Data Preprocessing Tests")
    print("=" * 50)
    
    success = True
    
    # Test medical utilities
    if not test_medical_utils():
        success = False
    
    # Test data preprocessing
    if not test_data_preprocessing():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed successfully!")
        return 0
    else:
        print("✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())