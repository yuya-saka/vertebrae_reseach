# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a vertebrae fracture detection research project with three main components:

1. **Sakaguchi_file**: ResNet18-based CNN approach for vertebrae fracture detection using axial CT slices
2. **prior_YOLO_file**: Previous YOLO implementation for vertebrae detection
3. **YOLO**: New PyTorch-based YOLOv8 implementation for vertebrae detection and fracture classification

## Common Development Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Activate development environment
uv run python -m ipykernel install --user --name=vertebrae
```

### Jupyter Lab
```bash
# Start Jupyter Lab for notebook development
jupyter lab

# Run specific notebook
jupyter nbconvert --to notebook --execute Sakaguchi_file/S_model_learning/resnet18_axial_learning.ipynb
```

### Data Processing (Sakaguchi_file)
```bash
# Data partitioning
python Sakaguchi_file/S_partitoning/S_trainval_partitioning.py

# Data cutting (extract vertebrae regions)
python Sakaguchi_file/S_data_cut/S_cut_train.py

# Create axial slices
python Sakaguchi_file/S_data_slice/S_slice_axial.py
```

### Model Training
```bash
# Train ResNet18 model
python Sakaguchi_file/S_model_learning/resnet18_axial_learning.py

# Train with online augmentation
python Sakaguchi_file/S_model_learning/resnet18_axial_learning_online_aug_1.py
```

### YOLO Development (Future)
```bash
# Install YOLOv8 dependencies
uv add ultralytics

# Train YOLO model
python YOLO/training/train_yolo.py

# Run inference
python YOLO/inference/predict.py
```

## Project Architecture

### Data Flow
1. **Input Data**: NIfTI files in `input_nii/`
   - CT scans: `inp*.nii.gz`
   - Segmentation masks: `seg*.nii`
   - Answer labels: `ans*.nii`
   - Cut coordinates: `cut_li*.txt`

2. **Data Processing Pipeline**:
   - **Partitioning**: Split into train/val/test sets
   - **Cutting**: Extract vertebrae regions using coordinates
   - **Slicing**: Create 2D slices from 3D volumes
   - **Augmentation**: Apply data augmentation

3. **Model Training**: Train classification models on processed slices

### Directory Structure

#### Sakaguchi_file (Current Implementation)
- `S_partitoning/`: Data splitting scripts
- `S_data_cut/`: Vertebrae region extraction
- `S_data_slice/`: 2D slice creation (axial/coronal)
- `S_data_augmentation/`: Data augmentation
- `S_model_learning/`: ResNet18 training scripts and notebooks
- `S_test/`, `S_train/`, `S_val/`: Split datasets

#### prior_YOLO_file (Previous Implementation)
- `MkSlice/`: Slice creation and normalization
- `Integrate/`: Result integration
- `Prediction/`: Model inference
- `training/`: YOLO training pipeline

#### YOLO (New Implementation)
- `計画書.md`: Development plan for PyTorch-based YOLOv8
- Future structure: data/, models/, training/, inference/, utils/

## Key Dependencies

### Core Libraries
- **PyTorch**: Deep learning framework with CUDA support
- **nibabel**: NIfTI medical image file handling
- **numpy**: Numerical operations
- **pandas**: Data manipulation
- **matplotlib**: Visualization

### Medical Image Processing
- **scikit-image**: Image processing
- **opencv-python**: Computer vision
- **scipy**: Scientific computing
- **SimpleITK**: Medical image analysis

### Machine Learning
- **scikit-learn**: Machine learning utilities
- **torch**: PyTorch deep learning
- **torchvision**: Computer vision models
- **wandb**: Experiment tracking

### Future YOLO Dependencies
- **ultralytics**: YOLOv8 implementation
- **albumentations**: Advanced data augmentation

## Medical Data Format

### NIfTI Files (.nii, .nii.gz)
- **Input CT**: `inp{patient_id}.nii.gz` - Original CT scans
- **Segmentation**: `seg{patient_id}.nii` - Vertebrae segmentation masks
- **Answer**: `ans{patient_id}.nii` - Fracture labels
- **Cut List**: `cut_li{patient_id}.txt` - Vertebrae bounding box coordinates

### Data Organization
- **Training**: 24 patients
- **Validation**: 6 patients  
- **Testing**: 8 patients
- **Slices**: Axial and coronal 2D slices extracted from 3D volumes

## Development Workflow

### 1. Data Preparation
1. Place raw NIfTI files in `input_nii/`
2. Run partitioning scripts to split data
3. Extract vertebrae regions using cut coordinates
4. Create 2D slices for training

### 2. Model Development
1. Use Jupyter notebooks for experimentation
2. Train models using PyTorch
3. Track experiments with Weights & Biases
4. Save best models in `S_model_learning/model_pth/`

### 3. Evaluation
1. Evaluate on test set
2. Compare with existing methods
3. Generate performance metrics (AUC, sensitivity, specificity)

## Important Notes

### Medical Image Considerations
- Images are in Hounsfield Units (HU)
- Proper windowing and normalization required
- Vertebrae segmentation used for region extraction
- Binary classification: fracture vs. non-fracture

### GPU Usage
- CUDA 12.1 support configured
- Models trained on GPU for faster training
- CuPy used for GPU-accelerated NumPy operations

### Experiment Tracking
- Weights & Biases integration in training scripts
- Model checkpoints saved regularly
- Hyperparameter sweeps for optimization

## Future Development

### YOLO Implementation
- Migrate to YOLOv8 for object detection approach
- Combine detection and classification in single model
- Compare performance with current ResNet18 approach

### Performance Optimization
- Implement ensemble methods
- Optimize inference speed
- Deploy models for clinical use

This project focuses on automated vertebrae fracture detection using deep learning, with emphasis on medical image processing and clinical applicability.