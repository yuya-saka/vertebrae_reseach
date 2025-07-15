#!/usr/bin/env python3
"""
Data preprocessing script for YOLO vertebrae fracture detection.
"""

import argparse
import yaml
from pathlib import Path
import sys
import json
from datetime import datetime

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

from utils.data_prep import YOLODatasetPreprocessor


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Preprocess medical images for YOLO training')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing NIfTI files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for YOLO dataset')
    parser.add_argument('--config', type=str, 
                        default='configs/dataset.yaml',
                        help='Path to dataset configuration file')
    parser.add_argument('--target_size', type=int, nargs=2, 
                        default=[640, 640],
                        help='Target image size (width height)')
    parser.add_argument('--window_center', type=int, default=40,
                        help='CT windowing center')
    parser.add_argument('--window_width', type=int, default=400,
                        help='CT windowing width')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
        print(f"Loaded configuration from {config_path}")
    else:
        print(f"Warning: Configuration file {config_path} not found. Using defaults.")
        config = {}
    
    # Override config with command line arguments
    target_size = tuple(args.target_size)
    window_center = args.window_center
    window_width = args.window_width
    
    # Update from config file if available
    if 'image_size' in config:
        target_size = (config['image_size'], config['image_size'])
    if 'window_center' in config:
        window_center = config['window_center']
    if 'window_width' in config:
        window_width = config['window_width']
    
    print(f"Preprocessing configuration:")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Target size: {target_size}")
    print(f"  Window center: {window_center}")
    print(f"  Window width: {window_width}")
    print()
    
    # Create preprocessor
    preprocessor = YOLODatasetPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=target_size,
        window_center=window_center,
        window_width=window_width
    )
    
    # Process all patients
    print("Starting data preprocessing...")
    start_time = datetime.now()
    
    results = preprocessor.process_all_patients()
    
    end_time = datetime.now()
    processing_time = end_time - start_time
    
    # Print results
    print(f"\nPreprocessing completed in {processing_time}")
    print(f"Total patients processed: {results['total_patients']}")
    print(f"Statistics:")
    for split, count in results['statistics'].items():
        print(f"  {split}: {count} slices")
    
    # Save detailed results
    results_file = Path(args.output_dir) / 'preprocessing_results.json'
    with open(results_file, 'w') as f:
        # Convert datetime objects to strings for JSON serialization
        json_results = results.copy()
        json_results['start_time'] = start_time.isoformat()
        json_results['end_time'] = end_time.isoformat()
        json_results['processing_time'] = str(processing_time)
        json.dump(json_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Print any errors
    errors = [r for r in results['results'] if r['status'] == 'error']
    if errors:
        print(f"\nErrors encountered:")
        for error in errors:
            print(f"  Patient {error.get('patient_id', 'unknown')}: {error.get('message', 'unknown error')}")
    
    # Print success summary
    successful = [r for r in results['results'] if r['status'] == 'success']
    if successful:
        print(f"\nSuccessfully processed {len(successful)} patients")
        if args.verbose:
            print("Details:")
            for result in successful:
                print(f"  Patient {result['patient_id']}: {result['processed_slices']} slices ({result['split']})")
    
    # Create dataset info file
    dataset_info = {
        'created': datetime.now().isoformat(),
        'source': args.input_dir,
        'total_patients': results['total_patients'],
        'splits': results['statistics'],
        'preprocessing_config': {
            'target_size': target_size,
            'window_center': window_center,
            'window_width': window_width,
        }
    }
    
    info_file = Path(args.output_dir) / 'dataset_info.json'
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Dataset information saved to: {info_file}")
    print(f"Dataset configuration saved to: {Path(args.output_dir) / 'dataset.yaml'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())