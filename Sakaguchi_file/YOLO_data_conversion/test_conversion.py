#!/usr/bin/env python3
"""
データ変換の小規模テストスクリプト
"""

import pandas as pd
import numpy as np
from pathlib import Path
from nifti_to_yolo import NiftiToYoloConverter

def test_conversion():
    """小規模なデータ変換テスト"""
    
    print("=== データ変換テスト開始 ===")
    
    # 変換器を初期化
    converter = NiftiToYoloConverter(
        input_dir="/mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/Sakaguchi_file",
        output_dir="/mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/Sakaguchi_file/YOLO_test",
        image_size=640
    )
    
    # 訓練データのCSVを読み込み
    csv_path = Path("/mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/Sakaguchi_file/slice_train/axial/train_labels_axial.csv")
    df = pd.read_csv(csv_path)
    
    # 骨折ラベル1のサンプル5つを取得
    fracture_samples = df[df['Fracture_Label'] == 1].head(5)
    
    print(f"テスト対象サンプル数: {len(fracture_samples)}")
    
    successful_conversions = 0
    total_bboxes = 0
    
    for idx, row in fracture_samples.iterrows():
        print(f"\n--- サンプル {idx + 1} ---")
        print(f"患者ID: {row['Case']}")
        print(f"椎体: {row['Vertebra']}")
        print(f"スライス: {row['SliceIndex']}")
        
        # 変換処理
        success = converter.process_slice(row.to_dict())
        
        if success:
            successful_conversions += 1
            print("✓ 変換成功")
            
            # 生成されたラベルファイルを確認
            patient_id = str(row['Case'])
            vertebra = str(row['Vertebra'])
            slice_idx = row['SliceIndex']
            output_filename = f"{patient_id}_{vertebra}_{slice_idx:03d}"
            
            label_path = converter.output_dir / 'train' / 'labels' / f"{output_filename}.txt"
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    bbox_count = len(lines)
                    total_bboxes += bbox_count
                    print(f"  バウンディングボックス数: {bbox_count}")
                    for line in lines:
                        print(f"  BBox: {line.strip()}")
            else:
                print("  ❌ ラベルファイルが見つかりません")
        else:
            print("❌ 変換失敗")
    
    print(f"\n=== テスト結果 ===")
    print(f"成功した変換: {successful_conversions}/{len(fracture_samples)}")
    print(f"総バウンディングボックス数: {total_bboxes}")
    print(f"平均バウンディングボックス数: {total_bboxes/successful_conversions if successful_conversions > 0 else 0:.2f}")
    
    # 統計情報を出力
    converter.print_dataset_stats()

if __name__ == "__main__":
    test_conversion()