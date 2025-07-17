#!/usr/bin/env python3
"""
骨折マスクの読み込みをデバッグするスクリプト
"""

import pandas as pd
import nibabel as nib
import numpy as np
from pathlib import Path

def debug_fracture_mask():
    """骨折マスクの読み込みをデバッグ"""
    
    # 骨折ラベル1のサンプルを取得
    csv_path = Path("/mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/Sakaguchi_file/slice_train/axial/train_labels_axial.csv")
    df = pd.read_csv(csv_path)
    
    # 骨折ラベル1のデータをフィルタ
    fracture_samples = df[df['Fracture_Label'] == 1]
    
    print(f"骨折ラベル1のサンプル数: {len(fracture_samples)}")
    print("最初の5つのサンプル:")
    print(fracture_samples.head())
    
    # 最初のサンプルでマスクを確認
    if len(fracture_samples) > 0:
        sample = fracture_samples.iloc[0]
        patient_id = str(sample['Case'])
        vertebra = str(sample['Vertebra'])
        slice_idx = sample['SliceIndex']
        
        print(f"\n=== サンプル詳細 ===")
        print(f"患者ID: {patient_id}")
        print(f"椎体: {vertebra}")
        print(f"スライス: {slice_idx}")
        print(f"パス: {sample['FullPath']}")
        
        # 骨折マスクパスを構築
        base_dir = Path("/mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/Sakaguchi_file")
        mask_path = base_dir / f"processed_train" / f"inp{patient_id}" / vertebra / f"cut_ans{patient_id}.nii"
        
        print(f"\n骨折マスクパス: {mask_path}")
        print(f"マスクファイル存在: {mask_path.exists()}")
        
        if mask_path.exists():
            # マスクを読み込み
            mask_nifti = nib.load(str(mask_path))
            mask_data = mask_nifti.get_fdata()
            
            print(f"マスクデータ形状: {mask_data.shape}")
            print(f"スライス数: {mask_data.shape[2]}")
            print(f"対象スライス: {slice_idx}")
            
            if slice_idx < mask_data.shape[2]:
                slice_mask = mask_data[:, :, slice_idx]
                print(f"スライスマスク形状: {slice_mask.shape}")
                print(f"スライスマスク最大値: {np.max(slice_mask)}")
                print(f"スライスマスク最小値: {np.min(slice_mask)}")
                print(f"非零ピクセル数: {np.sum(slice_mask > 0)}")
                
                # 骨折領域の座標を確認
                if np.sum(slice_mask > 0) > 0:
                    y_coords, x_coords = np.where(slice_mask > 0)
                    print(f"骨折領域座標範囲:")
                    print(f"  X: {x_coords.min()} - {x_coords.max()}")
                    print(f"  Y: {y_coords.min()} - {y_coords.max()}")
                else:
                    print("❌ 骨折領域が見つかりません")
            else:
                print("❌ スライスインデックスが範囲外")
        else:
            print("❌ マスクファイルが存在しません")

if __name__ == "__main__":
    debug_fracture_mask()