#!/usr/bin/env python3
"""
NIfTI形式の椎体スライスデータをYOLO形式に変換するスクリプト

このスクリプトは以下の処理を行います：
1. Sakaguchi_fileのaxialスライスデータを読み込み
2. 対応する骨折マスクからバウンディングボックスを抽出
3. YOLO形式（画像+アノテーション）に変換
4. 適切なディレクトリ構造で保存

使用方法:
    python nifti_to_yolo.py --input_dir /path/to/slice_data --output_dir /path/to/yolo_data
"""

import argparse
import pandas as pd
import nibabel as nib
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nifti_to_yolo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NiftiToYoloConverter:
    """NIfTI形式データをYOLO形式に変換するクラス"""
    
    def __init__(self, input_dir: str, output_dir: str, image_size: int = 640):
        """
        Args:
            input_dir: 入力データディレクトリ（Sakaguchi_fileのsliceディレクトリ）
            output_dir: 出力YOLO形式データディレクトリ
            image_size: YOLO用画像サイズ（デフォルト640x640）
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        
        # 出力ディレクトリ構造を作成
        self.setup_output_dirs()
        
        # HUウィンドウ設定（骨用）
        self.hu_window = (100, 2000)
        
    def setup_output_dirs(self):
        """YOLO形式の出力ディレクトリ構造を作成"""
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # データセット設定ファイル用ディレクトリ
        (self.output_dir / 'configs').mkdir(parents=True, exist_ok=True)
        
        logger.info(f"出力ディレクトリ構造を作成: {self.output_dir}")
    
    def normalize_hounsfield(self, image: np.ndarray) -> np.ndarray:
        """HU値を0-255に正規化"""
        # 骨窓での正規化
        img_windowed = np.clip(image, self.hu_window[0], self.hu_window[1])
        img_normalized = (img_windowed - self.hu_window[0]) / (self.hu_window[1] - self.hu_window[0]) * 255
        return img_normalized.astype(np.uint8)
    
    def load_nifti_slice(self, slice_path: Path) -> np.ndarray:
        """NIfTIスライスを読み込み、正規化"""
        try:
            nifti_obj = nib.load(str(slice_path))
            image_data = nifti_obj.get_fdata()
            
            # 2D画像に変換
            if image_data.ndim == 3:
                image_data = np.squeeze(image_data)
            
            # HU値正規化
            normalized_image = self.normalize_hounsfield(image_data)
            
            return normalized_image
            
        except Exception as e:
            logger.error(f"NIfTIスライス読み込みエラー {slice_path}: {e}")
            return None
    
    def get_fracture_mask(self, patient_id: str, vertebra: str, slice_idx: int) -> Optional[np.ndarray]:
        """対応する骨折マスクを取得"""
        try:
            # processed_*ディレクトリから対応する骨折マスクを読み込み
            for split in ['train', 'val', 'test']:
                mask_path = self.input_dir / f"processed_{split}" / f"inp{patient_id}" / vertebra / f"cut_ans{patient_id}.nii"
                if mask_path.exists():
                    mask_nifti = nib.load(str(mask_path))
                    mask_data = mask_nifti.get_fdata()
                    
                    # 指定されたスライスのマスクを取得
                    if slice_idx < mask_data.shape[2]:
                        slice_mask = mask_data[:, :, slice_idx]
                        return slice_mask
                    break
            
            return None
            
        except Exception as e:
            logger.error(f"骨折マスク取得エラー {patient_id}/{vertebra}/{slice_idx}: {e}")
            return None
    
    def mask_to_bounding_box(self, mask: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """マスクからバウンディングボックスを抽出（YOLO形式）
        
        prior_YOLO_fileの手法を参考に改良：
        - 複数の小さな骨折領域を統合
        - 適切なマージンとパディングを適用
        - 医療画像に適したサイズ調整
        """
        if mask is None or np.sum(mask) == 0:
            return []
        
        # バイナリマスクを作成
        mask_binary = (mask > 0).astype(np.uint8)
        
        # 小さなノイズを除去（モルフォロジー処理）
        kernel = np.ones((3, 3), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
        
        # 連結成分を取得
        num_labels, labels = cv2.connectedComponents(mask_cleaned)
        
        if num_labels <= 1:  # 背景のみ
            return []
        
        # 全ての骨折領域を統合してひとつのBBoxを作成（prior_YOLO_fileの手法）
        all_coords = []
        component_areas = []
        
        for label_id in range(1, num_labels):
            component_mask = (labels == label_id)
            y_coords, x_coords = np.where(component_mask)
            
            if len(y_coords) == 0 or len(x_coords) == 0:
                continue
            
            # 各成分の面積を計算
            area = len(y_coords)
            component_areas.append(area)
            
            # 座標を収集
            all_coords.extend(list(zip(x_coords, y_coords)))
        
        if not all_coords:
            return []
        
        # 全ての骨折領域を包含するバウンディングボックスを計算
        x_coords, y_coords = zip(*all_coords)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # prior_YOLO_fileの手法に基づくマージン追加
        h, w = mask.shape
        
        # 基本マージン（10ピクセル相当）
        margin_x = max(3, int(0.02 * w))  # 画像幅の2%または3ピクセル
        margin_y = max(3, int(0.02 * h))  # 画像高さの2%または3ピクセル
        
        # 骨折領域のサイズに応じた適応的マージン
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        # 小さな骨折領域には追加のマージン
        if bbox_width < 10 or bbox_height < 10:
            margin_x = max(margin_x, 5)
            margin_y = max(margin_y, 5)
        
        # 大きな骨折領域にはより大きなマージン
        if bbox_width > 50 or bbox_height > 50:
            margin_x = max(margin_x, int(0.1 * bbox_width))
            margin_y = max(margin_y, int(0.1 * bbox_height))
        
        # マージンを適用
        x_min = max(0, x_min - margin_x)
        x_max = min(w - 1, x_max + margin_x)
        y_min = max(0, y_min - margin_y)
        y_max = min(h - 1, y_max + margin_y)
        
        # アスペクト比の調整（医療画像での見やすさを考慮）
        current_width = x_max - x_min
        current_height = y_max - y_min
        
        # 極端なアスペクト比を避ける
        if current_width / current_height > 3:  # 横長すぎる
            center_y = (y_min + y_max) / 2
            target_height = current_width / 2
            y_min = max(0, int(center_y - target_height / 2))
            y_max = min(h - 1, int(center_y + target_height / 2))
        elif current_height / current_width > 3:  # 縦長すぎる
            center_x = (x_min + x_max) / 2
            target_width = current_height / 2
            x_min = max(0, int(center_x - target_width / 2))
            x_max = min(w - 1, int(center_x + target_width / 2))
        
        # 最小サイズの保証
        min_size = max(10, int(0.03 * min(w, h)))  # 画像の3%または10ピクセル
        
        if (x_max - x_min) < min_size:
            center_x = (x_min + x_max) / 2
            x_min = max(0, int(center_x - min_size / 2))
            x_max = min(w - 1, int(center_x + min_size / 2))
        
        if (y_max - y_min) < min_size:
            center_y = (y_min + y_max) / 2
            y_min = max(0, int(center_y - min_size / 2))
            y_max = min(h - 1, int(center_y + min_size / 2))
        
        # YOLO形式に変換（正規化座標）
        center_x = (x_min + x_max) / 2 / w
        center_y = (y_min + y_max) / 2 / h
        bbox_w = (x_max - x_min) / w
        bbox_h = (y_max - y_min) / h
        
        # 座標範囲の検証
        if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 
                0 < bbox_w <= 1 and 0 < bbox_h <= 1):
            logger.warning(f"無効なバウンディングボックス座標: center=({center_x:.4f}, {center_y:.4f}), size=({bbox_w:.4f}, {bbox_h:.4f})")
            return []
        
        return [(center_x, center_y, bbox_w, bbox_h)]
    
    def resize_image_and_adjust_bbox(self, image: np.ndarray, bboxes: List[Tuple]) -> Tuple[np.ndarray, List[Tuple]]:
        """画像をリサイズし、バウンディングボックスを調整"""
        original_h, original_w = image.shape[:2]
        
        # アスペクト比を保持してリサイズ
        scale = min(self.image_size / original_w, self.image_size / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        resized_image = cv2.resize(image, (new_w, new_h))
        
        # パディングを追加
        top = (self.image_size - new_h) // 2
        bottom = self.image_size - new_h - top
        left = (self.image_size - new_w) // 2
        right = self.image_size - new_w - left
        
        padded_image = cv2.copyMakeBorder(
            resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )
        
        # バウンディングボックスを調整
        adjusted_bboxes = []
        for center_x, center_y, bbox_w, bbox_h in bboxes:
            # 元の座標をリサイズ後の座標に変換
            new_center_x = (center_x * original_w * scale + left) / self.image_size
            new_center_y = (center_y * original_h * scale + top) / self.image_size
            new_bbox_w = bbox_w * scale * original_w / self.image_size
            new_bbox_h = bbox_h * scale * original_h / self.image_size
            
            adjusted_bboxes.append((new_center_x, new_center_y, new_bbox_w, new_bbox_h))
        
        return padded_image, adjusted_bboxes
    
    def process_slice(self, slice_info: Dict) -> bool:
        """単一スライスを処理"""
        try:
            slice_path = Path(slice_info['FullPath'])
            patient_id = str(slice_info['Case'])
            vertebra = str(slice_info['Vertebra'])
            slice_idx = slice_info['SliceIndex']
            fracture_label = slice_info['Fracture_Label']
            
            # 画像を読み込み
            image = self.load_nifti_slice(slice_path)
            if image is None:
                return False
            
            # 出力ファイル名を生成
            output_filename = f"{patient_id}_{vertebra}_{slice_idx:03d}"
            
            # データ分割を決定
            split = self.determine_split(slice_path)
            
            # 画像を3チャンネルに変換（YOLOv8要求）
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # バウンディングボックスを取得
            bboxes = []
            if fracture_label == 1:
                mask = self.get_fracture_mask(patient_id, vertebra, slice_idx)
                if mask is not None:
                    bboxes = self.mask_to_bounding_box(mask)
            
            # 画像をリサイズし、バウンディングボックスを調整
            resized_image, adjusted_bboxes = self.resize_image_and_adjust_bbox(image, bboxes)
            
            # 画像を保存
            image_output_path = self.output_dir / split / 'images' / f"{output_filename}.jpg"
            cv2.imwrite(str(image_output_path), resized_image)
            
            # アノテーションを保存
            label_output_path = self.output_dir / split / 'labels' / f"{output_filename}.txt"
            with open(label_output_path, 'w') as f:
                for center_x, center_y, bbox_w, bbox_h in adjusted_bboxes:
                    # クラス0（骨折）、YOLO形式
                    f.write(f"0 {center_x:.6f} {center_y:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"スライス処理エラー {slice_info}: {e}")
            return False
    
    def determine_split(self, slice_path: Path) -> str:
        """スライスパスから訓練/検証/テスト分割を決定"""
        path_str = str(slice_path)
        if '/slice_train/' in path_str:
            return 'train'
        elif '/slice_val/' in path_str:
            return 'val'
        elif '/slice_test/' in path_str:
            return 'test'
        else:
            return 'train'  # デフォルト
    
    def create_dataset_config(self):
        """YOLO用データセット設定ファイルを作成"""
        config_content = f"""# Vertebrae Fracture Detection Dataset
path: {self.output_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
nc: 1  # number of classes
names: ['fracture']  # class names
"""
        
        config_path = self.output_dir / 'configs' / 'vertebrae_fracture.yaml'
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        logger.info(f"データセット設定ファイル作成: {config_path}")
    
    def convert_dataset(self):
        """データセット全体を変換"""
        logger.info("NIfTI → YOLO形式変換開始")
        
        # 各分割のCSVファイルを処理
        total_processed = 0
        total_success = 0
        
        for split in ['train', 'val', 'test']:
            csv_path = self.input_dir / f"slice_{split}" / "axial" / f"{split}_labels_axial.csv"
            if not csv_path.exists():
                logger.warning(f"CSVファイルが見つかりません: {csv_path}")
                continue
            
            logger.info(f"{split}データセットを処理中...")
            
            # CSVを読み込み
            df = pd.read_csv(csv_path)
            
            # 並列処理で各スライスを変換
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for _, row in df.iterrows():
                    future = executor.submit(self.process_slice, row.to_dict())
                    futures.append(future)
                
                # 結果を収集
                for future in as_completed(futures):
                    total_processed += 1
                    if future.result():
                        total_success += 1
                    
                    if total_processed % 100 == 0:
                        logger.info(f"処理済み: {total_processed}, 成功: {total_success}")
        
        # データセット設定ファイルを作成
        self.create_dataset_config()
        
        logger.info(f"変換完了: {total_success}/{total_processed} スライス")
        
        # 統計情報を出力
        self.print_dataset_stats()
    
    def print_dataset_stats(self):
        """データセット統計情報を出力"""
        stats = {}
        for split in ['train', 'val', 'test']:
            images_dir = self.output_dir / split / 'images'
            labels_dir = self.output_dir / split / 'labels'
            
            if images_dir.exists():
                num_images = len(list(images_dir.glob('*.jpg')))
                num_labels = len(list(labels_dir.glob('*.txt')))
                
                # 骨折ありのラベル数をカウント
                fracture_count = 0
                for label_file in labels_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            fracture_count += 1
                
                stats[split] = {
                    'images': num_images,
                    'labels': num_labels,
                    'fractures': fracture_count,
                    'normal': num_images - fracture_count
                }
        
        logger.info("\n=== データセット統計 ===")
        for split, stat in stats.items():
            logger.info(f"{split.upper()}:")
            logger.info(f"  画像数: {stat['images']}")
            logger.info(f"  ラベル数: {stat['labels']}")
            logger.info(f"  骨折あり: {stat['fractures']}")
            logger.info(f"  正常: {stat['normal']}")
            if stat['images'] > 0:
                fracture_ratio = stat['fractures'] / stat['images'] * 100
                logger.info(f"  骨折率: {fracture_ratio:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='NIfTI形式椎体スライスをYOLO形式に変換')
    parser.add_argument('--input_dir', 
                       default='/mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/Sakaguchi_file',
                       help='入力データディレクトリ')
    parser.add_argument('--output_dir',
                       default='/mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/Sakaguchi_file/YOLO_datasets/vertebrae_fracture',
                       help='出力YOLOデータディレクトリ')
    parser.add_argument('--image_size', type=int, default=640,
                       help='YOLO用画像サイズ')
    
    args = parser.parse_args()
    
    converter = NiftiToYoloConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_size=args.image_size
    )
    
    converter.convert_dataset()


if __name__ == "__main__":
    main()