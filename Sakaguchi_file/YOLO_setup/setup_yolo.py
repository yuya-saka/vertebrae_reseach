#!/usr/bin/env python3
"""
YOLO椎体骨折検出システムのセットアップスクリプト

このスクリプトは以下の処理を行います：
1. 必要なパッケージのインストール確認
2. データセットの準備
3. 環境設定の確認
4. 初期テストの実行

使用方法:
    python setup_yolo.py --check_all
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging
import importlib.util

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YOLOSetup:
    """YOLO環境セットアップクラス"""
    
    def __init__(self):
        self.required_packages = [
            'torch',
            'torchvision', 
            'ultralytics',
            'numpy',
            'opencv-python',
            'pandas',
            'nibabel',
            'scikit-learn',
            'matplotlib',
            'seaborn',
            'albumentations',
            'wandb',
            'pyyaml'
        ]
        
        self.base_dir = Path(__file__).parent.parent
        self.sakaguchi_dir = self.base_dir
        
    def check_package_installation(self) -> bool:
        """必要なパッケージのインストール状況を確認"""
        logger.info("パッケージインストール状況確認開始")
        
        missing_packages = []
        
        for package in self.required_packages:
            try:
                if package == 'opencv-python':
                    import cv2
                elif package == 'pyyaml':
                    import yaml
                else:
                    importlib.import_module(package)
                logger.info(f"✓ {package}: インストール済み")
            except ImportError:
                logger.warning(f"✗ {package}: 未インストール")
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"未インストールパッケージ: {missing_packages}")
            return False
        
        logger.info("全パッケージインストール確認完了")
        return True
    
    def install_missing_packages(self):
        """未インストールパッケージをインストール"""
        logger.info("未インストールパッケージのインストール開始")
        
        # Ultralyticsのインストール
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics'])
            logger.info("ultralytics インストール完了")
        except subprocess.CalledProcessError:
            logger.error("ultralytics インストール失敗")
        
        # その他のパッケージ
        packages_to_install = [
            'opencv-python',
            'albumentations',
            'wandb',
            'seaborn'
        ]
        
        for package in packages_to_install:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                logger.info(f"{package} インストール完了")
            except subprocess.CalledProcessError:
                logger.error(f"{package} インストール失敗")
    
    def check_cuda_availability(self):
        """CUDA利用可能性確認"""
        logger.info("CUDA利用可能性確認")
        
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"✓ CUDA利用可能: {torch.cuda.get_device_name(0)}")
                logger.info(f"✓ CUDAバージョン: {torch.version.cuda}")
                logger.info(f"✓ 利用可能GPU数: {torch.cuda.device_count()}")
            else:
                logger.warning("✗ CUDA利用不可 - CPUモードで実行")
        except ImportError:
            logger.error("PyTorchがインストールされていません")
    
    def check_data_structure(self):
        """データ構造の確認"""
        logger.info("データ構造確認開始")
        
        # 必要なディレクトリの確認
        required_dirs = [
            self.sakaguchi_dir / 'slice_train' / 'axial',
            self.sakaguchi_dir / 'slice_val' / 'axial', 
            self.sakaguchi_dir / 'slice_test' / 'axial',
            self.sakaguchi_dir / 'processed_train',
            self.sakaguchi_dir / 'processed_val',
            self.sakaguchi_dir / 'processed_test'
        ]
        
        for dir_path in required_dirs:
            if dir_path.exists():
                logger.info(f"✓ {dir_path}: 存在")
            else:
                logger.warning(f"✗ {dir_path}: 不存在")
        
        # CSVファイルの確認
        csv_files = [
            self.sakaguchi_dir / 'slice_train' / 'axial' / 'train_labels_axial.csv',
            self.sakaguchi_dir / 'slice_val' / 'axial' / 'val_labels_axial.csv',
            self.sakaguchi_dir / 'slice_test' / 'axial' / 'test_labels_axial.csv'
        ]
        
        for csv_file in csv_files:
            if csv_file.exists():
                logger.info(f"✓ {csv_file}: 存在")
            else:
                logger.warning(f"✗ {csv_file}: 不存在")
    
    def create_directory_structure(self):
        """YOLOプロジェクトのディレクトリ構造を作成"""
        logger.info("YOLOディレクトリ構造作成")
        
        yolo_dirs = [
            self.sakaguchi_dir / 'YOLO_datasets',
            self.sakaguchi_dir / 'YOLO_models',
            self.sakaguchi_dir / 'YOLO_results',
            self.sakaguchi_dir / 'YOLO_logs'
        ]
        
        for dir_path in yolo_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ ディレクトリ作成: {dir_path}")
    
    def run_conversion_test(self):
        """データ変換テストを実行"""
        logger.info("データ変換テスト開始")
        
        conversion_script = self.sakaguchi_dir / 'YOLO_data_conversion' / 'nifti_to_yolo.py'
        
        if not conversion_script.exists():
            logger.error(f"変換スクリプトが見つかりません: {conversion_script}")
            return False
        
        try:
            # テスト用の小さなデータセットで変換テスト
            test_cmd = [
                sys.executable, 
                str(conversion_script),
                '--input_dir', str(self.sakaguchi_dir),
                '--output_dir', str(self.sakaguchi_dir / 'YOLO_datasets' / 'test_conversion'),
                '--image_size', '640'
            ]
            
            logger.info("変換テスト実行中...")
            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info("✓ データ変換テスト成功")
                return True
            else:
                logger.error(f"✗ データ変換テスト失敗: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("✗ データ変換テストタイムアウト")
            return False
        except Exception as e:
            logger.error(f"✗ データ変換テストエラー: {e}")
            return False
    
    def run_yolo_test(self):
        """YOLOモデルテストを実行"""
        logger.info("YOLOモデルテスト開始")
        
        try:
            from ultralytics import YOLO
            
            # YOLOv8モデルの初期化テスト
            model = YOLO('yolov8n.pt')
            logger.info("✓ YOLOv8モデル初期化成功")
            
            # サンプル画像での推論テスト
            import numpy as np
            sample_image = np.zeros((640, 640, 3), dtype=np.uint8)
            results = model(sample_image)
            logger.info("✓ YOLO推論テスト成功")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ YOLOモデルテスト失敗: {e}")
            return False
    
    def create_config_template(self):
        """設定ファイルテンプレートを作成"""
        logger.info("設定ファイルテンプレート作成")
        
        config_template = f"""# Vertebrae Fracture Detection Dataset Configuration
path: {self.sakaguchi_dir / 'YOLO_datasets' / 'vertebrae_fracture'}
train: train/images
val: val/images
test: test/images

# Classes
nc: 1  # number of classes
names: ['fracture']  # class names

# Training parameters
epochs: 100
batch_size: 16
img_size: 640
lr0: 0.01
weight_decay: 0.0005

# Model settings
model_size: yolov8m.pt
device: cuda  # or cpu
"""
        
        config_path = self.sakaguchi_dir / 'YOLO_setup' / 'config_template.yaml'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            f.write(config_template)
        
        logger.info(f"✓ 設定ファイルテンプレート作成: {config_path}")
    
    def run_full_setup(self):
        """完全なセットアップを実行"""
        logger.info("=== YOLO椎体骨折検出システムセットアップ開始 ===")
        
        # 1. パッケージ確認
        if not self.check_package_installation():
            logger.info("不足パッケージのインストールを試行")
            self.install_missing_packages()
        
        # 2. CUDA確認
        self.check_cuda_availability()
        
        # 3. データ構造確認
        self.check_data_structure()
        
        # 4. ディレクトリ作成
        self.create_directory_structure()
        
        # 5. 設定ファイルテンプレート作成
        self.create_config_template()
        
        # 6. YOLOテスト
        if self.run_yolo_test():
            logger.info("✓ YOLO環境セットアップ成功")
        else:
            logger.error("✗ YOLO環境セットアップ失敗")
        
        logger.info("=== セットアップ完了 ===")
    
    def print_next_steps(self):
        """次のステップを表示"""
        logger.info("=== 次のステップ ===")
        logger.info("1. データ変換実行:")
        logger.info("   python YOLO_data_conversion/nifti_to_yolo.py")
        logger.info("2. モデル訓練実行:")
        logger.info("   python YOLO_training/train_yolo.py --data_config /path/to/config.yaml")
        logger.info("3. 推論・評価実行:")
        logger.info("   python YOLO_inference/predict_yolo.py --model_path /path/to/model.pt --data_config /path/to/config.yaml")


def main():
    parser = argparse.ArgumentParser(description='YOLO椎体骨折検出システムセットアップ')
    parser.add_argument('--check_packages', action='store_true',
                       help='パッケージインストール状況のみ確認')
    parser.add_argument('--check_data', action='store_true',
                       help='データ構造のみ確認')
    parser.add_argument('--check_cuda', action='store_true',
                       help='CUDA利用可能性のみ確認')
    parser.add_argument('--install_packages', action='store_true',
                       help='不足パッケージのインストール')
    parser.add_argument('--test_conversion', action='store_true',
                       help='データ変換テストのみ実行')
    parser.add_argument('--test_yolo', action='store_true',
                       help='YOLOテストのみ実行')
    parser.add_argument('--check_all', action='store_true',
                       help='完全なセットアップ実行')
    
    args = parser.parse_args()
    
    setup = YOLOSetup()
    
    if args.check_packages:
        setup.check_package_installation()
    elif args.check_data:
        setup.check_data_structure()
    elif args.check_cuda:
        setup.check_cuda_availability()
    elif args.install_packages:
        setup.install_missing_packages()
    elif args.test_conversion:
        setup.run_conversion_test()
    elif args.test_yolo:
        setup.run_yolo_test()
    elif args.check_all:
        setup.run_full_setup()
        setup.print_next_steps()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()