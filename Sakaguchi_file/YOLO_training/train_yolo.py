#!/usr/bin/env python3
"""
YOLOv8を使用した椎体骨折検出モデルの学習スクリプト

このスクリプトは以下の処理を行います：
1. YOLO形式に変換されたデータセットを読み込み
2. YOLOv8モデルを初期化
3. 医療画像に特化した設定で学習
4. モデルの評価とチェックポイント保存

使用方法:
    python train_yolo.py --data_config /path/to/config.yaml --epochs 100
"""

import argparse
import yaml
from pathlib import Path
import torch
import wandb
from ultralytics import YOLO
from typing import Dict, Any
import logging
from datetime import datetime

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VertebralFractureYOLO:
    """椎体骨折検出用YOLOv8トレーナー"""
    
    def __init__(self, config_path: str, model_size: str = 'yolov8m.pt'):
        """
        Args:
            config_path: データセット設定ファイルパス
            model_size: YOLOv8モデルサイズ（n/s/m/l/x）
        """
        self.config_path = Path(config_path)
        self.model_size = model_size
        
        # 設定を読み込み
        self.config = self.load_config()
        
        # YOLOv8モデルを初期化
        self.model = YOLO(model_size)
        
        # 出力ディレクトリを設定
        self.output_dir = Path(f"runs/train/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"YOLOv8モデル初期化: {model_size}")
        logger.info(f"出力ディレクトリ: {self.output_dir}")
    
    def load_config(self) -> Dict[str, Any]:
        """データセット設定ファイルを読み込み"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"設定ファイル読み込み完了: {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"設定ファイル読み込みエラー: {e}")
            raise
    
    def setup_wandb(self, project_name: str = "vertebrae-fracture-yolo"):
        """Weights & Biases実験管理の設定"""
        try:
            wandb.init(
                project=project_name,
                config={
                    "model_size": self.model_size,
                    "dataset": self.config_path.name,
                    "image_size": 640,
                    "classes": self.config.get('nc', 1),
                    "framework": "YOLOv8-Ultralytics"
                }
            )
            logger.info("Weights & Biases初期化完了")
            
        except Exception as e:
            logger.warning(f"Weights & Biases初期化失敗: {e}")
    
    def get_training_params(self) -> Dict[str, Any]:
        """医療画像に特化した学習パラメータを取得"""
        return {
            # 基本設定
            'data': str(self.config_path),
            'epochs': 100,
            'batch': 16,
            'imgsz': 640,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            
            # 学習率設定
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            
            # データ拡張設定（医療画像用）
            'hsv_h': 0.015,  # 色相変化を制限
            'hsv_s': 0.3,    # 彩度変化を制限
            'hsv_v': 0.3,    # 明度変化を制限
            'degrees': 20,   # 回転角度制限
            'translate': 0.1,  # 平行移動制限
            'scale': 0.2,    # スケール変化制限
            'shear': 0.1,    # せん断変形制限
            'perspective': 0.0,  # 透視変換無効
            'flipud': 0.0,   # 上下反転無効
            'fliplr': 0.5,   # 左右反転は有効
            'mosaic': 0.8,   # モザイク拡張確率
            'mixup': 0.1,    # ミックスアップ確率
            'copy_paste': 0.1,  # コピー&ペースト確率
            
            # 正則化設定
            'dropout': 0.0,
            'label_smoothing': 0.0,
            
            # 評価設定
            'val': True,
            'save_period': 10,
            'save_json': True,
            'project': str(self.output_dir.parent),
            'name': self.output_dir.name,
            
            # 早期停止設定
            'patience': 30,
            
            # その他設定
            'workers': 4,
            'seed': 42,
            'deterministic': True,
            'single_cls': True,  # 単一クラス（骨折）
            'rect': True,  # 矩形学習
            'cos_lr': True,  # コサイン学習率スケジューリング
            'close_mosaic': 10,  # モザイク拡張を終了するエポック
            'resume': False,
            'amp': True,  # 自動混合精度
            'fraction': 1.0,  # データセットの使用割合
            'profile': False,
            'freeze': None,  # 凍結レイヤー
            'multi_scale': True,  # マルチスケール学習
            'overlap_mask': True,  # 重複マスク
            'mask_ratio': 4,  # マスク比率
            'box': 7.5,  # ボックス損失重み
            'cls': 0.5,  # 分類損失重み
            'dfl': 1.5,  # 分布焦点損失重み
            'pose': 12.0,  # ポーズ損失重み
            'kobj': 2.0,  # キーポイント損失重み
            'nbs': 64,  # 正規化バッチサイズ
            'optimizer': 'auto',  # オプティマイザー
        }
    
    def train(self, epochs: int = 100, batch_size: int = 16):
        """YOLOv8モデルの学習"""
        logger.info("YOLOv8モデル学習開始")
        
        # 学習パラメータを取得
        params = self.get_training_params()
        params['epochs'] = epochs
        params['batch'] = batch_size
        
        try:
            # 学習実行
            results = self.model.train(**params)
            
            logger.info("学習完了")
            return results
            
        except Exception as e:
            logger.error(f"学習エラー: {e}")
            raise
    
    def evaluate(self, data_split: str = 'val'):
        """モデルの評価"""
        logger.info(f"{data_split}データセットでの評価開始")
        
        try:
            # 評価実行
            results = self.model.val(
                data=str(self.config_path),
                split=data_split,
                save_json=True,
                save_hybrid=True
            )
            
            logger.info("評価完了")
            return results
            
        except Exception as e:
            logger.error(f"評価エラー: {e}")
            raise
    
    def export_model(self, format: str = 'onnx'):
        """モデルを指定形式でエクスポート"""
        logger.info(f"モデルを{format}形式でエクスポート")
        
        try:
            exported_model = self.model.export(format=format)
            logger.info(f"エクスポート完了: {exported_model}")
            return exported_model
            
        except Exception as e:
            logger.error(f"エクスポートエラー: {e}")
            raise
    
    def predict_sample(self, image_path: str, conf_threshold: float = 0.25):
        """サンプル画像での推論"""
        try:
            results = self.model(image_path, conf=conf_threshold)
            return results
            
        except Exception as e:
            logger.error(f"推論エラー: {e}")
            raise
    
    def save_training_summary(self, results):
        """学習結果のサマリーを保存"""
        summary = {
            'model_size': self.model_size,
            'config_path': str(self.config_path),
            'output_dir': str(self.output_dir),
            'final_metrics': {
                'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
                'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
                'precision': results.results_dict.get('metrics/precision(B)', 0),
                'recall': results.results_dict.get('metrics/recall(B)', 0)
            },
            'training_params': self.get_training_params()
        }
        
        summary_path = self.output_dir / 'training_summary.yaml'
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f"学習サマリー保存: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='YOLOv8椎体骨折検出モデル学習')
    parser.add_argument('--data_config', 
                       default='/mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/Sakaguchi_file/YOLO_datasets/vertebrae_fracture/configs/vertebrae_fracture.yaml',
                       help='データセット設定ファイルパス')
    parser.add_argument('--model_size', 
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                       default='yolov8m.pt',
                       help='YOLOv8モデルサイズ')
    parser.add_argument('--epochs', type=int, default=100,
                       help='学習エポック数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='バッチサイズ')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Weights & Biasesを使用')
    parser.add_argument('--evaluate_only', action='store_true',
                       help='評価のみ実行')
    parser.add_argument('--export_format', 
                       choices=['onnx', 'torchscript', 'tensorrt'],
                       help='モデルエクスポート形式')
    
    args = parser.parse_args()
    
    # トレーナーを初期化
    trainer = VertebralFractureYOLO(
        config_path=args.data_config,
        model_size=args.model_size
    )
    
    # Weights & Biasesセットアップ
    if args.use_wandb:
        trainer.setup_wandb()
    
    # 評価のみの場合
    if args.evaluate_only:
        results = trainer.evaluate()
        logger.info(f"評価結果: {results}")
        return
    
    # 学習実行
    results = trainer.train(epochs=args.epochs, batch_size=args.batch_size)
    
    # 学習サマリーを保存
    trainer.save_training_summary(results)
    
    # 評価実行
    val_results = trainer.evaluate()
    logger.info(f"検証結果: {val_results}")
    
    # モデルエクスポート
    if args.export_format:
        trainer.export_model(args.export_format)
    
    # W&Bセッション終了
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()