#!/usr/bin/env python3
"""
YOLOv8椎体骨折検出モデルの推論・評価スクリプト

このスクリプトは以下の処理を行います：
1. 学習済みYOLOv8モデルの読み込み
2. テストデータセットでの推論実行
3. 性能評価とメトリクス計算
4. 結果の可視化と保存

使用方法:
    python predict_yolo.py --model_path /path/to/model.pt --data_config /path/to/config.yaml
"""

import argparse
import yaml
from pathlib import Path
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import json

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo_inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VertebralFracturePredictor:
    """椎体骨折検出用YOLOv8推論クラス"""
    
    def __init__(self, model_path: str, config_path: str):
        """
        Args:
            model_path: 学習済みモデルパス
            config_path: データセット設定ファイルパス
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        
        # モデルを読み込み
        self.model = YOLO(str(self.model_path))
        
        # 設定を読み込み
        self.config = self.load_config()
        
        # 出力ディレクトリを設定
        self.output_dir = Path(f"runs/predict/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"モデル読み込み完了: {self.model_path}")
        logger.info(f"出力ディレクトリ: {self.output_dir}")
    
    def load_config(self) -> Dict[str, Any]:
        """データセット設定ファイルを読み込み"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"設定ファイル読み込みエラー: {e}")
            raise
    
    def predict_single_image(self, image_path: str, conf_threshold: float = 0.25) -> Dict:
        """単一画像での推論"""
        try:
            results = self.model(image_path, conf=conf_threshold)
            result = results[0]
            
            # 結果を整理
            predictions = {
                'image_path': image_path,
                'boxes': result.boxes.xyxy.cpu().numpy() if result.boxes is not None else [],
                'confidences': result.boxes.conf.cpu().numpy() if result.boxes is not None else [],
                'classes': result.boxes.cls.cpu().numpy() if result.boxes is not None else [],
                'has_fracture': len(result.boxes) > 0 if result.boxes is not None else False
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"推論エラー {image_path}: {e}")
            return None
    
    def predict_dataset(self, data_split: str = 'test', conf_threshold: float = 0.25) -> List[Dict]:
        """データセット全体での推論"""
        logger.info(f"{data_split}データセットでの推論開始")
        
        # データセットパスを取得
        dataset_path = Path(self.config['path']) / data_split / 'images'
        
        if not dataset_path.exists():
            logger.error(f"データセットパスが存在しません: {dataset_path}")
            return []
        
        # 画像ファイルリストを取得
        image_files = list(dataset_path.glob('*.jpg')) + list(dataset_path.glob('*.png'))
        
        predictions = []
        for i, image_file in enumerate(image_files):
            pred = self.predict_single_image(str(image_file), conf_threshold)
            if pred is not None:
                predictions.append(pred)
            
            if (i + 1) % 100 == 0:
                logger.info(f"推論進捗: {i + 1}/{len(image_files)}")
        
        logger.info(f"推論完了: {len(predictions)}枚")
        return predictions
    
    def load_ground_truth(self, data_split: str = 'test') -> Dict[str, List]:
        """グランドトゥルースアノテーションを読み込み"""
        labels_path = Path(self.config['path']) / data_split / 'labels'
        
        ground_truth = {}
        for label_file in labels_path.glob('*.txt'):
            image_name = label_file.stem + '.jpg'
            
            # ラベルファイルを読み込み
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            boxes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    boxes.append({
                        'class': class_id,
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': width,
                        'height': height
                    })
            
            ground_truth[image_name] = boxes
        
        return ground_truth
    
    def calculate_metrics(self, predictions: List[Dict], ground_truth: Dict[str, List]) -> Dict:
        """評価指標を計算"""
        logger.info("評価指標計算開始")
        
        # 分類レベルの評価
        y_true = []
        y_pred = []
        y_scores = []
        
        for pred in predictions:
            image_name = Path(pred['image_path']).name
            
            # グランドトゥルース
            gt_boxes = ground_truth.get(image_name, [])
            has_gt_fracture = len(gt_boxes) > 0
            
            # 予測結果
            has_pred_fracture = pred['has_fracture']
            confidence = max(pred['confidences']) if len(pred['confidences']) > 0 else 0.0
            
            y_true.append(int(has_gt_fracture))
            y_pred.append(int(has_pred_fracture))
            y_scores.append(confidence)
        
        # 基本メトリクス
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC計算
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # 感度・特異度
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': roc_auc,
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
            }
        }
        
        logger.info("評価指標計算完了")
        return metrics
    
    def visualize_results(self, predictions: List[Dict], ground_truth: Dict[str, List], 
                         metrics: Dict, num_samples: int = 10):
        """結果を可視化"""
        logger.info("結果可視化開始")
        
        # 1. 混同行列の可視化
        plt.figure(figsize=(8, 6))
        cm = [[metrics['confusion_matrix']['tn'], metrics['confusion_matrix']['fp']],
              [metrics['confusion_matrix']['fn'], metrics['confusion_matrix']['tp']]]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Fracture'],
                   yticklabels=['Normal', 'Fracture'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
        
        # 2. ROC曲線の可視化
        y_true = []
        y_scores = []
        for pred in predictions:
            image_name = Path(pred['image_path']).name
            gt_boxes = ground_truth.get(image_name, [])
            has_gt_fracture = len(gt_boxes) > 0
            confidence = max(pred['confidences']) if len(pred['confidences']) > 0 else 0.0
            
            y_true.append(int(has_gt_fracture))
            y_scores.append(confidence)
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {metrics["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curve.png', dpi=300)
        plt.close()
        
        # 3. サンプル画像の可視化
        self.visualize_sample_predictions(predictions, ground_truth, num_samples)
        
        logger.info("結果可視化完了")
    
    def visualize_sample_predictions(self, predictions: List[Dict], ground_truth: Dict[str, List], 
                                   num_samples: int = 10):
        """サンプル予測結果の可視化"""
        
        # 正例と負例を分けてサンプリング
        positive_samples = [p for p in predictions if p['has_fracture']]
        negative_samples = [p for p in predictions if not p['has_fracture']]
        
        samples = positive_samples[:num_samples//2] + negative_samples[:num_samples//2]
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for i, pred in enumerate(samples[:10]):
            if i >= len(axes):
                break
            
            # 画像を読み込み
            image = cv2.imread(pred['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 予測結果を描画
            for j, (box, conf) in enumerate(zip(pred['boxes'], pred['confidences'])):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, f'{conf:.2f}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # グランドトゥルースを描画
            image_name = Path(pred['image_path']).name
            gt_boxes = ground_truth.get(image_name, [])
            
            for gt_box in gt_boxes:
                # YOLO形式からピクセル座標に変換
                h, w = image.shape[:2]
                center_x = gt_box['center_x'] * w
                center_y = gt_box['center_y'] * h
                width = gt_box['width'] * w
                height = gt_box['height'] * h
                
                x1 = int(center_x - width/2)
                y1 = int(center_y - height/2)
                x2 = int(center_x + width/2)
                y2 = int(center_y + height/2)
                
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            axes[i].imshow(image)
            axes[i].set_title(f'GT: {len(gt_boxes) > 0}, Pred: {pred["has_fracture"]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_predictions.png', dpi=300)
        plt.close()
    
    def save_results(self, predictions: List[Dict], metrics: Dict):
        """結果を保存"""
        # 予測結果をJSON形式で保存
        predictions_serializable = []
        for pred in predictions:
            pred_copy = pred.copy()
            pred_copy['boxes'] = pred_copy['boxes'].tolist()
            pred_copy['confidences'] = pred_copy['confidences'].tolist()
            pred_copy['classes'] = pred_copy['classes'].tolist()
            predictions_serializable.append(pred_copy)
        
        with open(self.output_dir / 'predictions.json', 'w') as f:
            json.dump(predictions_serializable, f, indent=2)
        
        # メトリクスを保存
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # CSVレポートを保存
        results_df = pd.DataFrame([
            {
                'image_path': pred['image_path'],
                'has_fracture': pred['has_fracture'],
                'num_detections': len(pred['boxes']),
                'max_confidence': max(pred['confidences']) if len(pred['confidences']) > 0 else 0.0
            }
            for pred in predictions
        ])
        
        results_df.to_csv(self.output_dir / 'results.csv', index=False)
        
        logger.info(f"結果保存完了: {self.output_dir}")
    
    def run_evaluation(self, data_split: str = 'test', conf_threshold: float = 0.25):
        """完全な評価プロセスを実行"""
        logger.info("評価プロセス開始")
        
        # 推論実行
        predictions = self.predict_dataset(data_split, conf_threshold)
        
        # グランドトゥルース読み込み
        ground_truth = self.load_ground_truth(data_split)
        
        # メトリクス計算
        metrics = self.calculate_metrics(predictions, ground_truth)
        
        # 結果出力
        logger.info("=== 評価結果 ===")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"Sensitivity: {metrics['sensitivity']:.4f}")
        logger.info(f"Specificity: {metrics['specificity']:.4f}")
        logger.info(f"AUC: {metrics['auc']:.4f}")
        
        # 結果可視化
        self.visualize_results(predictions, ground_truth, metrics)
        
        # 結果保存
        self.save_results(predictions, metrics)
        
        return predictions, metrics


def main():
    parser = argparse.ArgumentParser(description='YOLOv8椎体骨折検出モデル推論・評価')
    parser.add_argument('--model_path', required=True,
                       help='学習済みモデルパス')
    parser.add_argument('--data_config', required=True,
                       help='データセット設定ファイルパス')
    parser.add_argument('--data_split', default='test',
                       choices=['train', 'val', 'test'],
                       help='評価するデータ分割')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='信頼度閾値')
    parser.add_argument('--single_image', 
                       help='単一画像での推論')
    
    args = parser.parse_args()
    
    # 予測器を初期化
    predictor = VertebralFracturePredictor(
        model_path=args.model_path,
        config_path=args.data_config
    )
    
    # 単一画像推論
    if args.single_image:
        result = predictor.predict_single_image(args.single_image, args.conf_threshold)
        logger.info(f"単一画像推論結果: {result}")
        return
    
    # データセット評価
    predictions, metrics = predictor.run_evaluation(args.data_split, args.conf_threshold)
    
    logger.info("評価完了")


if __name__ == "__main__":
    main()