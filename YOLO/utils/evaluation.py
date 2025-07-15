"""
Evaluation utilities for YOLO model performance assessment.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve
)
from typing import List, Dict, Tuple, Optional
import pandas as pd
import seaborn as sns
from pathlib import Path
import json


class YOLOEvaluator:
    """
    Evaluator for YOLO model performance on medical images.
    """
    
    def __init__(self, output_dir: str = './evaluation_results'):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}
    
    def evaluate_classification(self, 
                              y_true: np.ndarray, 
                              y_pred: np.ndarray, 
                              y_proba: Optional[np.ndarray] = None,
                              class_names: List[str] = None) -> Dict:
        """
        Evaluate classification performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            class_names: List of class names
            
        Returns:
            Evaluation metrics dictionary
        """
        if class_names is None:
            class_names = ['normal', 'fracture']
        
        # Basic metrics
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        metrics = {
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'accuracy': report['accuracy'],
            'precision_fracture': report['fracture']['precision'],
            'recall_fracture': report['fracture']['recall'],
            'f1_fracture': report['fracture']['f1-score'],
            'precision_normal': report['normal']['precision'],
            'recall_normal': report['normal']['recall'],
            'f1_normal': report['normal']['f1-score'],
        }
        
        # Add ROC and PR metrics if probabilities are provided
        if y_proba is not None:
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = roc_auc_score(y_true, y_proba)
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            pr_auc = average_precision_score(y_true, y_proba)
            
            metrics.update({
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'precision_curve': precision.tolist(),
                'recall_curve': recall.tolist(),
            })
        
        # Medical-specific metrics
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        metrics.update({
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,  # Positive Predictive Value
            'npv': npv,  # Negative Predictive Value
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
        })
        
        self.results['classification'] = metrics
        return metrics
    
    def evaluate_detection(self, 
                          ground_truth: List[Dict], 
                          predictions: List[Dict],
                          iou_threshold: float = 0.5) -> Dict:
        """
        Evaluate object detection performance.
        
        Args:
            ground_truth: List of ground truth annotations
            predictions: List of predictions
            iou_threshold: IoU threshold for positive detection
            
        Returns:
            Detection metrics dictionary
        """
        def calculate_iou(box1, box2):
            """Calculate IoU between two bounding boxes."""
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2
            
            # Calculate intersection
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)
            
            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0.0
            
            intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            
            # Calculate union
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        tp = 0
        fp = 0
        fn = 0
        
        for gt, pred in zip(ground_truth, predictions):
            gt_boxes = gt.get('boxes', [])
            pred_boxes = pred.get('boxes', [])
            
            # Match predictions to ground truth
            matched_gt = set()
            
            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                for i, gt_box in enumerate(gt_boxes):
                    if i in matched_gt:
                        continue
                    
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                if best_iou >= iou_threshold:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            
            # Count unmatched ground truth boxes as false negatives
            fn += len(gt_boxes) - len(matched_gt)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'iou_threshold': iou_threshold,
        }
        
        self.results['detection'] = metrics
        return metrics
    
    def plot_confusion_matrix(self, 
                            class_names: List[str] = None,
                            figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot confusion matrix.
        
        Args:
            class_names: List of class names
            figsize: Figure size
        """
        if 'classification' not in self.results:
            raise ValueError("No classification results available")
        
        if class_names is None:
            class_names = ['normal', 'fracture']
        
        cm = np.array(self.results['classification']['confusion_matrix'])
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
    
    def plot_roc_curve(self, figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot ROC curve.
        
        Args:
            figsize: Figure size
        """
        if 'classification' not in self.results or 'fpr' not in self.results['classification']:
            raise ValueError("No ROC curve data available")
        
        fpr = self.results['classification']['fpr']
        tpr = self.results['classification']['tpr']
        roc_auc = self.results['classification']['roc_auc']
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curve.png', dpi=300)
        plt.close()
    
    def plot_precision_recall_curve(self, figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot Precision-Recall curve.
        
        Args:
            figsize: Figure size
        """
        if 'classification' not in self.results or 'precision_curve' not in self.results['classification']:
            raise ValueError("No PR curve data available")
        
        precision = self.results['classification']['precision_curve']
        recall = self.results['classification']['recall_curve']
        pr_auc = self.results['classification']['pr_auc']
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'precision_recall_curve.png', dpi=300)
        plt.close()
    
    def generate_report(self, model_name: str = "YOLO") -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Report string
        """
        report_lines = [
            f"# {model_name} Evaluation Report",
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        if 'classification' in self.results:
            metrics = self.results['classification']
            report_lines.extend([
                "## Classification Performance",
                f"- **Accuracy**: {metrics['accuracy']:.3f}",
                f"- **Sensitivity (Recall)**: {metrics['sensitivity']:.3f}",
                f"- **Specificity**: {metrics['specificity']:.3f}",
                f"- **Precision (Fracture)**: {metrics['precision_fracture']:.3f}",
                f"- **F1-Score (Fracture)**: {metrics['f1_fracture']:.3f}",
                f"- **PPV**: {metrics['ppv']:.3f}",
                f"- **NPV**: {metrics['npv']:.3f}",
                "",
                "### Confusion Matrix",
                f"- True Positives: {metrics['true_positives']}",
                f"- True Negatives: {metrics['true_negatives']}",
                f"- False Positives: {metrics['false_positives']}",
                f"- False Negatives: {metrics['false_negatives']}",
                "",
            ])
            
            if 'roc_auc' in metrics:
                report_lines.extend([
                    f"- **ROC AUC**: {metrics['roc_auc']:.3f}",
                    f"- **PR AUC**: {metrics['pr_auc']:.3f}",
                    "",
                ])
        
        if 'detection' in self.results:
            metrics = self.results['detection']
            report_lines.extend([
                "## Detection Performance",
                f"- **Precision**: {metrics['precision']:.3f}",
                f"- **Recall**: {metrics['recall']:.3f}",
                f"- **F1-Score**: {metrics['f1_score']:.3f}",
                f"- **IoU Threshold**: {metrics['iou_threshold']:.2f}",
                "",
                f"- True Positives: {metrics['true_positives']}",
                f"- False Positives: {metrics['false_positives']}",
                f"- False Negatives: {metrics['false_negatives']}",
                "",
            ])
        
        report_text = "\n".join(report_lines)
        
        # Save report
        with open(self.output_dir / 'evaluation_report.md', 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def save_results(self, filename: str = 'evaluation_results.json'):
        """
        Save evaluation results to JSON file.
        
        Args:
            filename: Output filename
        """
        with open(self.output_dir / filename, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def compare_models(self, other_results: Dict, model_names: List[str]) -> pd.DataFrame:
        """
        Compare performance between models.
        
        Args:
            other_results: Other model results
            model_names: List of model names
            
        Returns:
            Comparison DataFrame
        """
        models_data = [self.results, other_results]
        
        comparison_data = []
        
        for i, (data, name) in enumerate(zip(models_data, model_names)):
            if 'classification' in data:
                metrics = data['classification']
                row = {
                    'Model': name,
                    'Accuracy': metrics['accuracy'],
                    'Sensitivity': metrics['sensitivity'],
                    'Specificity': metrics['specificity'],
                    'Precision': metrics['precision_fracture'],
                    'F1-Score': metrics['f1_fracture'],
                    'PPV': metrics['ppv'],
                    'NPV': metrics['npv'],
                }
                
                if 'roc_auc' in metrics:
                    row['ROC AUC'] = metrics['roc_auc']
                    row['PR AUC'] = metrics['pr_auc']
                
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(self.output_dir / 'model_comparison.csv', index=False)
        
        return comparison_df