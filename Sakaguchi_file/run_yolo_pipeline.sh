#!/bin/bash
"""
YOLO椎体骨折検出システムの完全パイプライン実行スクリプト

このスクリプトは以下の処理を順次実行します：
1. 環境セットアップ
2. データ変換（NIfTI → YOLO形式）
3. モデル訓練
4. 推論・評価

使用方法:
    chmod +x run_yolo_pipeline.sh
    ./run_yolo_pipeline.sh
"""

set -e  # エラー時に停止

# 色付きログ関数
log_info() {
    echo -e "\033[32m[INFO]\033[0m $1"
}

log_warn() {
    echo -e "\033[33m[WARN]\033[0m $1"
}

log_error() {
    echo -e "\033[31m[ERROR]\033[0m $1"
}

# 設定
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAKAGUCHI_DIR="$SCRIPT_DIR"
YOLO_DATASET_DIR="$SAKAGUCHI_DIR/YOLO_datasets/vertebrae_fracture"
YOLO_MODEL_DIR="$SAKAGUCHI_DIR/YOLO_models"
YOLO_RESULTS_DIR="$SAKAGUCHI_DIR/YOLO_results"
CONFIG_FILE="$YOLO_DATASET_DIR/configs/vertebrae_fracture.yaml"

# 出力ディレクトリを作成
mkdir -p "$YOLO_MODEL_DIR" "$YOLO_RESULTS_DIR"

log_info "=== YOLO椎体骨折検出システム パイプライン開始 ==="

# ステップ1: 環境セットアップ
log_info "ステップ1: 環境セットアップ"
cd "$SAKAGUCHI_DIR"

if [ ! -f "YOLO_setup/setup_yolo.py" ]; then
    log_error "セットアップスクリプトが見つかりません"
    exit 1
fi

python YOLO_setup/setup_yolo.py --check_all

if [ $? -ne 0 ]; then
    log_error "環境セットアップ失敗"
    exit 1
fi

log_info "✓ 環境セットアップ完了"

# ステップ2: データ変換
log_info "ステップ2: データ変換（NIfTI → YOLO形式）"

if [ ! -f "YOLO_data_conversion/nifti_to_yolo.py" ]; then
    log_error "データ変換スクリプトが見つかりません"
    exit 1
fi

python YOLO_data_conversion/nifti_to_yolo.py \
    --input_dir "$SAKAGUCHI_DIR" \
    --output_dir "$YOLO_DATASET_DIR" \
    --image_size 640

if [ $? -ne 0 ]; then
    log_error "データ変換失敗"
    exit 1
fi

log_info "✓ データ変換完了"

# ステップ3: 設定ファイル確認
log_info "ステップ3: 設定ファイル確認"

if [ ! -f "$CONFIG_FILE" ]; then
    log_error "設定ファイルが見つかりません: $CONFIG_FILE"
    exit 1
fi

log_info "✓ 設定ファイル確認完了: $CONFIG_FILE"

# ステップ4: モデル訓練
log_info "ステップ4: YOLOv8モデル訓練"

if [ ! -f "YOLO_training/train_yolo.py" ]; then
    log_error "訓練スクリプトが見つかりません"
    exit 1
fi

# 訓練パラメータ
EPOCHS=100
BATCH_SIZE=16
MODEL_SIZE="yolov8m.pt"

log_info "訓練パラメータ:"
log_info "  エポック数: $EPOCHS"
log_info "  バッチサイズ: $BATCH_SIZE"
log_info "  モデルサイズ: $MODEL_SIZE"

python YOLO_training/train_yolo.py \
    --data_config "$CONFIG_FILE" \
    --model_size "$MODEL_SIZE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --use_wandb

if [ $? -ne 0 ]; then
    log_error "モデル訓練失敗"
    exit 1
fi

log_info "✓ モデル訓練完了"

# ステップ5: 最新の訓練済みモデルを取得
log_info "ステップ5: 訓練済みモデル取得"

# 最新の訓練結果ディレクトリを取得
LATEST_TRAIN_DIR=$(find YOLO_training/runs/train -name "20*" -type d | sort -r | head -1)

if [ -z "$LATEST_TRAIN_DIR" ]; then
    log_error "訓練結果ディレクトリが見つかりません"
    exit 1
fi

BEST_MODEL="$LATEST_TRAIN_DIR/weights/best.pt"

if [ ! -f "$BEST_MODEL" ]; then
    log_error "最適モデルが見つかりません: $BEST_MODEL"
    exit 1
fi

# モデルをコピー
cp "$BEST_MODEL" "$YOLO_MODEL_DIR/vertebrae_fracture_best.pt"
log_info "✓ 最適モデル保存: $YOLO_MODEL_DIR/vertebrae_fracture_best.pt"

# ステップ6: 推論・評価
log_info "ステップ6: 推論・評価"

if [ ! -f "YOLO_inference/predict_yolo.py" ]; then
    log_error "推論スクリプトが見つかりません"
    exit 1
fi

# テストデータで評価
python YOLO_inference/predict_yolo.py \
    --model_path "$YOLO_MODEL_DIR/vertebrae_fracture_best.pt" \
    --data_config "$CONFIG_FILE" \
    --data_split test \
    --conf_threshold 0.25

if [ $? -ne 0 ]; then
    log_error "推論・評価失敗"
    exit 1
fi

log_info "✓ 推論・評価完了"

# ステップ7: 結果サマリー
log_info "ステップ7: 結果サマリー"

# 最新の推論結果ディレクトリを取得
LATEST_PREDICT_DIR=$(find YOLO_inference/runs/predict -name "20*" -type d | sort -r | head -1)

if [ -n "$LATEST_PREDICT_DIR" ]; then
    # 結果をコピー
    cp -r "$LATEST_PREDICT_DIR" "$YOLO_RESULTS_DIR/final_results"
    log_info "✓ 結果保存: $YOLO_RESULTS_DIR/final_results"
    
    # メトリクスを表示
    if [ -f "$YOLO_RESULTS_DIR/final_results/metrics.json" ]; then
        log_info "=== 最終評価結果 ==="
        python -c "
import json
with open('$YOLO_RESULTS_DIR/final_results/metrics.json', 'r') as f:
    metrics = json.load(f)
    print(f'Accuracy: {metrics[\"accuracy\"]:.4f}')
    print(f'Precision: {metrics[\"precision\"]:.4f}')
    print(f'Recall: {metrics[\"recall\"]:.4f}')
    print(f'F1-Score: {metrics[\"f1_score\"]:.4f}')
    print(f'AUC: {metrics[\"auc\"]:.4f}')
"
    fi
fi

# ステップ8: 開発計画書の更新
log_info "ステップ8: 開発計画書更新"

# 進捗をログに記録
PROGRESS_LOG="$SAKAGUCHI_DIR/YOLO_progress.log"
echo "$(date): axial学習パイプライン完了" >> "$PROGRESS_LOG"
echo "  - データ変換: 完了" >> "$PROGRESS_LOG"
echo "  - モデル訓練: 完了" >> "$PROGRESS_LOG"
echo "  - 推論・評価: 完了" >> "$PROGRESS_LOG"
echo "  - 最適モデル: $YOLO_MODEL_DIR/vertebrae_fracture_best.pt" >> "$PROGRESS_LOG"
echo "  - 結果: $YOLO_RESULTS_DIR/final_results" >> "$PROGRESS_LOG"

log_info "=== パイプライン完了 ==="
log_info "訓練済みモデル: $YOLO_MODEL_DIR/vertebrae_fracture_best.pt"
log_info "評価結果: $YOLO_RESULTS_DIR/final_results"
log_info "進捗ログ: $PROGRESS_LOG"

# 次のステップを表示
log_info "=== 次のステップ（オプション） ==="
log_info "1. 他の方向（coronal/sagittal）での訓練"
log_info "2. ハイパーパラメータ最適化"
log_info "3. アンサンブル手法の適用"
log_info "4. ResNet18モデルとの性能比較"

log_info "パイプライン実行時間: $SECONDS 秒"