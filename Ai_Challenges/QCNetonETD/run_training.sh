#!/bin/bash

# ETRI 경로 예측 챌린지 - 학습 실행 스크립트
# 이 스크립트는 Docker 컨테이너 내부에서 실행됩니다.

set -e  # 오류 발생 시 중단

echo "========================================="
echo "ETRI Trajectory Prediction - Training"
echo "========================================="

# 기본 설정
PROJECT_DIR="/workspace/QCNetonETD"
DATA_ROOT="/workspace/ETRITrajPredChallenage/competition_data"
PRETRAINED_WEIGHTS="QCNet_AV2.ckpt"
TRAIN_NAME="train_qcnet_raw_hybrid"
VAL_NAME="val_qcnet_real"

# GPU 설정 (기본값: GPU 3 대 사용)
GPU_DEVICES=${1:-"0,1,2"}
NUM_GPUS=$(echo $GPU_DEVICES | tr ',' '\n' | wc -l)

echo "GPU 설정: $GPU_DEVICES (총 ${NUM_GPUS}개)"
echo "프로젝트 디렉토리: $PROJECT_DIR"
echo "데이터 경로: $DATA_ROOT"
echo ""

# 프로젝트 디렉토리로 이동
cd $PROJECT_DIR

# 데이터 경로 확인
echo "데이터 경로 확인 중..."
if [ ! -d "$DATA_ROOT/$TRAIN_NAME" ]; then
    echo "❌ 오류: 학습 데이터를 찾을 수 없습니다: $DATA_ROOT/$TRAIN_NAME"
    exit 1
fi

if [ ! -d "$DATA_ROOT/$VAL_NAME" ]; then
    echo "❌ 오류: 검증 데이터를 찾을 수 없습니다: $DATA_ROOT/$VAL_NAME"
    exit 1
fi

echo "✅ 데이터 경로 확인 완료"
echo ""

# 사전학습 모델 확인
if [ -f "$PRETRAINED_WEIGHTS" ]; then
    echo "✅ 사전학습 모델 발견: $PRETRAINED_WEIGHTS"
    PRETRAINED_ARG="--pretrained_weights $PRETRAINED_WEIGHTS"
else
    echo "⚠️  사전학습 모델 없음. 처음부터 학습을 시작합니다."
    PRETRAINED_ARG=""
fi
echo ""

# 체크포인트 디렉토리 확인
CHECKPOINT_DIR="checkpoints/submission_run"
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "⚠️  기존 체크포인트 디렉토리가 있습니다: $CHECKPOINT_DIR"
    read -p "계속하시겠습니까? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "학습을 취소합니다."
        exit 1
    fi
fi

# 학습 시작
echo "========================================="
echo "학습 시작..."
echo "========================================="
echo ""

CUDA_VISIBLE_DEVICES=$GPU_DEVICES python train.py \
    --root "$DATA_ROOT" \
    --train_batch_size 16 \
    --val_batch_size 16 \
    --num_workers 2 \
    --devices $NUM_GPUS \
    --max_epochs 250 \
    --wandb_project "submission" \
    --wandb_name "final_submission" \
    $PRETRAINED_ARG \
    2>&1 | tee training_log_$(date +%Y%m%d_%H%M%S).txt

echo ""
echo "========================================="
echo "학습 완료!"
echo "========================================="
echo "체크포인트 저장 위치: $CHECKPOINT_DIR"
echo "로그 파일: training_log_*.txt"
