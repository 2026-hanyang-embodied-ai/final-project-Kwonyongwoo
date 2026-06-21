#!/bin/bash

# ETRI 경로 예측 챌린지 - 추론 실행 스크립트
# 이 스크립트는 Docker 컨테이너 내부에서 실행됩니다.

set -e  # 오류 발생 시 중단

echo "========================================="
echo "ETRI Trajectory Prediction - Inference"
echo "========================================="

# 기본 설정
PROJECT_DIR="/workspace/QCNetonETD"
DATA_ROOT="/workspace/ETRITrajPredChallenage/competition_data"
CHECKPOINT_PATH=${1:-"third_model.ckpt"}

# GPU 설정
GPU_DEVICE=${2:-"0"}

echo "GPU 설정: $GPU_DEVICE"
echo "프로젝트 디렉토리: $PROJECT_DIR"
echo "체크포인트: $CHECKPOINT_PATH"
echo ""

# 프로젝트 디렉토리로 이동
cd $PROJECT_DIR

# 체크포인트 파일 확인
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ 오류: 체크포인트 파일을 찾을 수 없습니다: $CHECKPOINT_PATH"
    echo ""
    echo "사용 가능한 체크포인트:"
    find checkpoints/ -name "*.ckpt" -type f
    exit 1
fi

echo "✅ 체크포인트 확인 완료"
echo ""

# 테스트 데이터 확인
if [ ! -d "$DATA_ROOT/test_qcnet" ]; then
    echo "❌ 오류: 테스트 데이터를 찾을 수 없습니다: $DATA_ROOT/test_qcnet"
    exit 1
fi

echo "✅ 테스트 데이터 확인 완료"
echo ""

# 추론 시작
echo "========================================="
echo "추론 시작..."
echo "========================================="
echo ""
cd ..
cd ETRITrajPredChallenage
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python prediction_results_submission.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --root "$DATA_ROOT" \
    --test_batch_size 1 \
    --num_workers 4 \
    2>&1 | tee inference_log_$(date +%Y%m%d_%H%M%S).txt

echo ""
echo "========================================="
echo "추론 완료!"
echo "========================================="
echo "결과 파일: predictions/submission.csv"
echo "로그 파일: inference_log_*.txt"
