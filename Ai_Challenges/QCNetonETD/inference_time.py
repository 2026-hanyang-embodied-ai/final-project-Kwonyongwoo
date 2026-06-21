import os
import time
import torch
import argparse
from pathlib import Path
import numpy as np
from torch_geometric.data import DataLoader
from datamodules import ETRIDataModule
from predictors import QCNet

def measure_inference_time(model, dataloader, num_samples=None, device='cuda'):
    """
    모델의 추론 시간을 측정하는 함수
    
    Args:
        model: 추론할 모델
        dataloader: 테스트 데이터로더
        num_samples: 측정할 샘플 수 (None이면 전체)
        device: 사용할 디바이스
    
    Returns:
        dict: 추론 시간 통계
    """
    model.eval()
    model.to(device)
    
    inference_times = []
    total_scenes = 0
    
    print(f"🚀 Starting inference time measurement...")
    print(f"   Device: {device}")
    print(f"   Samples: {'All' if num_samples is None else num_samples}")
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            if num_samples is not None and batch_idx >= num_samples:
                break
                
            # 데이터를 디바이스로 이동
            data = data.to(device)
            
            # 추론 시간 측정 시작
            start_time = time.time()
            
            # 모델 추론 수행
            pred = model(data)
            
            # GPU 동기화 (정확한 시간 측정을 위해)
            if device == 'cuda':
                torch.cuda.synchronize()
            
            # 추론 시간 측정 종료
            end_time = time.time()
            
            # ms 단위로 변환
            inference_time_ms = (end_time - start_time) * 1000.0
            inference_times.append(inference_time_ms)
            
            # 씬 개수 계산 (배치 내 씬의 수)
            if hasattr(data, 'batch_size'):
                batch_scenes = data.batch_size
            elif hasattr(data, 'ptr'):
                batch_scenes = len(data.ptr) - 1
            else:
                batch_scenes = 1
                
            total_scenes += batch_scenes
            
            # 진행 상황 출력
            if (batch_idx + 1) % 10 == 0:
                avg_time = np.mean(inference_times[-10:])
                print(f"   Batch {batch_idx + 1}: Avg time = {avg_time:.2f} ms")
    
    # 통계 계산
    inference_times = np.array(inference_times)
    
    stats = {
        'total_batches': len(inference_times),
        'total_scenes': total_scenes,
        'avg_per_batch_ms': np.mean(inference_times),
        'avg_per_scene_ms': np.sum(inference_times) / total_scenes,
        'min_batch_ms': np.min(inference_times),
        'max_batch_ms': np.max(inference_times),
        'std_batch_ms': np.std(inference_times),
        'median_batch_ms': np.median(inference_times),
        'p95_batch_ms': np.percentile(inference_times, 95),
        'p99_batch_ms': np.percentile(inference_times, 99),
        'total_time_sec': np.sum(inference_times) / 1000.0,
        'throughput_scenes_per_sec': total_scenes / (np.sum(inference_times) / 1000.0)
    }
    
    return stats, inference_times


def print_inference_stats(stats):
    """추론 시간 통계를 출력하는 함수"""
    print("\n" + "="*70)
    print("🎯 INFERENCE TIME MEASUREMENT RESULTS")
    print("="*70)
    print(f"Total Batches:           {stats['total_batches']:,}")
    print(f"Total Scenes:            {stats['total_scenes']:,}")
    print(f"Total Time:              {stats['total_time_sec']:.2f} seconds")
    print("-" * 70)
    print("📊 PER-BATCH STATISTICS (ms)")
    print("-" * 70)
    print(f"Average:                 {stats['avg_per_batch_ms']:.2f}")
    print(f"Minimum:                 {stats['min_batch_ms']:.2f}")
    print(f"Maximum:                 {stats['max_batch_ms']:.2f}")
    print(f"Median:                  {stats['median_batch_ms']:.2f}")
    print(f"Standard Deviation:      {stats['std_batch_ms']:.2f}")
    print(f"95th Percentile:         {stats['p95_batch_ms']:.2f}")
    print(f"99th Percentile:         {stats['p99_batch_ms']:.2f}")
    print("-" * 70)
    print("🎯 PER-SCENE STATISTICS")
    print("-" * 70)
    print(f"Avg Time per Scene:      {stats['avg_per_scene_ms']:.2f} ms")
    print(f"Throughput:              {stats['throughput_scenes_per_sec']:.1f} scenes/sec")
    print("="*70)


def save_results(stats, inference_times, output_dir, model_name):
    """결과를 파일로 저장하는 함수"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 통계 파일 저장
    stats_file = output_dir / f"{model_name}_inference_stats.txt"
    with open(stats_file, 'w') as f:
        f.write("INFERENCE TIME MEASUREMENT RESULTS\n")
        f.write("="*50 + "\n")
        for key, value in stats.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    # 상세 시간 데이터 저장
    times_file = output_dir / f"{model_name}_inference_times.npy"
    np.save(times_file, inference_times)
    
    print(f"📁 Results saved to:")
    print(f"   Stats: {stats_file}")
    print(f"   Times: {times_file}")


def main():
    parser = argparse.ArgumentParser(description="Measure QCNet inference time")
    
    # 모델 관련 arguments
    parser.add_argument('--model_checkpoint', type=str, default='/workspace/ETRITrajPredChallenage/yong.ckpt',
                       help='Path to model checkpoint')
    parser.add_argument('--model_name', type=str, default='QCNet',
                       help='Model name for output files')
    
    # 데이터 관련 arguments  
    parser.add_argument('--root', type=str, default="/workspace/ETRITrajPredChallenage/competition_data")
    parser.add_argument('--train_processed_dir', type=str, default="train_qcnet",
                       help='Train dataset directory (needed by ETRIDataModule)')
    parser.add_argument('--val_processed_dir', type=str, default="val_qcnet",
                       help='Validation dataset directory (needed by ETRIDataModule)')
    parser.add_argument('--test_processed_dir', type=str, default="test_qcnet",
                       help='Test dataset directory')
    parser.add_argument('--train_batch_size', type=int, default=1,
                       help='Train batch size (needed by ETRIDataModule)')
    parser.add_argument('--val_batch_size', type=int, default=1,
                       help='Validation batch size (needed by ETRIDataModule)')
    parser.add_argument('--test_batch_size', type=int, default=1,
                       help='Batch size for testing (1 recommended for accurate timing)')
    parser.add_argument('--shuffle', type=bool, default=False,
                       help='Shuffle data (needed by ETRIDataModule)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of workers (0 recommended for timing)')
    parser.add_argument('--pin_memory', type=bool, default=True,
                       help='Pin memory (needed by ETRIDataModule)')
    parser.add_argument('--persistent_workers', type=bool, default=False,
                       help='Persistent workers (needed by ETRIDataModule)')
    
    # 측정 관련 arguments
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to test (None for all)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--warmup_batches', type=int, default=10,
                       help='Number of warmup batches before measurement')
    
    # 출력 관련 arguments
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='Directory to save results')
    parser.add_argument('--save_results', action='store_true',
                       help='Save detailed results to files')
    
    # 모델 하이퍼파라미터 (QCNet에 필요한 기본 설정)
    parser.add_argument('--dataset', type=str, default='ETRI_Dataset')
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--output_head', action='store_true')
    parser.add_argument('--num_historical_steps', type=int, default=20)
    parser.add_argument('--num_future_steps', type=int, default=60)
    parser.add_argument('--num_modes', type=int, default=6)
    parser.add_argument('--num_recurrent_steps', type=int, default=3)
    parser.add_argument('--pl2pl_radius', type=int, default=150)
    parser.add_argument('--pl2a_radius', type=int, default=50)
    parser.add_argument('--a2a_radius', type=int, default=50)
    parser.add_argument('--pl2m_radius', type=int, default=150)
    parser.add_argument('--a2m_radius', type=int, default=150)
    parser.add_argument('--num_t2m_steps', type=int, default=10)
    parser.add_argument('--time_span', type=int, default=10)
    parser.add_argument('--num_freq_bands', type=int, default=64)
    parser.add_argument('--num_map_layers', type=int, default=2)
    parser.add_argument('--num_agent_layers', type=int, default=2)
    parser.add_argument('--num_dec_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--head_dim', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--T_max', type=int, default=20)
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"🔧 Loading model from: {args.model_checkpoint}")
    
    # 모델 로드
    try:
        model = QCNet(**vars(args))
        checkpoint = torch.load(args.model_checkpoint, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    try:
        datamodule = ETRIDataModule(**vars(args))
        datamodule.setup(stage='test')
        test_dataloader = datamodule.test_dataloader()
        print(f"✅ Test dataloader created with {len(test_dataloader)} batches")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Warmup (첫 몇 배치는 GPU 초기화로 인해 느릴 수 있음)
    if args.warmup_batches > 0:
        print(f"🔥 Warming up with {args.warmup_batches} batches...")
        model.eval()
        model.to(args.device)
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                if i >= args.warmup_batches:
                    break
                data = data.to(args.device)
                _ = model(data)
                if args.device == 'cuda':
                    torch.cuda.synchronize()
        print("✅ Warmup completed")
    
    # 추론 시간 측정
    stats, inference_times = measure_inference_time(
        model=model,
        dataloader=test_dataloader,
        num_samples=args.num_samples,
        device=args.device
    )
    
    # 결과 출력
    print_inference_stats(stats)
    
    # 결과 저장 (옵션)
    if args.save_results:
        save_results(stats, inference_times, args.output_dir, args.model_name)


if __name__ == "__main__":
    main()
