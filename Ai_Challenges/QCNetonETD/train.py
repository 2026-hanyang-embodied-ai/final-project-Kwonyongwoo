
from argparse import ArgumentParser

import torch
import torch.multiprocessing as mp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger

from datamodules import ETRIDataModule
from predictors import QCNet

# 메모리 최적화 설정
torch.set_float32_matmul_precision('medium')  # Tensor Core 최적화
torch.backends.cudnn.benchmark = True         # cuDNN 벤치마크 활성화

if __name__ == '__main__':
    # Fix for Bus error: Set multiprocessing start method to 'spawn' for DDP compatibility
    # This must be done before any other multiprocessing code runs
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default="/workspace/Team_1/hun_qcnet/Ai_Challenges/ETRITrajPredChallenage/competition_data") #root는 변환된 QCNet형식 데이터들이 저장된 최상위 폴더를 가리킴 
    parser.add_argument('--train_batch_size', type=int, default=16) # 원본 batch_size=16
    parser.add_argument('--val_batch_size', type=int, default=16) # 원본 batch_size=8
    parser.add_argument('--test_batch_size', type=int, default=1) # 원본 batch_size=1
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=2) # Reduced from 8 to prevent Bus errors with DDP
    parser.add_argument('--devices', type=int, default=3, help='The number of possible GPU devices') 
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_processed_dir', type=str, default="train_qcnet_raw_hybrid")
    parser.add_argument('--val_processed_dir', type=str, default="val_qcnet_real")
    parser.add_argument('--test_processed_dir', type=str, default="test_qcnet")
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--max_epochs', type=int, default=250) #원본 max_epochs=20
    parser.add_argument('--dataset', type=str, default='ETRI_Dataset', help='DO NOT ALTER THIS')
    parser.add_argument('--num_historical_steps', type=int, default=20, help='DO NOT ALTER THIS')
    parser.add_argument('--num_future_steps', type=int, default=60, help='DO NOT ALTER THIS')
    parser.add_argument('--num_recurrent_steps', type=int, default=3, help='DO NOT ALTER THIS')
    parser.add_argument('--pl2pl_radius', type=int, default=150)
    parser.add_argument('--pl2a_radius', type=int, default=50)
    parser.add_argument('--a2a_radius', type=int, default=50)
    parser.add_argument('--pl2m_radius', type=int, default=150)
    parser.add_argument('--a2m_radius', type=int, default=150)
    parser.add_argument('--num_t2m_steps', type=int, default=10)
    parser.add_argument('--time_span', type=int, default=10)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=128) #원본 hidden_dim=128
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--output_head', action='store_true') # default는 False
    parser.add_argument('--num_modes', type=int, default=6, help='DO NOT ALTER THIS')
    parser.add_argument('--num_freq_bands', type=int, default=64)
    parser.add_argument('--num_map_layers', type=int, default=2) #원본 num_map_layers=1
    parser.add_argument('--num_agent_layers', type=int, default=2) #원본 num_agent_layers=2
    parser.add_argument('--num_dec_layers', type=int, default=3) #원본 num_dec_layers=2
    parser.add_argument('--num_heads', type=int, default=8) #원본 num_heads=8 
    parser.add_argument('--head_dim', type=int, default=16) #원본 head_dim=16
    parser.add_argument('--dropout', type=float, default=0.5) #원본 dropout=0.3  #0.3061978796339918
    #parser.add_argument('--dropout', type=float, default=0.261978796339918)
    parser.add_argument('--lr', type=float, default=0.00001329) #원본 lr=1e-5 0.00001368863950364079
    #parser.add_argument('--lr', type=float, default=0.00002368863950364079)
    parser.add_argument('--weight_decay', type=float, default=0.00046187109390049094) #원본 weight_decay=1e-4
    parser.add_argument('--T_max', type=int, default=250) #원본 T_max=64 (max_epochs와 맞춤)
    
    #Wandb 관련 arguments
    parser.add_argument('--use_wandb', action='store_true', default=True, help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='hybrid_trainset', help='WandB project name')
    parser.add_argument('--wandb_name', type=str, default='mag+map+223+16(*2),timeloss', help='WandB run name')
    
    #QCNet github에서 가져온 checkpoint 관련 argument
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to a checkpoint file to resume training from')
    parser.add_argument('--pretrained_weights', type=str, default=None, help='Path to pretrained checkpoint file to load weights from')
    
    #Gradient Accumulation 관련 argument
    parser.add_argument('--accumulate_grad_batches', type=int, default=2, help='Number of batches to accumulate gradients over')
    
    args = parser.parse_args()

    datamodule = {'ETRI_Dataset': ETRIDataModule, }[args.dataset](**vars(args))
    model = QCNet(**vars(args))
    
    # Pretrained weights 로딩 
    if args.pretrained_weights:
        print(f"🔄 Loading pretrained weights from {args.pretrained_weights}")
        try:
            checkpoint = torch.load(args.pretrained_weights, map_location='cpu')
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print("✅ Pretrained weights loaded successfully!! from checkpoint's state_dict")
            else:
                model.load_state_dict(checkpoint, strict=False)
                print("✅ Pretrained weights loaded successfully!! from checkpoint directly")
        except Exception as e:
            print(f"❌ Failed to load pretrained weights: {e}") 
            
            
    # 체크포인트 저장 경로 설정 (wandb_name 폴더 생성)
    checkpoint_dir = f'./checkpoints/{args.wandb_name}' if args.wandb_name else './checkpoints/default'
    model_checkpoint = ModelCheckpoint(dirpath=checkpoint_dir, monitor='val_first_em', save_top_k=4, mode='min', save_last=True) # val_first_em 기준 상위 4개 + 마지막 epoch 1개 ckpt 저장 -> checkpoints/wandb_name/에 저장됨 -> test.py에서 불러다 씀
    last_checkpoint = ModelCheckpoint(dirpath=checkpoint_dir, filename='last_epoch', save_top_k=1, save_last=True, every_n_epochs=1) # 마지막 epoch 체크포인트 저장
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    #Wandb logger 설정
    logger = None
    if args.use_wandb:
        logger = WandbLogger(project=args.wandb_project, name=args.wandb_name, save_dir='./wandb_logs')
        #하이퍼파라미터 로깅 
        logger.log_hyperparams(vars(args))
        print(f"🚀 WandB logging enabled - Project: {args.wandb_project}, Name: {args.wandb_name}")
    else:
        print("📊 Using default PyTorch Lightning logger")
        
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),  # unused parameters 허용
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs, logger=logger,
                        )       
    #trainer.fit(model, datamodule) 
    
    # Checkpoint에서 이어서 학습
    trainer.fit(model, datamodule, ckpt_path=args.resume_from_checkpoint) if args.resume_from_checkpoint else trainer.fit(model, datamodule)