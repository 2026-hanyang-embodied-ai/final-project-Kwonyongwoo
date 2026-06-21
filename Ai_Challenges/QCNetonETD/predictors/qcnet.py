import os
import pickle
from itertools import chain
from itertools import compress
from pathlib import Path
from typing import Optional, Mapping
from torch.nn.utils.rnn import pad_sequence
import time
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from losses import MixtureNLLLoss
from losses import NLLLoss
from metrics import Brier
from metrics import MR
from metrics import minADE
from metrics import minAHE
from metrics import minFDE
from metrics import minFHE
from modules import QCNetDecoder
from modules import QCNetEncoder
#%%
try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object


class QCNet(pl.LightningModule):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps: int,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_dec_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 **kwargs) -> None:
        super(QCNet, self).__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_freq_bands = num_freq_bands
        self.num_map_layers = num_map_layers
        self.num_agent_layers = num_agent_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.pl2pl_radius = pl2pl_radius      # polygon_to_polygon
        self.time_span = time_span
        self.pl2a_radius = pl2a_radius     #polygon_to_agent
        self.a2a_radius = a2a_radius        #agent_to_agent
        self.num_t2m_steps = num_t2m_steps    # num of t
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max

        # agent_type 가중치
        self.vehicle_weight = 1.0
        self.pedestrian_weight = 0.0
        self.cyclist_weight = 0.0

        self.test_time_spent = 0
        self.num_test_scenes = 0

        self.encoder = QCNetEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_map_layers=num_map_layers,
            num_agent_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.decoder = QCNetDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            output_head=output_head,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            num_modes=num_modes,
            num_recurrent_steps=num_recurrent_steps,
            num_t2m_steps=num_t2m_steps,
            pl2m_radius=pl2m_radius,
            a2m_radius=a2m_radius,
            num_freq_bands=num_freq_bands,
            num_dec_layers=num_dec_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        # == Loss ==
        self.reg_loss = NLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                reduction='none')
        self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                       reduction='none')

        # == Metrics ==
        self.minADE1 = minADE(max_guesses=1)
        self.minADE6 = minADE(max_guesses=num_modes)
        self.minFDE1 = minFDE(max_guesses=1)
        self.minFDE6 = minFDE(max_guesses=num_modes)
        self.first_em = 0.5 * (self.minADE1 + self.minADE6)

        self.validation_results = []
        self.test_results = []

    def prepare_decoder_inputs(self, data: HeteroData, scene_enc: Mapping[str, torch.Tensor]):

        pos_m = data['agent']['position'][:, self.num_historical_steps - 1, :self.input_dim]
        head_m = data['agent']['heading'][:, self.num_historical_steps - 1]

        x_t = scene_enc['x_a'].reshape(-1, self.hidden_dim)
        x_pl = scene_enc['x_pl'][:, self.num_historical_steps - 1].repeat(self.num_modes, 1)
        x_a = scene_enc['x_a'][:, -1].repeat(self.num_modes, 1)

        mask_src = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()
        mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
        mask_dst = data['agent']['predict_mask'].any(dim=-1, keepdim=True).repeat(1, self.num_modes)

        pos_t = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].reshape(-1, self.input_dim)
        head_t = data['agent']['heading'][:, :self.num_historical_steps].reshape(-1)

        pos_pl = data['map_polygon']['position'][:, :self.input_dim]
        orient_pl = data['map_polygon']['orientation']

        agent_batch = data['agent']['batch'] if isinstance(data, Batch) else None
        map_polygon_batch = data['map_polygon']['batch'] if isinstance(data, Batch) else None

        map_num_nodes = data['map_polygon']['num_nodes']
        agent_num_nodes = data['agent']['num_nodes']

        return (pos_m, head_m, x_t, x_pl, x_a, mask_src, mask_dst, pos_t, head_t, pos_pl, orient_pl, agent_batch,
                map_polygon_batch, map_num_nodes, agent_num_nodes)

    def forward(self, data: HeteroData):
        scene_enc = self.encoder(data)
        pred = self.decoder(*self.prepare_decoder_inputs(data, scene_enc))
        return pred

    def jerk_loss(self, trajectory, weight=50, type_weights=None):
        """
        궤적의 3차 미분(jerk)을 최소화하여 부드러운 궤적 생성
    
        Args:
            trajectory: Tensor of shape [N, T, 2] - 예측된 궤적
            weight: jerk loss의 가중치
            type_weights: Tensor of shape [N] - agent type별 가중치 (optional)
    
        Returns:
            weighted jerk loss
        """
        # 1차 미분 (속도) - shape: [N, T-1, 2]
        velocity = torch.diff(trajectory, n=1, dim=1)
        # 2차 미분 (가속도) - shape: [N, T-2, 2]
        acceleration = torch.diff(velocity, n=1, dim=1) 
        # 3차 미분 (jerk - 가속도 변화율) - shape: [N, T-3, 2]
        jerk = torch.diff(acceleration, n=1, dim=1)
    
        # L2 norm으로 jerk 크기 계산 - shape: [N, T-3]
        jerk_norm = torch.norm(jerk, dim=-1)
        
        # Agent type별 가중치 적용
        if type_weights is not None:
            # type_weights: [N] -> [N, 1]로 확장하여 timestep 차원에 브로드캐스팅
            jerk_norm = jerk_norm * type_weights.unsqueeze(1)
            loss = jerk_norm.sum() / type_weights.sum().clamp_(min=1) / jerk_norm.size(1)
        else:
            loss = torch.mean(jerk_norm)
        
        return loss * weight

    def training_step(self, data, batch_idx):

        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]

        # agent_types -> tensor
        agent_types = data['agent']['type']
        if isinstance(agent_types, list) and len(agent_types) > 0 and hasattr(agent_types[0], 'shape'):
            import numpy as np
            agent_types_flat = []
            for batch_categories in agent_types:
                agent_types_flat.extend(batch_categories.flatten())
            agent_types = torch.tensor(agent_types_flat, device=data['agent']['position'].device, dtype=torch.long)
        elif not isinstance(agent_types, torch.Tensor):
            if hasattr(agent_types, 'shape'):
                agent_types = torch.from_numpy(agent_types).to(data['agent']['position'].device).long()
            else:
                agent_types = torch.tensor(agent_types, device=data['agent']['position'].device, dtype=torch.long)

        type_weights = torch.ones_like(agent_types, dtype=torch.float32)
        type_weights[agent_types == 0] = self.vehicle_weight
        type_weights[agent_types == 1] = self.pedestrian_weight
        type_weights[agent_types == 2] = self.cyclist_weight

        pred = self(data)
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']

        gt = torch.cat([data['agent']['target'][..., :self.output_dim],
                        data['agent']['target'][..., -1:]], dim=-1)


        # === Refine 기준으로 좋은 mode를 선택하는 방식 ===
        # 1단계: 확률 기반으로 trajectory 정렬 (가장 높은 확률 모드를 index 0으로)
        pi_softmax = F.softmax(pi, dim=-1)
        order = torch.argsort(pi_softmax, dim=-1, descending=True)  # 확률 높은 순서

        batch_size = traj_propose.size(0)
        gidx_propose = order.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, traj_propose.size(2), traj_propose.size(3))
        gidx_refine = order.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, traj_refine.size(2), traj_refine.size(3))

        traj_propose_sorted = torch.gather(traj_propose, 1, gidx_propose)
        traj_refine_sorted = torch.gather(traj_refine, 1, gidx_refine)

        # 2단계: refine trajectory에서 GT와의 거리 계산 (reg_loss 기준)
        l2_norm_sorted_refine = (torch.norm(traj_refine_sorted[..., :self.output_dim] -
                            gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)

################################################
        # use_mode0 = torch.rand(batch_size, device=traj_propose.device) < 0.0
        
        
        # # Mode 0 선택 (0% 확률) - 이제 Mode 0는 가장 확률 높은 모드
        # mode0_selection = torch.zeros(batch_size, dtype=torch.long, device=traj_propose.device)
        
        # # 거리 기반 선택 (100% 확률) - 정렬된 상태에서 GT와 가장 가까운 모드
        # distance_selection = l2_norm_sorted_refine.argmin(dim=-1)
        
        # # 조건부 선택
        # best_mode = torch.where(use_mode0, mode0_selection, distance_selection)
        
        # traj_propose_best = traj_propose_sorted[torch.arange(batch_size), best_mode]
        # traj_refine_best = traj_refine_sorted[torch.arange(batch_size), best_mode]

################################################
       
        # 3단계: refine 기준으로 가장 GT와 가까운 mode 선택 
        best_mode = l2_norm_sorted_refine.argmin(dim=-1)

        traj_propose_best = traj_propose_sorted[torch.arange(batch_size), best_mode]
        traj_refine_best = traj_refine_sorted[torch.arange(batch_size), best_mode]

        # 시간대별 가중치 생성 (10Hz, 60 timesteps = 6초)
        # 0~2초 (0~20 steps): 1.5배
        # 2~4초 (20~40 steps): 1.0배
        # 4~6초 (40~60 steps): 2.0배
        time_weights = torch.ones(self.num_future_steps, device=traj_propose_best.device)
        time_weights[:20] = 2.0   # 0~2초: 높은 가중치
        time_weights[20:40] = 1.0  # 2~4초: 보통 가중치
        time_weights[40:] = 1.5    # 4~6초: 가장 높은 가중치

        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        # 시간 가중치 적용
        reg_loss_propose = reg_loss_propose * time_weights.unsqueeze(0)
        reg_loss_propose = (reg_loss_propose * type_weights.unsqueeze(1)).sum(dim=0) / \
                           (reg_mask * type_weights.unsqueeze(1) * time_weights.unsqueeze(0)).sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()

        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        # 시간 가중치 적용
        reg_loss_refine = reg_loss_refine * time_weights.unsqueeze(0)
        reg_loss_refine = (reg_loss_refine * type_weights.unsqueeze(1)).sum(dim=0) / \
                          (reg_mask * type_weights.unsqueeze(1) * time_weights.unsqueeze(0)).sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()

        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim + self.output_head],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = (cls_loss * type_weights.unsqueeze(1)).sum() / \
                   (cls_mask * type_weights.unsqueeze(1)).sum().clamp_(min=1)

        self.log('train_reg_loss_propose', reg_loss_propose, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_reg_loss_refine', reg_loss_refine, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss', cls_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        
        # Jerk Loss 추가 - 궤적의 부드러움 제약 (agent type별 가중치 적용)
        jerk_loss_propose = self.jerk_loss(
            traj_propose_best[..., :self.output_dim], 
            weight=100,
            type_weights=type_weights
        )
        jerk_loss_refine = self.jerk_loss(
            traj_refine_best[..., :self.output_dim], 
            weight=100,
            type_weights=type_weights
        )
        jerk_total = jerk_loss_propose + jerk_loss_refine
        
        self.log('train_jerk_loss', jerk_total, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        
        # 최종 Loss 계산 (기존 loss + jerk loss)
        loss = reg_loss_propose + reg_loss_refine + cls_loss + jerk_total
        self.log('train_loss_total', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, data, batch_idx):

        # forward
        pred = self(data)

        # prepare GT
        gt = torch.cat([data['agent']['target'][..., :self.output_dim],
                        data['agent']['target'][..., -1:]], dim=-1)

        # gather predictions
        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)

        # 평가 대상: category == 2
        eval_mask_list = []
        all_categories = data['agent']['category']
        for categories in all_categories:
            categories = np.array(categories)
            mask = categories == 2
            eval_mask_list.append(mask)
        eval_mask_np = np.concatenate(eval_mask_list)
        dev = data['agent']['position'].device
        eval_mask = torch.tensor(eval_mask_np, dtype=torch.bool, device=dev)

        # 유효 타임스텝
        valid_mask_future_horizon = data['agent']['predict_mask'][:, self.num_historical_steps:]
        valid_mask_eval = valid_mask_future_horizon[eval_mask]

        # 평가용 텐서
        traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]
        if not self.output_head:
            traj_2d_with_start_pos_eval = torch.cat(
                [traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                 traj_eval[..., :2]], dim=-2)
            motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
            head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
            traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)

        pi_eval = F.softmax(pred['pi'][eval_mask], dim=-1)
        gt_eval = gt[eval_mask]

        # === 기존 minADE/minFDE (확률 사용) ===
        self.minADE1.update(pred=traj_eval[..., :self.output_dim],
                            target=gt_eval[..., :self.output_dim],
                            prob=pi_eval,
                            valid_mask=valid_mask_eval)
        self.minADE6.update(pred=traj_eval[..., :self.output_dim],
                            target=gt_eval[..., :self.output_dim],
                            prob=pi_eval,
                            valid_mask=valid_mask_eval)
        self.minFDE1.update(pred=traj_eval[..., :self.output_dim],
                            target=gt_eval[..., :self.output_dim],
                            prob=pi_eval,
                            valid_mask=valid_mask_eval)
        self.minFDE6.update(pred=traj_eval[..., :self.output_dim],
                            target=gt_eval[..., :self.output_dim],
                            prob=pi_eval,
                            valid_mask=valid_mask_eval)

        # 🔧 Step별 로깅은 제거하고 epoch 단위에서만 처리 (on_validation_epoch_end에서)
        # 메트릭은 자동으로 누적되고 epoch 끝에서 계산됨
        pass  # validation_step에서는 메트릭 update만 하고 로깅은 epoch_end에서

        # === 제출 방식 점검: pi로 정렬 후 mode0-only ADE ===
        order = torch.argsort(pi_eval, dim=-1, descending=True)  # (N, M)
        gidx = order.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, traj_eval.size(2), traj_eval.size(3))
        traj_eval_sorted = torch.gather(traj_eval, dim=1, index=gidx)  # (N, M, T, D)

        pred_mode0 = traj_eval_sorted[:, 0, :, :self.output_dim]       # (N, T, 2)
        gt_pos = gt_eval[:, :, :self.output_dim]                        # (N, T, 2)
        valid = valid_mask_eval                                         # (N, T) [bool]

        ade_per_agent = ((pred_mode0 - gt_pos).pow(2).sum(-1).sqrt() * valid).sum(-1) \
                        / valid.sum(-1).clamp(min=1)
        val_minADE1_mode0 = ade_per_agent.mean()
        self.log('val_minADE1_mode0', val_minADE1_mode0, prog_bar=True,
                 on_step=False, on_epoch=True, batch_size=gt_eval.size(0))

        # === 진단: argmax(pi)가 ADE 기준 베스트 모드를 맞췄는가? ===
        gt_pos_exp = gt_pos.unsqueeze(1)                 # (N, 1, T, 2)
        valid_exp = valid.unsqueeze(1).float()           # (N, 1, T)

        dist = (traj_eval[..., :self.output_dim] - gt_pos_exp).pow(2).sum(-1).sqrt()  # (N, M, T)
        ade_all = (dist * valid_exp).sum(-1) / valid_exp.sum(-1).clamp(min=1)         # (N, M)
        best_mode_by_ADE = ade_all.argmin(dim=1)                                      # (N,)
        top1_pi_hits_ADE = (order[:, 0] == best_mode_by_ADE).float().mean()
        self.log('val_top1_pi_hits_ADE', top1_pi_hits_ADE, prog_bar=True,
                 on_step=False, on_epoch=True, batch_size=gt_eval.size(0))

    def on_validation_epoch_end(self):
        # 📊 Epoch 단위로 메트릭 최종 계산 및 로깅 (더 정확한 WandB 시각화를 위해)
        val_minADE1 = self.minADE1.compute()
        val_minADE6 = self.minADE6.compute() 
        val_minFDE1 = self.minFDE1.compute()
        val_minFDE6 = self.minFDE6.compute()
        val_first_em = 0.5 * (val_minADE1 + val_minADE6)
        
        # WandB에 최종 값 로깅 (prefix 추가로 구분)
        self.log('val_minADE1', val_minADE1, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_minADE6', val_minADE6, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_minFDE1', val_minFDE1, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_minFDE6', val_minFDE6, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_first_em', val_first_em, on_step=False, on_epoch=True, prog_bar=False)
        
        # 디버깅용 출력
        print(f"\n📊 Validation Epoch {self.current_epoch} Results:")
        print(f"   minADE1: {val_minADE1:.4f}")
        print(f"   minADE6: {val_minADE6:.4f}")
        print(f"   minFDE1: {val_minFDE1:.4f}")
        print(f"   minFDE6: {val_minFDE6:.4f}")
        print(f"   first_em: {val_first_em:.4f}")
        
        # 메트릭 초기화 (다음 epoch을 위해)
        self.minADE1.reset()
        self.minADE6.reset()
        self.minFDE1.reset()
        self.minFDE6.reset()

    def test_step(self, data, batch_idx):


        # measure inference time (ms) -------
        start = time.time()
        pred = self(data)
        end = time.time()
        self.test_time_spent += (end - start) * 1000.0
        # measure inference time (ms) -------

        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)

        # reorder modes by probability so that mode-0 is the most probable (helps top-1 metrics on submission)
        if 'pi' in pred:
            with torch.no_grad():
                pi = F.softmax(pred['pi'], dim=-1)  # [N, M]
                order = torch.argsort(pi, dim=-1, descending=True)  # [N, M]
                # gather along mode dimension (dim=1)
                gather_index = order.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, traj_refine.size(2), traj_refine.size(3))
                traj_refine = torch.gather(traj_refine, dim=1, index=gather_index)

        # agent-centric to global coordinate system
        origin_eval = data['agent']['position'][:, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][:, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(data['agent']['num_nodes'], 2, 2, device=self.device)
        rot_mat[:, 0, 0], rot_mat[:, 0, 1] = cos, sin
        rot_mat[:, 1, 0], rot_mat[:, 1, 1] = -sin, cos
        traj_eval_pred = torch.matmul(traj_refine[:, :, :, :2], rot_mat.unsqueeze(1)) \
                         + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        traj_eval_pred = traj_eval_pred.cpu()

        # save test results
        # start_end_indices = np.cumsum(np.insert(data['agent']['num_nodes'].to('cpu').numpy(), 0, 0))
        start_end_indices = data['agent']['ptr'].cpu().numpy()

        for idx, (start, end) in enumerate(zip(start_end_indices[:-1], start_end_indices[1:])):
            agent = {
                'num_nodes': end - start,
                'num_valid_nodes': data['agent']['num_valid_nodes'][idx].item(),
                'id': data['agent']['id'][idx],
                'category': data['agent']['category'][idx],
                'predictions': traj_eval_pred[start:end].numpy()
            }

            scene = {
                'log_id': data['log_id'][idx],
                'frm_idx': data['frm_idx'][idx].item(),
                'agent': agent
            }

            self.num_test_scenes += 1
            self.test_results.append(scene)

    
    def on_test_end(self):

        results_dir = os.path.join(self.trainer.log_dir, 'test_results')
        os.makedirs(results_dir, exist_ok=True)

        for scene in self.test_results:
            log_id, frm_idx = scene['log_id'], scene['frm_idx']
            results_path = os.path.join(results_dir, f'log_{log_id}_{frm_idx:07d}_submission.pkl')
            with open(results_path, 'wb') as f:
                pickle.dump(scene, f)
        self.test_results.clear()

        avg_inf_time = self.test_time_spent / self.num_test_scenes
        print(f">> Average Inference Time Per Scene (ms) : {avg_inf_time:.2f}")

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]