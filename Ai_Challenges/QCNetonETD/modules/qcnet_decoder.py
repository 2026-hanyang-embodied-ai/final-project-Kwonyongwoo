
import math
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse

from layers import AttentionLayer
from layers import FourierEmbedding
from layers import MLPLayer
from utils import angle_between_2d_vectors
from utils import bipartite_dense_to_sparse
from utils import weight_init
from utils import wrap_angle

from torch.nn.utils.rnn import pad_sequence

class QCNetDecoder(nn.Module):

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
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 num_freq_bands: int,
                 num_dec_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 **kwargs) -> None:
        super(QCNetDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_t2m_steps = num_t2m_steps if num_t2m_steps is not None else num_historical_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_dec_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        input_dim_r_t = 4
        input_dim_r_pl2m = 3
        input_dim_r_a2m = 3

        self.mode_emb = nn.Embedding(num_modes, hidden_dim)
        # Map context embedding (9 map types: 0-8) 
        #역할 : 9개의 서로 다른 map_type을 hidden_dim크기의 벡터로 변환하는 embedding
        #self.map_type_emb = nn.Embedding(9, hidden_dim)
        
        self.r_t2m_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2m_emb = FourierEmbedding(input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2m_emb = FourierEmbedding(input_dim=input_dim_r_a2m, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.y_emb = FourierEmbedding(input_dim=output_dim + output_head, hidden_dim=hidden_dim,
                                      num_freq_bands=num_freq_bands)
        self.traj_emb = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                               batch_first=False, dropout=0.0, bidirectional=False)
        self.traj_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        
        #Proposal 단계 (Anchored-Free)
        self.t2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(self.num_layers)]
        )
        self.pl2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(self.num_layers)]
        )
        self.a2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(self.num_layers)]
        )
        self.m2m_propose_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                     dropout=dropout, bipartite=False, has_pos_emb=False)
        
        #Refinement 단계 (Anchor-based)
        self.t2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(self.num_layers)]
        )
        self.pl2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(self.num_layers)]
        )
        self.a2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(self.num_layers)]
        )
        self.m2m_refine_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                    dropout=dropout, bipartite=False, has_pos_emb=False)
        self.to_loc_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim=num_future_steps * output_dim // num_recurrent_steps)
        self.to_scale_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                             output_dim=num_future_steps * output_dim // num_recurrent_steps)
        self.to_loc_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                          output_dim=num_future_steps * output_dim)
        self.to_scale_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                            output_dim=num_future_steps * output_dim)
        if output_head:
            self.to_loc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps // num_recurrent_steps)
            self.to_conc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                 output_dim=num_future_steps // num_recurrent_steps)
            self.to_loc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_future_steps)
            self.to_conc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps)
        else:
            self.to_loc_propose_head = None
            self.to_conc_propose_head = None
            self.to_loc_refine_head = None
            self.to_conc_refine_head = None
        self.to_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
        self.apply(weight_init)

    # def forward(self,
    #             data: HeteroData,
    #             scene_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

    def forward(self, pos_m: torch.Tensor,
                head_m: torch.Tensor,
                x_t: torch.Tensor,
                x_pl: torch.Tensor,
                x_a: torch.Tensor,
                mask_src: torch.Tensor,
                mask_dst: torch.Tensor,
                pos_t: torch.Tensor,
                head_t: torch.Tensor,
                pos_pl: torch.Tensor,
                orient_pl: torch.Tensor,
                agent_batch: torch.Tensor,
                map_polygon_batch: torch.Tensor,
                map_num_nodes: int, #Map_num_nodes=scene내의 모든 polygon 수) 
                agent_num_nodes: int) -> Dict[str, torch.Tensor]:

        # map_num_nodes = data['map_polygon']['num_nodes'].sum().item()
        # agent_num_nodes = data['agent']['num_nodes'].sum().item()
        #
        # pos_m = data['agent']['position'][:, self.num_historical_steps - 1, :self.input_dim]
        # head_m = data['agent']['heading'][:, self.num_historical_steps - 1]
        #
        # x_t = scene_enc['x_a'].reshape(-1, self.hidden_dim)
        # x_pl = scene_enc['x_pl'][:, self.num_historical_steps - 1].repeat(self.num_modes, 1)
        # x_a = scene_enc['x_a'][:, -1].repeat(self.num_modes, 1)
        #
        # mask_src = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()
        # mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
        # mask_dst = data['agent']['predict_mask'].any(dim=-1, keepdim=True).repeat(1, self.num_modes)
        #
        # pos_t = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].reshape(-1, self.input_dim)
        # head_t = data['agent']['heading'][:, :self.num_historical_steps].reshape(-1)
        #
        # pos_pl = data['map_polygon']['position'][:, :self.input_dim]
        # orient_pl = data['map_polygon']['orientation']
        #
        # agent_batch = data['agent']['batch'] if isinstance(data, Batch) else None
        # map_polygon_batch = data['map_polygon']['batch'] if isinstance(data, Batch) else None

        #m = self.mode_emb.weight.repeat(agent_num_nodes, 1) #배치 내 총 에이전트 수 × 모드 수 만큼 mode query 생성 (Axnum_modes, hidden_dim)
#======================================================================================================
        # --- Efficient Map-Aware Mode Embedding ---
        # 기본 mode embedding 생성 (A*M, H)
        m = self.mode_emb.weight.repeat(agent_num_nodes, 1)  # (A*M, H)
        
        # Map context embedding (A, H) → (A*M, H)로 확장(모드 수 만큼 복제해서 확장)
        #map_context_emb = self.map_type_emb(map_agent_type)  # (A, H)=(agent_num_nodes, hidden_dim)
        #map_context_expanded = map_context_emb.repeat_interleave(self.num_modes, dim=0)  # (A*M, H) (각 agent의 map context를 mode 수만큼 복제)
        #
        ## 학습 가능한 융합 가중치 (0~1 범위)
        #fusion_weight = torch.sigmoid(torch.sum(m * map_context_expanded, dim=-1, keepdim=True) / self.hidden_dim)  # (A*M, 1)
        #
        ## Map context와 mode embedding의 가중 결합
        #m = m + fusion_weight * map_context_expanded  # (A*M, H)
#======================================================================================================

        head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1)
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst[:, -1:].unsqueeze(1))
        
        #좌표 프레임 변환(각 대상 에이전트의 현재 시점으로 투영)
        #모든 상대 위치를 각 대상 에이전트 중심으로 계산 (에이전트별로 독립적인 좌표계에서 예측 수행)
        
        #==================================================================================================
        #1. r_t2m (Relative Positional Embedding for Trajectory-to-Mode)
        #==================================================================================================
        rel_pos_t2m = pos_t[edge_index_t2m[0]] - pos_m[edge_index_t2m[1]] # 과거 궤적과 Mode queries 간 상대 위치
        rel_head_t2m = wrap_angle(head_t[edge_index_t2m[0]] - head_m[edge_index_t2m[1]]) #상대 방향
        r_t2m = torch.stack(
            [torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1), #거리
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2m[1]], nbr_vector=rel_pos_t2m[:, :2]), #각도
             rel_head_t2m, #방향
             (edge_index_t2m[0] % self.num_historical_steps) - self.num_historical_steps + 1], dim=-1) #시간 차이
        
        #의미: 과거 궤적과 현재 mode간의 상대적 위치, 방향, 시간 관계를 인코딩 
        r_t2m = self.r_t2m_emb(continuous_inputs=r_t2m.float(), categorical_embs=None) # (E_t2m, hidden_dim) 
        
        #의미: 어떤 과거 궤적 포인트가 어떤 mode query와 연결되는지 정의하는 연결 그래프 
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst.unsqueeze(1)) # shape: (2, E_t2m) - [source_indices, target_indices] = [trajectory points, mode queries]
        r_t2m = r_t2m.repeat_interleave(repeats=self.num_modes, dim=0)


        #====================================================================================================
        #2. r_pl2m (Relative Positional Embedding for MapPolygon-to-Mode)
        #====================================================================================================
        pos_m = pos_m.double()
        pos_pl = pos_pl.double()
        edge_index_pl2m = radius(
            x=pos_m[:, :2], #Agent 위치
            y=pos_pl[:, :2], #Map polygon 위치
            r=self.pl2m_radius, #반경 (150m)
            batch_x=agent_batch,
            batch_y=map_polygon_batch,
            max_num_neighbors=300)
        edge_index_pl2m = edge_index_pl2m[:, mask_dst[edge_index_pl2m[1], 0]]
        rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_m[edge_index_pl2m[1]]
        rel_orient_pl2m = wrap_angle(orient_pl[edge_index_pl2m[0]] - head_m[edge_index_pl2m[1]])
        r_pl2m = torch.stack(
            [torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2m[1]], nbr_vector=rel_pos_pl2m[:, :2]),
             rel_orient_pl2m], dim=-1)
        r_pl2m = self.r_pl2m_emb(continuous_inputs=r_pl2m.float(), categorical_embs=None)



        edge_index_pl2m = torch.cat([
            edge_index_pl2m + i * edge_index_pl2m.new_tensor([[map_num_nodes], [agent_num_nodes]])
            for i in range(self.num_modes)
        ], dim=1)
        r_pl2m = r_pl2m.repeat(self.num_modes, 1)

        #====================================================================================================
        #3. r_a2m (Relative Positional Embedding for OtherAgent-to-Mode)
        #====================================================================================================
        edge_index_a2m = radius_graph(
            x=pos_m[:, :2],
            r=self.a2m_radius,
            batch=agent_batch,
            loop=False,
            max_num_neighbors=300)
        edge_index_a2m = edge_index_a2m[:, mask_src[:, -1][edge_index_a2m[0]] & mask_dst[edge_index_a2m[1], 0]]
        rel_pos_a2m = pos_m[edge_index_a2m[0]] - pos_m[edge_index_a2m[1]]
        rel_head_a2m = wrap_angle(head_m[edge_index_a2m[0]] - head_m[edge_index_a2m[1]])
        r_a2m = torch.stack(
            [torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_a2m[1]], nbr_vector=rel_pos_a2m[:, :2]),
             rel_head_a2m], dim=-1)
        r_a2m = self.r_a2m_emb(continuous_inputs=r_a2m.float(), categorical_embs=None)

        offset = edge_index_a2m.new_tensor(agent_num_nodes)  # make it a scalar tensor with correct dtype/device

        edge_index_a2m = torch.cat([
            edge_index_a2m + i * offset for i in range(self.num_modes)
        ], dim=1)
        r_a2m = r_a2m.repeat(self.num_modes, 1)

        edge_index_m2m = dense_to_sparse(mask_dst.unsqueeze(2) & mask_dst.unsqueeze(1))[0]

        locs_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        scales_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        locs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        concs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        
        #====================================================================================
        #1-A. Recurrence x T_rec 만큼 반복해서 Anchor-free한 Proposal 생성
        #====================================================================================
        for t in range(self.num_recurrent_steps):
            for i in range(self.num_layers):
                m = m.reshape(-1, self.hidden_dim) #mode query 준비 (AxM, hidden_dim) (M=num_modes)
                
                #=================================================================
                #A-a. mode2scene cross-attention
                #=================================================================
                #A-a-1. ---Trajectory -> Mode cross-attention--- (temporal attn같은 느낌)
                    #Query: m [mode queries (A×M, hidden_dim)]
                    #Key, Value: x_t [trajectory features (A×T_t, hidden_dim)]
                    #output: update된 m [(A×M, hidden_dim)]
                    #의미: 각 mode query가 과거 궤적 정보 참고해서 update
                m = self.t2m_propose_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
                
                m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim) # (AxM, hidden_dim) → (M×A, hidden_dim) [reshape 과정]
                
                
                #A-a-2. ---Map polygon -> Mode cross-attention--- (agent-map attn같은 느낌)
                    #Query: m [mode queries (M×A, hidden_dim)]
                    #Key, Value: x_pl [map polygon features (A×P, hidden_dim)] (P: Map_num_nodes=scene내의 모든 polygon 수) 
                    #output: update된 m [(M×A, hidden_dim)]
                    #의미: 각 mode query가 map polygon(주변 도로, 차선 정보) 정보 참고해서 update
                m = self.pl2m_propose_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                
                #A-a-3. ---Other Agent -> Mode cross-attention--- (social attn같은 느낌)
                    #Query: m [mode queries (M×A, hidden_dim)]
                    #Key, Value: x_a [other agent features (A×1, hidden_dim)] (1: 각 에이전트의 마지막 시점 feature)
                    #output: update된 m [(M×A, hidden_dim)]
                    #의미: 각 mode query가 other agent 정보 참고해서 update
                m = self.a2m_propose_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
                
                m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim) # (M×A, hidden_dim) → (A×M, hidden_dim) [reshape 과정]
            #=======================================================
            #A-b. mode2mode self-attention
            #=======================================================
            m = self.m2m_propose_attn_layer(m, None, edge_index_m2m)
            m = m.reshape(-1, self.num_modes, self.hidden_dim)
            locs_propose_pos[t] = self.to_loc_propose_pos(m)
            scales_propose_pos[t] = self.to_scale_propose_pos(m)
            if self.output_head:
                locs_propose_head[t] = self.to_loc_propose_head(m)
                concs_propose_head[t] = self.to_conc_propose_head(m)
        loc_propose_pos = torch.cumsum(
            torch.cat(locs_propose_pos, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
            dim=-2)
        scale_propose_pos = torch.cumsum(
            F.elu_(
                torch.cat(scales_propose_pos, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
                alpha=1.0) +
            1.0,
            dim=-2) + 0.1
        if self.output_head:
            loc_propose_head = torch.cumsum(torch.tanh(torch.cat(locs_propose_head, dim=-1).unsqueeze(-1)) * math.pi,
                                            dim=-2)
            conc_propose_head = 1.0 / (torch.cumsum(F.elu_(torch.cat(concs_propose_head, dim=-1).unsqueeze(-1)) + 1.0,
                                                    dim=-2) + 0.02)
            m = self.y_emb(torch.cat([loc_propose_pos.detach(),
                                      wrap_angle(loc_propose_head.detach())], dim=-1).view(-1, self.output_dim + 1))
        else:
            loc_propose_head = loc_propose_pos.new_zeros((loc_propose_pos.size(0), self.num_modes,
                                                          self.num_future_steps, 1))
            conc_propose_head = scale_propose_pos.new_zeros((scale_propose_pos.size(0), self.num_modes,
                                                             self.num_future_steps, 1))
            m = self.y_emb(loc_propose_pos.detach().view(-1, self.output_dim))
        m = m.reshape(-1, self.num_future_steps, self.hidden_dim).transpose(0, 1)
        m = self.traj_emb(m, self.traj_emb_h0.unsqueeze(1).repeat(1, m.size(1), 1))[1].squeeze(0)
        
        #====================================================================================
        #numlayers 만큼 반복해서 Anchor-based한 Refinement 생성
        #====================================================================================
        for i in range(self.num_layers):
            #mode2scene cross-attention
            m = self.t2m_refine_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
            m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m = self.pl2m_refine_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
            m = self.a2m_refine_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
            m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            
        #mode2mode self-attention
        m = self.m2m_refine_attn_layer(m, None, edge_index_m2m)
        m = m.reshape(-1, self.num_modes, self.hidden_dim)
        loc_refine_pos = self.to_loc_refine_pos(m).view(-1, self.num_modes, self.num_future_steps, self.output_dim)
        loc_refine_pos = loc_refine_pos + loc_propose_pos.detach()
        scale_refine_pos = F.elu_(
            self.to_scale_refine_pos(m).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
            alpha=1.0) + 1.0 + 0.1
        if self.output_head:
            loc_refine_head = torch.tanh(self.to_loc_refine_head(m).unsqueeze(-1)) * math.pi
            loc_refine_head = loc_refine_head + loc_propose_head.detach()
            conc_refine_head = 1.0 / (F.elu_(self.to_conc_refine_head(m).unsqueeze(-1)) + 1.0 + 0.02)
        else:
            loc_refine_head = loc_refine_pos.new_zeros((loc_refine_pos.size(0), self.num_modes, self.num_future_steps,
                                                        1))
            conc_refine_head = scale_refine_pos.new_zeros((scale_refine_pos.size(0), self.num_modes,
                                                           self.num_future_steps, 1))
        pi = self.to_pi(m).squeeze(-1)

        return {
            'loc_propose_pos': loc_propose_pos, # this represents the proposed position of agents in the future timesteps
            'scale_propose_pos': scale_propose_pos, # indicates how much the proposed position might change
            'loc_propose_head': loc_propose_head, # in which direction the agents move in the future (proposed headings)
            'conc_propose_head': conc_propose_head, # confidence of the model for the proposed head
            'loc_refine_pos': loc_refine_pos,
            'scale_refine_pos': scale_refine_pos,
            'loc_refine_head': loc_refine_head,
            'conc_refine_head': conc_refine_head,
            'pi': pi,
        }
