
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import subgraph

from layers.attention_layer import AttentionLayer
from layers.fourier_embedding import FourierEmbedding
from utils import angle_between_2d_vectors
from utils import weight_init
from utils import wrap_angle

class QCNetAgentEncoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_agent_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 **kwargs) -> None:
        super(QCNetAgentEncoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_agent_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        if dataset == 'ETRI_Dataset':
            input_dim_x_a = 4  # agent dim
            input_dim_r_t = 4  # time dim
            input_dim_r_pl2a = 3  # polygon to agent dim (3 features)
            input_dim_r_a2a = 3  # agent_to_agent
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))

        if dataset == 'ETRI_Dataset':
            self.type_a_emb = nn.Embedding(10, hidden_dim)
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))
        self.x_a_emb = FourierEmbedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_t_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2a_emb = FourierEmbedding(input_dim=input_dim_r_pl2a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r_a2a, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.t_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(self.num_layers)]
        )
        self.pl2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(self.num_layers)]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(self.num_layers)]
        )
        self.apply(weight_init)

    def forward(self, map_enc_x_pl: torch.Tensor,
               mask: torch.Tensor,
               pos_a: torch.Tensor,
               motion_vector_a: torch.Tensor,
               head_a: torch.Tensor,
               pos_pl: torch.Tensor,
               orient_pl: torch.Tensor,
               vel: torch.Tensor,
               agent_type: torch.Tensor,
               batch_s: torch.Tensor,
               batch_pl: torch.Tensor,
               magnitude_info: torch.Tensor = None):

        categorical_embs = [self.type_a_emb(agent_type)
                            .repeat_interleave(repeats=self.num_historical_steps,
                                               dim=0), ]  # (num_agent x obs_len) x 128

        motion_vector_a = motion_vector_a
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

        if self.dataset == 'ETRI_Dataset':
            x_a = torch.stack(
                [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]),
                 torch.norm(vel[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel[:, :, :2])], dim=-1)
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        x_a = x_a.view(-1, x_a.size(-1)).float()
        x_a = self.x_a_emb(continuous_inputs=x_a, categorical_embs=categorical_embs)
        x_a = x_a.view(-1, self.num_historical_steps, self.hidden_dim)
        # ===================================================
        # Temporal attention 준비 
        pos_t = pos_a.reshape(-1, self.input_dim)
        head_t = head_a.reshape(-1)
        head_vector_t = head_vector_a.reshape(-1, 2)
        # 위 시간 축을 기준으로 모든 데이터를 flatten (temporal attention하기 위해서)
        mask_t = mask.unsqueeze(2) & mask.unsqueeze(1) # [11]
        edge_index_t = dense_to_sparse(mask_t)[0] # [12] 
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]] # [13]
        edge_index_t = edge_index_t[:, edge_index_t[1] - edge_index_t[0] <= self.time_span] # [14]

        # [11]valid한 시간 step들 간의 연결 마스크 생성 
        # [12] dense mask -> sparse edge index 변환
        # [13]시간 축의 미래로만 attention이 가도록 (causal attention) (과거 -> 미래) (인과관계 보장)
        # [14] time span 제한 (너무 먼 미래로 attention이 가지 않도록) -> Temporal attention에서 과거 정보만 사용하고 너무 먼 과거는 제외
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
             rel_head_t,
             edge_index_t[0] - edge_index_t[1]], dim=-1)
        r_t = r_t.float()
        # Temporal relation을 Fourier embedding으로 고차원 벡터로 변환
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)

        # ===================================================
        # Polygon to agent attention 준비 (map-agent attention)
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        mask_s = mask.transpose(0, 1).reshape(-1)

        # map polygon도 시간 축에 맞춰 복제 (각 time step에 동일한 map 정보 사용)
        pos_pl = pos_pl.repeat(self.num_historical_steps, 1) 
        orient_pl = orient_pl.repeat(self.num_historical_steps)

        # polygon to agent attention 준비
        pos_s = pos_s.double()
        pos_pl = pos_pl.double()
        edge_index_pl2a = radius(x=pos_s[:, :2], y=pos_pl[:, :2], r=self.pl2a_radius, batch_x=batch_s, batch_y=batch_pl,
                                 max_num_neighbors=300)
        edge_index_pl2a = edge_index_pl2a[:, mask_s[edge_index_pl2a[1]]]
        # polygon - agent relation encoding (거리, 방향, orientation 차이) 
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])
        ######################### add new feature for magnitude #########################
        # Magnitude 정보 추가 - 실제 magnitude 값 사용
        if magnitude_info is not None and magnitude_info.numel() > 0 and edge_index_pl2a.size(1) > 0:
            # polygon들과 연결된 point들의 magnitude 정보 사용
            # edge_index_pl2a[0]은 polygon 인덱스이므로, 해당 polygon과 연결된 point들의 magnitude 사용
            # 실제로는 polygon-point 매핑이 필요하지만, 여기서는 근사적으로 처리
            avg_magnitude = magnitude_info.mean()
            polygon_magnitudes = torch.ones(edge_index_pl2a.size(1), device=rel_pos_pl2a.device) * avg_magnitude
            # 안전하게 max 계산 (텐서가 비어있지 않을 때만)
            max_magnitude = polygon_magnitudes.max() if polygon_magnitudes.numel() > 0 else 1.0
            magnitude_normalized = polygon_magnitudes / (max_magnitude + 1e-6)
        else:
            # magnitude 정보가 없거나 edge가 없을 때는 기본값 사용
            polygon_magnitudes = torch.zeros(edge_index_pl2a.size(1), device=rel_pos_pl2a.device)
            magnitude_normalized = torch.zeros(edge_index_pl2a.size(1), device=rel_pos_pl2a.device)
        ################################################################################
        r_pl2a = torch.stack(
            [torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]),
             rel_orient_pl2a + 0.5*(polygon_magnitudes + magnitude_normalized), # Sum features to maintain dimension
            ], dim=-1)
        r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a.float(), categorical_embs=None)

        # ===================================================
        # Agent to agent attention 준비 (agent-agent attention)
        edge_index_a2a = radius_graph(x=pos_s[:, :2], r=self.a2a_radius, batch=batch_s, loop=False,
                                      max_num_neighbors=300)
        edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0]
        # agent들 간의 spatial neighborhood graph 생성 
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
             rel_head_a2a], dim=-1)
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a.float(), categorical_embs=None)

        # Multi-layer attention부분 
        for i in range(self.num_layers):
            # 1) Temporal attention
            x_a = x_a.reshape(-1, self.hidden_dim)
            x_a = self.t_attn_layers[i](x_a, r_t, edge_index_t)
            # 2) Polygon to agent attention
            x_a = x_a.reshape(-1, self.num_historical_steps,
                              self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            x_a = self.pl2a_attn_layers[i]((map_enc_x_pl.transpose(0, 1).reshape(-1, self.hidden_dim), x_a), r_pl2a,
                                           edge_index_pl2a)
            # 3) Agent to agent attention
            x_a = self.a2a_attn_layers[i](x_a, r_a2a, edge_index_a2a)
            x_a = x_a.reshape(self.num_historical_steps, -1, self.hidden_dim).transpose(0, 1)

        return {'x_a': x_a}
