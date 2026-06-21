
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from utils import weight_init
class AttentionLayer(MessagePassing):

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 bipartite: bool,
                 has_pos_emb: bool,
                 **kwargs) -> None:
        super(AttentionLayer, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.has_pos_emb = has_pos_emb
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_k = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
        self.to_v = nn.Linear(hidden_dim, head_dim * num_heads)
        if has_pos_emb:
            self.to_k_r = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
            self.to_v_r = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_s = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_g = nn.Linear(head_dim * num_heads + hidden_dim, head_dim * num_heads)
        self.to_out = nn.Linear(head_dim * num_heads, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.ff_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(inplace=True), # default negative_slope = 0.01
            # nn.GELU(),  # GELU doesn't support inplace parameter
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        if bipartite:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = nn.LayerNorm(hidden_dim)
        else:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = self.attn_prenorm_x_src
        if has_pos_emb:
            self.attn_prenorm_r = nn.LayerNorm(hidden_dim)
        self.attn_postnorm = nn.LayerNorm(hidden_dim)
        self.ff_prenorm = nn.LayerNorm(hidden_dim)
        self.ff_postnorm = nn.LayerNorm(hidden_dim)
        self.apply(weight_init)

    def forward(self,
                x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
                r: Optional[torch.Tensor], 
                edge_index: torch.Tensor) -> torch.Tensor:
                # x: (x_src, x_dst) 또는 단일 tensor
                # r: positional embedding (r_t2m, r_pl2m 등)
                # edge_index: 연결 정보 (edge_index_t2m, edge_index_pl2m 등)
        
        #self-attention인지 cross-attention인지 구분
        if isinstance(x, torch.Tensor):
            x_src = x_dst = self.attn_prenorm_x_src(x) # self-attention
        else: 
            x_src, x_dst = x # cross-attention
            x_src = self.attn_prenorm_x_src(x_src) # x_t (trajectory) → Key, Value [x_src = x_t]
            x_dst = self.attn_prenorm_x_dst(x_dst) # m (mode query) → Query [x_dst = m]
            x = x[1]
        if self.has_pos_emb and r is not None:
            r = self.attn_prenorm_r(r)
        x = x + self.attn_postnorm(self._attn_block(x_src, x_dst, r, edge_index))
        x = x + self.ff_postnorm(self._ff_block(self.ff_prenorm(x)))
        return x

    #sparse softmax
    #전체 N×N이 아니라 실제 연결된 E개만 계산
    def message(self,
                q_i: torch.Tensor,
                k_j: torch.Tensor,
                v_j: torch.Tensor,
                r: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: Optional[torch.Tensor]) -> torch.Tensor:
        if self.has_pos_emb and r is not None:
            #🎯 여기가 핵심! positional embedding을 Key와 Value에 추가
            #기본 key: 궤적의 feature정보 +positional Key: 거리, 각도, 시간 관계 정보 
            #최종 Key: 내용 + 위치 정보가 합쳐진 enhanced representation
            k_j = k_j + self.to_k_r(r).view(-1, self.num_heads, self.head_dim)
            v_j = v_j + self.to_v_r(r).view(-1, self.num_heads, self.head_dim)
        sim = (q_i * k_j).sum(dim=-1) * self.scale # similarity 계산
        attn = softmax(sim, index, ptr) #sparse softmax!!!(= edge_index로 연결된 것들만 softmax 적용)
        attn = self.attn_drop(attn)
        return v_j * attn.unsqueeze(-1)

    #Update 함수 (Gating)
    def update(self,
               inputs: torch.Tensor,
               x_dst: torch.Tensor) -> torch.Tensor:

        inputs = inputs.view(-1, self.num_heads * self.head_dim) # attention으로 얻은 새로운 정보 
        g = torch.sigmoid(self.to_g(torch.cat([inputs, x_dst], dim=-1))) #gate 0~1 얼마나 새 정보를 받을 지 
        return inputs + g * (self.to_s(x_dst) - inputs)

    def _attn_block(self,
                    x_src: torch.Tensor,
                    x_dst: torch.Tensor,
                    r: Optional[torch.Tensor],
                    edge_index: torch.Tensor) -> torch.Tensor:
        q = self.to_q(x_dst).view(-1, self.num_heads, self.head_dim) #Query
        k = self.to_k(x_src).view(-1, self.num_heads, self.head_dim) #Key
        v = self.to_v(x_src).view(-1, self.num_heads, self.head_dim) #Value
        agg = self.propagate(edge_index=edge_index, x_dst=x_dst, q=q, k=k, v=v, r=r)
        return self.to_out(agg)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff_mlp(x)
