
from typing import List, Union

import torch
import torch.nn as nn

from losses.gaussian_nll_loss import GaussianNLLLoss
from losses.laplace_nll_loss import LaplaceNLLLoss
from losses.von_mises_nll_loss import VonMisesNLLLoss

#== 확률의 음의 로그값(NLL)을 계산하는 클래스==
class NLLLoss(nn.Module):

    def __init__(self,
                 component_distribution: Union[str, List[str]],eps: float = 1e-6,reduction: str = 'mean') -> None:
        super(NLLLoss, self).__init__()
        self.reduction = reduction

        loss_dict = {
            'gaussian': GaussianNLLLoss,
            'laplace': LaplaceNLLLoss, #이거쓰임1
            'von_mises': VonMisesNLLLoss, #이거쓰임2
        }
        if isinstance(component_distribution, str):
            self.nll_loss = loss_dict[component_distribution](eps=eps, reduction='none')
        else:
            self.nll_loss = nn.ModuleList([loss_dict[dist](eps=eps, reduction='none')for dist in component_distribution])

    def forward(self,pred: torch.Tensor,target: torch.Tensor) -> torch.Tensor:
        if isinstance(self.nll_loss, nn.ModuleList):
            nll = torch.cat(
                [self.nll_loss[i](pred=pred[..., [i, target.size(-1) + i]], target=target[..., [i]])for i in range(target.size(-1))], dim=-1)
        else:
            nll = self.nll_loss(pred=pred, target=target)
        
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

# 예시 데이터
#pred = torch.tensor([[[5.0, 1.0]]])     # x_mean=5.0, x_scale=1.0
#target = torch.tensor([[[6.0]]])         # 실제 x좌표=6.0

# LaplaceNLLLoss 계산
#loc, scale = pred.chunk(2, dim=-1)       # loc=5.0, scale=1.0
#nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
#     = log(2 * 1.0) + |6.0 - 5.0| / 1.0
#     = log(2) + 1.0
#     = 0.693 + 1.0 = 1.693