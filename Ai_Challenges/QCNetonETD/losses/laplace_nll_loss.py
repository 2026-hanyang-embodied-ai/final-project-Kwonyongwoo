
import torch
import torch.nn as nn


class LaplaceNLLLoss(nn.Module):

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,pred: torch.Tensor,target: torch.Tensor) -> torch.Tensor:
        #pred : [batch, time, 2] - [mean, scale]
        #target : [batch, time, 2] - [x, y]
        loc, scale = pred.chunk(2, dim=-1) #mean=loc, scale=퍼짐정도(불확실성) 분리 
        scale = scale.clone() 

        with torch.no_grad():
            scale.clamp_(min=self.eps)
        #Laplce 분포의 NLL공식 
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

        # Laplace 분포: f(x) = (1/2b) * exp(-|x-μ|/b)
        # NLL = -log(f(x)) = log(2b) + |x-μ|/b
