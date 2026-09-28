import torch


class L1:

    def __call__(self, batch_rep):
        return torch.sum(torch.abs(batch_rep), dim=-1).mean()

class L0:
    """non-differentiable
    """

    def __call__(self, batch_rep):
        return torch.count_nonzero(batch_rep, dim=-1).float().mean()

def selu(batch_rep:torch.Tensor)->torch.Tensor:
    return 0.01 * torch.nn.functional.softplus(100 * batch_rep)

class FLOPS:
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __call__(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)

class Scaled_FLOPS_with_mask:
    
    def __call__(self, batch_rep, thresh_len=74,scaled_vec=None):
        non_zeros = torch.count_nonzero(batch_rep,dim=-1)
        batch_rep = batch_rep * (non_zeros > thresh_len).unsqueeze(-1)
        #batch_rep = mask.unsqueeze(-1) * batch_rep
        if scaled_vec is None:
            return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)
        else:
            ret = torch.mean(torch.abs(batch_rep), dim=0) ** 2
            return torch.sum(ret/scaled_vec)
        
class Treshold_reg:

    def __call__(self, threshold_value, mask=None):
        '''
        type = false for query, type = true for document
        '''
        if threshold_value.size()[-1] > 1:
            # if mask is not None:
            # mask = (threshold_value > 0)
            threshold_value = torch.mean(threshold_value).squeeze()
            #return torch.sum(torch.log(1+torch.exp(threshold_value)))
            #return torch.log(1+torch.exp(-torch.mean(5*torch.sum(threshold_value / (1+torch.sum(mask,dim=-1,keepdim=True)),dim=-1))))# mean over batch
                #torch.sum(torch.log(1+torch.exp(-self.d_thres * 5)))
        return torch.sum(torch.log(1+torch.exp(-threshold_value*5)))

class RegWeightScheduler:
    """same scheduling as in: Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __init__(self, lambda_, T):
        self.lambda_ = lambda_
        self.T = T
        self.t = 0
        self.lambda_t = 0

    def step(self):
        """quadratic increase until time T
        """
        if self.t >= self.T:
            pass
        else:
            self.t += 1
            self.lambda_t = self.lambda_ * (self.t / self.T) ** 2
        return self.lambda_t

    def get_lambda(self):
        return self.lambda_t
    
    def get_lambda_t(self):
        return self.lambda_t * (self.t / self.T) ** 2


def init_regularizer(reg, **kwargs):
    if reg == "L0":
        return L0()
    elif reg == "L1":
        return L1()
    elif reg == "FLOPS":
        return FLOPS()
    else:
        raise NotImplementedError("provide valid regularizer")