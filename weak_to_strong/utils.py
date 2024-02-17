# From https://github.com/namisan/mt-dnn
import torch 
import torch.nn.functional as F
def KL(input, target, reduction="sum"):
    input = input.float()
    target = target.float()
    loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target, dim=-1, dtype=torch.float32), reduction=reduction)
    return loss

def SKL(logit, target, epsilon=1e-8):
    logit = logit.view(-1, logit.size(-1)).float()
    target = target.view(-1, target.size(-1)).float()
    #bs = logit.size(0)
    p = F.log_softmax(logit, 1).exp()
    y = F.log_softmax(target, 1).exp()
    rp = -(1.0/(p + epsilon) -1 + epsilon).detach().log()
    ry = -(1.0/(y + epsilon) -1 + epsilon).detach().log()
    return (p* (rp- ry) * 2).sum()


def adv_project(grad, norm_type='inf', eps=1e-6):
    if norm_type == 'l2':
        direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
    elif norm_type == 'l1':
        direction = grad.sign()
    else:
        direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
    return direction

@dataclass 
class adv_config:
    noise_var: float = 1e-5
    adv_step_size: float = 1e-3
    noise_gamma: float = 1e-6
    project_norm_type: str = 'inf'
    adv_alpha: float = 1.
