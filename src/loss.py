import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

eps = 1e-7

class Similarity(nn.Module):

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y, m=0.45, do_ifm=False, ifm_type='p-n+', ifm_logit_type='constant'):
        N = x.shape[0]
        device = x.device
        logits = self.cos(x, y) 
        if do_ifm:
            mask = torch.eye(N).to(device)

            if ifm_logit_type == 'constant':
                epsilon = m
            elif ifm_logit_type == 'dynamic':
                epsilon = logits[mask.bool()].detach() / (N - 1)
            elif ifm_logit_type == 'softmax':
                epsilon = logits[mask.bool()].detach() * F.softmax(logits, dim=-1)
            
            if 'p-' in ifm_type:
                logits -= epsilon * mask
            elif 'p+' in ifm_type:
                logits += epsilon * mask
            
            if 'n-' in ifm_type:
                logits -= epsilon * (1 - mask)
            elif 'n+' in ifm_type:
                logits += epsilon * (1 - mask)
    
        return logits / self.temp

def ifm_loss(x1, x2, x3=None, temp=0.05, m=10, hard_negative_weight=0., do_ifm=False, ifm_type='p-n+', ifm_logit_type='constant'):
    device = x1.device    
    loss_fct = nn.CrossEntropyLoss() 
    sim = Similarity(temp=temp)
    
    cos = sim(x1.unsqueeze(1), x2.unsqueeze(0), m=m, do_ifm=do_ifm, ifm_type=ifm_type, ifm_logit_type=ifm_logit_type)
    # Hard negative
    if x3 is not None:
        x1_x3_cos = sim(x1.unsqueeze(1), x3.unsqueeze(0), m=0, do_ifm=False) 
        cos = torch.cat([cos, x1_x3_cos], 1)

    labels = torch.arange(cos.size(0)).long().to(device)

    if x3 is not None:
        # Note that weights are actually logits of weights
        z3_weight = hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos.size(-1) - x1_x3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (x1_x3_cos.size(-1) - i - 1) for i in range(x1_x3_cos.size(-1))]
        ).to(device)
        cos = cos + weights

    loss = loss_fct(cos, labels)

    return loss, cos