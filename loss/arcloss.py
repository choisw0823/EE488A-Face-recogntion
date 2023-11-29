import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFunction(nn.Module):
    def __init__(self, s=64.0, m=0.50, easy_margin=False, **kwargs):
        super(LossFunction, self).__init__()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        # Convert m to a tensor before applying torch.cos and torch.sin
        m_tensor = torch.tensor(m, dtype=torch.float32)

        self.cos_m = torch.cos(m_tensor)
        self.sin_m = torch.sin(m_tensor)
        self.th = torch.cos(torch.pi - m_tensor)
        self.mm = torch.sin(torch.pi - m_tensor) * m_tensor

    def forward(self, cosine, label):
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return F.cross_entropy(output, label)