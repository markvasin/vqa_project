import torch.nn as nn
import torch.nn.functional as F


class StackedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StackedAttention, self).__init__()
        self.Wv = nn.Conv2d(input_dim, hidden_dim, kernel_size=1, padding=0)
        self.Wu = nn.Linear(input_dim, hidden_dim)
        self.Wp = nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0)
        self.hidden_dim = hidden_dim
        self.attention_maps = None

    def forward(self, v, u):
        """
        Input:
        - v: N x D x H x W
        - u: N x D

        Returns:
        - next_u: N x D
        """
        N, K = v.size(0), self.hidden_dim
        D, H, W = v.size(1), v.size(2), v.size(3)
        v_proj = self.Wv(v)  # N x K x H x W
        u_proj = self.Wu(u)  # N x K
        u_proj_expand = u_proj.view(N, K, 1, 1).expand(N, K, H, W)
        h = F.tanh(v_proj + u_proj_expand)
        p = F.softmax(self.Wp(h).view(N, H * W)).view(N, 1, H, W)
        self.attention_maps = p.data.clone()

        v_tilde = (p.expand_as(v) * v).sum(2).sum(3).view(N, D)
        next_u = u + v_tilde
        return next_u
