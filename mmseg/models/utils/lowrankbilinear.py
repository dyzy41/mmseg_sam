import torch
import torch.nn as nn


class LowRankBilinearAttention(nn.Module):
    def __init__(self, d_q=256, d_k=512, channel=256, d_v=512):
        super(LowRankBilinearAttention, self).__init__()
        self.WQ = nn.Linear(d_q, channel)
        self.WK = nn.Linear(d_k, channel)
        self.WV = nn.Linear(d_v, channel)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K):
        K = K.unsqueeze(1)
        V = K.clone()
        q = self.WQ(Q)
        k = self.WK(K)
        v = self.WV(V)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn_probs = self.softmax(attn_scores)
        output = torch.matmul(attn_probs, v)
        return output