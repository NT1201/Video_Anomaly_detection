import torch.nn as nn
import torch.nn.functional as F
import torch


class Learner(nn.Module):
    def __init__(self, input_dim, drop_p=0.30, use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        self.cls_mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(drop_p),
            nn.Linear(512, 1),        # raw logit
        )

        if use_attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

    # --------------------------------------------------------------
    def forward(self, x, *, return_attention=False):
        if not self.use_attention:
            return self.cls_mlp(x)

        att = F.softmax(self.att_mlp(x) / 0.5, dim=0)
        log = self.cls_mlp(x)
        if return_attention:
            return log, att
        return log
