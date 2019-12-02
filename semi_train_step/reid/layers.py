import torch
import torch.nn as nn
import torch.nn.functional as F


class Cosine(nn.Module):
    def __init__(self, in_features, out_features, scale=14):
        super(Cosine, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.weights = torch.nn.Parameter(torch.randn(in_features, out_features), requires_grad=True)
        nn.init.xavier_normal_(self.weights, gain=1)

    def forward(self, x):
        cosine = torch.mm(F.normalize(x), F.normalize(self.weights, dim=0))
        return self.scale * cosine

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', scale=' + str(self.scale) + ')'\


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super(AdaptiveConcatPool2d, self).__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class Flatten(nn.Module):
    "Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"
    def __init__(self, full=False):
        super(Flatten, self).__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)
