import torch
import torch.nn as nn
import torch.nn.functional as F

class Sine(nn.Module):
    """Return Sine function that can be used inside sequential modules"""
    def forward(self, x_inp):
        return torch.sin(x_inp)

class NormalizedSine(nn.Module):
    """Return normalized Sine function that can be used inside sequential modules"""
    def forward(self, x_inp):
        return (1 + torch.sin(x_inp)) /2

class L2(nn.Module):
    def forward(self, features):
        return torch.div(features.T,torch.norm(features, dim=1)).T

class BatchNormalize(nn.Module):
    def forward(self, features):
        return (features - features.mean(dim=0)) / features.std(dim=0)

class BatchCenter(nn.Module):
    def forward(self, features):
        return (features - features.mean(dim=0))

class SineReLU(nn.Module):
    def forward(self, x_inp):
        return torch.relu(torch.sin(x_inp))