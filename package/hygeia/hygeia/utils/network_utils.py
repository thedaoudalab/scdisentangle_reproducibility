import os 
import time 
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import random
import numpy as np
from torch.autograd import Function
from hygeia.utils import activations as ha
import torch.nn.init as init

class ParameterLayer(nn.Module):
    def __init__(self, in_neurons, out_neurons, orthogonal = False):
        super(ParameterLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_neurons, out_neurons))
        self.orthogonal = orthogonal

        init.xavier_uniform_(self.weight)
        
    def forward(self, x):
        if self.orthogonal:
            weight = (self.weight - self.weight.t()) / 2
            
            I = torch.eye(self.weight.shape[0]).to(x.device)

            O = torch.inverse(I + weight) @ (I - weight) 

            return torch.matmul(x, O) #+ self.bias
        else:
            return torch.matmul(x, self.weight)

    def __repr__(self):
        return f'CustomModule(in_neurons={self.weight.shape[1]}, out_neurons={self.weight.shape[0]})'

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

def set_seed(seed_value=42, use_cuda=True):
    """Set seed for reproducibility."""
    
    # Python built-in random module
    random.seed(seed_value)
    
    # Numpy
    np.random.seed(seed_value)
    
    # Torch
    torch.manual_seed(seed_value)
    
    # if you are using GPU
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _init_params(model_obj):
    """ Asymmetric params initialization """
    for m in model_obj.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
    return

def _make_layers(
    layers, 
    dropout_rate=None, 
    last_activation=False,
    batch_norm=True, 
    hidden_activation=nn.ReLU(), 
    use_skip_connections=False, 
    latent_size=None,
    input_activation=False,
    layer_type='linear', # or parameter
    orthogonal=False,
    input_dropout=None,
    ):
   
    if isinstance(layers, torch.nn.modules.container.Sequential):
        return layers

    nb_layers = len(layers)

    assert nb_layers > 1, 'Number of layers should be at least 2 (input, output)'

    net_layers = []
    linear_layers = _linear_from_list(layers, use_skip_connections, latent_size)

    if input_dropout:
        net_layers.append(nn.Dropout(p=input_dropout))

    if input_activation:
        net_layers.append(input_activation)
    for idx, (in_neurons, out_neurons) in enumerate(linear_layers):
        if layer_type == 'linear':
            net_layers.append(nn.Linear(in_neurons, out_neurons))
        elif layer_type == 'parameter':
            print('ORG', orthogonal)
            net_layers.append(ParameterLayer(in_neurons, out_neurons, orthogonal))

        else:
            raise ValueError('layer_type should be in ["linear", "parameter"]')

        if idx != len(linear_layers) - 1:
            if batch_norm:
                net_layers.append(nn.BatchNorm1d(out_neurons))
            if dropout_rate:
                net_layers.append(nn.Dropout(p=dropout_rate))
            net_layers.append(hidden_activation)
        else:
            if last_activation:
                #net_layers.append(hidden_activation)
                net_layers.append(last_activation)
    
    return nn.Sequential(*net_layers)

def _linear_from_list(layers, use_skip_connections=False, latent_size=None):
   
    assert isinstance(layers, list), 'Provide a list with nb of neurons/layer'
    nb_layers = len(layers)
    linear_layers = []

    for idx in range(nb_layers):
        if idx != nb_layers - 1:
            in_neurons = layers[idx]
            out_neurons = layers[idx + 1]

            if use_skip_connections and idx != 0 and latent_size is not None:
                in_neurons += latent_size

            linear_layers.append((in_neurons, out_neurons))

    return linear_layers

def activate_output(x, activation_fn):
    """ Applies an activation function """
    if activation_fn:
        return activation_fn(x)
    else:
        return x

def get_n_params(model):
    """ Return number of parameters of a pytorch model """
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def batch_normalize(features):
    """ Normalize across the batch stats """
    #TODO: Remove | Moved to activations
    return (features - features.mean(dim=0)) / features.std(dim=0)

def l2_normalize(features):
    """ L2 normalize cells """
    #TODO: Remove | Moved to activations
    return torch.div(features.T,torch.norm(features, dim=1)).T

def center(features):
    return (features - features.mean(dim=0))
    
def str_to_nn_activation(activation_fn_str):
    """
    Convert string to torch activation function
    """
    if activation_fn_str == "identity":
        return None

    modules = [ha, nn]
 
    for mod in modules:
        try :
            return getattr(mod, activation_fn_str)()
        except:
            pass

    raise ValueError(f"unknown activation {activation_fn_str}")

