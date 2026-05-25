import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from hygeia import utils as hu
import hygeia.utils.network_utils as hnn

from icecream import ic


class MLP(nn.Module):
    def __init__(
        self,
        module_list,
        dropout_rate = None,
        last_activation = False,
        batch_norm = False,
        hidden_activation = nn.ReLU(),
        input_activation = False,
        name = None,
        xavier_init = True,
        layer_type = 'linear',
        mapping_operation = 'map', # bias scale,
        use_skip_connections =False,
        input_dropout=None,
        orthogonal=False,
        **kwargs
    ):
        """
        Parameters
        ------------
        module_list: List/nn.Sequential
            List with number of neurons per layer
            Or a custom nn.Sequential
        
        dropout_rate: Float
            % neurons to dropout, default=None
        
        batch_norm: Bool
            Whether to perform batch norm
        
        last_activation: Pytorch activation function
            Activation function to apply to output
            Default=False (Identity)

        hidden_activation: Pytorch activation function
            Activation function to use between layers
        
        """
        super(MLP, self).__init__()

        self.input_activation = input_activation
        self.last_activation = last_activation
        self.hidden_activation = hidden_activation
        if isinstance(last_activation, str):
            self.last_activation = hnn.str_to_nn_activation(self.last_activation)

        if isinstance(hidden_activation, str):
            self.hidden_activation = hnn.str_to_nn_activation(self.hidden_activation)

        if isinstance(input_activation, str):
            self.input_activation = hnn.str_to_nn_activation(self.input_activation)

        self.mapping_operation = mapping_operation

        self.name = name
        self.sequence = hu._make_layers(
            layers=module_list,
            dropout_rate=dropout_rate,
            input_activation=self.input_activation,
            last_activation=self.last_activation,
            batch_norm=batch_norm,
            hidden_activation=self.hidden_activation,
            use_skip_connections=use_skip_connections,
            latent_size = module_list[0],
            layer_type=layer_type,
            orthogonal=orthogonal,
            input_dropout=input_dropout,
        )

               
        if xavier_init:
            self.initialize_weights()

    def forward(self, x_inp, **kwargs):
       
        for i, layer in enumerate(self.sequence):
            if i == 0:
                out = layer(x_inp)
            else:
                out = layer(out)

        if self.mapping_operation == 'map':
            return out
        elif self.mapping_operation == 'bias':
            return x_inp + out
        elif self.mapping_operation == 'scale':
            return x_inp @ out

        else:
            raise ValueError('Mapping operation should be in ["map", "bias", "scale"]')

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
                    
class ConditionEmbedding(nn.Module):
    def __init__(
        self, 
        nb_classes, 
        embedding_dim, 
        post_processer=None, 
        **kw_pytorch
        ):
        """
        Class(Labels) embedding layer
        
        Parameters:
        ------------
        nb_classes: Int 
            Number of classes
        
        embedding_dim: Int
            Dimension of class/condition embedding
        
        post_processer: Pytorch Module
            Transformation/Mapping to apply to condition embedding
            Default=None (embedding is returned as it is)
        """
        super(ConditionEmbedding, self).__init__()
        self.encoder = nn.Embedding(
            nb_classes, embedding_dim, 
            **kw_pytorch)
        self.post_processer = post_processer

    def forward(self, x_inp):
        out = self.encoder(x_inp)
        if self.post_processer:
            out = self.post_processer(out)

        return out