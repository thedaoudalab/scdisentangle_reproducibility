import wandb 
import json 
import numpy as np
import hygeia.models.mlp_parts as hmlp

def log_important_notes(important_notes):   
    # Push config parameters to wandb

    html_notes = f"<pre>{important_notes}</pre>"

    # Log the notes using wandb.log with the HTML object
    wandb.log({"important_notes": wandb.Html(html_notes)})

def is_apply(func):
    def wrapper(*args, **kwargs):
        apply = kwargs.get('apply', True)

        if apply:
            result = func(*args, **kwargs)
            return result
        else:
            return 0

    return wrapper

def init_wandb(hparams):
    '''
    Initialize wandb

    Parameters:
    ------------
    hparams: Dict
        Dictionary fo the config file

    '''
    if hparams['wandb']['wandb_log']:
        print('Initializing wandb')
        wandb.init(
            name=hparams['wandb']['name'], 
            group=hparams['wandb']['group'],
            project=hparams['wandb']['project']
            )
        filtered_params = json.dumps(hparams, indent = 4)
        log_important_notes(filtered_params)

    else:
        print('Wandb is off')

def return_dict_value(value, params_dict):
    '''Check whether a key exists, return its value else return False'''

    if value in params_dict.keys():
        return params_dict[value]
    else:
        return False

def load_models(hparams, device):
    '''
    Loop through a dict and load models into the models dictionary

    Parameters:
    ------------
    hparams: Dict
        Dictionary:
            keys: model_name
            values: parameters in hygeia convention
        device: torch.device

    Returns
    ------------
    models: Dict
        Dictionary:
            keys: model_name
            values: PyTorch Module
    '''
    models = {}
    for model_name, model_dict in hparams['models'].items():
        xavier_init = return_dict_value('xavier_init', model_dict)
        batch_norm = return_dict_value('batch_norm', model_dict)
        dropout_rate = return_dict_value('dropout_rate', model_dict)
        orthogonal = return_dict_value('orthogonal', model_dict)
        input_activation = return_dict_value('input_activation', model_dict)
        input_dropout = return_dict_value('input_dropout', model_dict)

        if 'layer_type' in model_dict.keys():
            layer_type = model_dict['layer_type']
        else:
            layer_type = 'linear'

        if 'mapping_operation' in model_dict.keys():
            mapping_operation = model_dict['mapping_operation']
        else:
            mapping_operation = 'map'

        models[model_name] = hmlp.MLP(
            module_list = model_dict['layers'],
            last_activation = model_dict['last_activation'],
            hidden_activation = model_dict['hidden_activation'],
            input_activation = input_activation,
            xavier_init = xavier_init,
            batch_norm = batch_norm, 
            dropout_rate = dropout_rate,
            name = model_name,
            layer_type = layer_type,
            mapping_operation = mapping_operation,
            orthogonal = orthogonal,
            input_dropout = input_dropout,
            ).to(device)

    if 'embedders' in hparams.keys():
        for embedder_name, embedder_dict in hparams['embedders'].items():
            models[embedder_name] = hmlp.ConditionEmbedding(
                nb_classes = embedder_dict['n_classes'],
                embedding_dim = embedder_dict['n_dim']
                ).to(device)

    return models