import scanpy as sc 
import anndata as ad
import numpy as np
import torch
from scipy.sparse import csr_matrix
from tqdm import tqdm

counterfactual_dict = {
    'condition': 'control'
}

condition_name = 'condition'

def downsample_balance_by_cell_type(
    adata, 
    key='cell_type', 
    seed=0
    ):
    """
    downsample each cell type to min cell type n_cells
    """

    ct = adata.obs[key]
    n_min = ct.value_counts().min()
    rng = np.random.default_rng(seed)

    groups = sorted(ct.value_counts().index.tolist(), key=lambda x: str(x))
    chosen = []
    for g in groups:
        idx = np.flatnonzero(ct == g)
        chosen.append(rng.choice(idx, size=n_min, replace=False))

    chosen = np.concatenate(chosen)
    chosen.sort()  # keep original order
    
    return adata[chosen].copy()

def get_progressive_latent(
    trainer,
    adata,
    counterfactual_dict,
    get_recs=False,
    balance_clusters=False,
    covariate_name=None,
    ):
    
    adata = adata.copy()

    if balance_clusters:
        adata = downsample_balance_by_cell_type(
            adata=adata,
            key=covariate_name,
            )

    # Get X
    _X = adata.layers['org_expression']
    if not isinstance(_X, np.ndarray):
        _X = _X.toarray()

    # Library size
    library = torch.log(
        torch.tensor(_X).sum(1).unsqueeze(1)
        ).to(trainer.device).float()

    # To sparse:
    adata.X = csr_matrix(adata.X)

    # Disentangled latent
    dis_latent_stack = adata.obsm['dis_latent_stack'].copy()

    # Disentangled latent dimensionality
    n_latent_size = dis_latent_stack.shape[1]

    # Create variables dictionary
    label_keys = trainer.hparams['data']['label_keys']
    covariates_batch = {}
    for label_key in label_keys:
        str_labels = adata.obs[label_key].tolist()
        code_labels = [trainer.dataset.reverse_label_mapping[label_key][v] for v in str_labels]
        code_labels = torch.tensor(code_labels).long().to(trainer.device)
        covariates_batch[label_key] = code_labels

    # Generate latent levels
    adata_levels_cat = []
    for idx in tqdm( range( n_latent_size) ):

        _adata = sc.AnnData(
            adata.X.copy(), 
            obs=adata.obs.copy()
            )

        collapsed_latent = dis_latent_stack.copy()

        # Collapse latent dimensiosn to their mean over cells
        if idx != (n_latent_size -1):
            collapsed_latent[:, idx+1:] = collapsed_latent[:, idx+1:].mean(axis=0)

        collapsed_latent_list = [
            torch.tensor(
                collapsed_latent[:, x]
                ).unsqueeze(1).to(trainer.device).float() for x in range(n_latent_size)
            ]
        
        # Get disentangled latent for level idx
        counterfactual_latent = trainer.get_counterfactuals(
            x_inp=torch.tensor(dis_latent_stack).to(trainer.device).float(),
            variables=covariates_batch,
            dis_latent=collapsed_latent_list,
            counterfactual_dict=counterfactual_dict,
            suffixe=''
            )
        
        if get_recs:
            
            # Get reconstruction (counterfactuals) for level idx
            counterfactual_recs = trainer.get_recs(
                decoder_name='decoder',
                decoder_input=counterfactual_latent['map_latent_summed'],
                px_name='px_r',
                library=library,
                suffixe=''
                )

            # Set to X
            _adata.X = counterfactual_recs['reconstructed'].cpu().detach().numpy()

        # Add the disentangled spaces to obsm
        _adata.obsm['mapped_latent'] = counterfactual_latent['map_latent_summed'].cpu().detach().numpy()
        
        # Add level index (1 to 16)
        _adata.obs['level'] = idx+1

        # Append the _adata of level idx+1
        adata_levels_cat.append(_adata)

    # Concatenate adatas
    adata_levels_cat = ad.concat(adata_levels_cat)

    return adata_levels_cat
