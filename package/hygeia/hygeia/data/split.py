import numpy as np 
import scanpy as sc
from icecream import ic

def split_anndata(
    anndata, 
    test_size, 
    val_size=0, 
    seed=42
    ):

    anndata.obs['split_anndata'] = 'HygeiaNA' 

    if not 0 <= test_size < 1 or not 0 <= val_size < 1:
        raise ValueError("test_size and val_size must be between 0 and 1")
    if test_size + val_size >= 1:
        raise ValueError("The sum of test_size and val_size must be less than 1")
    
    # Set the seed for reproducibility
    np.random.seed(seed)
    
    # Generate a random permutation of all indices
    n_obs = anndata.n_obs
    shuffled_indices = np.random.permutation(n_obs)
    
    # Calculate the number of observations for each set
    n_test = int(np.floor(test_size * n_obs))
    n_val = int(np.floor(val_size * n_obs))
    

    # Determine the indices for each set
    test_indices = shuffled_indices[:n_test]
    val_indices = shuffled_indices[n_test:n_test+n_val]
    train_indices = shuffled_indices[n_test+n_val:]
    

    anndata.obs.iloc[test_indices, anndata.obs.columns.get_loc('split_anndata')] = 'test'
    anndata.obs.iloc[val_indices, anndata.obs.columns.get_loc('split_anndata')] = 'val'
    anndata.obs.iloc[train_indices, anndata.obs.columns.get_loc('split_anndata')] = 'train'

    return anndata