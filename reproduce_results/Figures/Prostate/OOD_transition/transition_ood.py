import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from scdisentangle.train.tools import set_seed
import scvelo as scv

import sys
sys.path.append('../')
import local_tools as lt
import yaml
import tqdm
    
if __name__ == '__main__':

    # C/R cycle time_points
    time_points = [
        'T01_Cast_Day1', 'T02_Cast_Day7', 
        'T03_Cast_Day14', 'T04_Cast_Day28',
        'T05_Regen_Day1', 'T06_Regen_Day2', 
        'T07_Regen_Day3', 'T08_Regen_Day7',
        'T09_Regen_Day14', 'T10_Regen_Day28'
        ]

    # Config
    yaml_path = '../../../Benchmarks/SCDISENTANGLE/Prostate/configs/prostate.yaml'

    # Set seed
    seed_nb = 42
    set_seed(seed_nb)
    
    all_results_mean = []
    all_results_cell = []
    for idx in range(1, len(time_points)):

        current_time_point = time_points[idx-1]
        future_time_point = time_points[idx]
        
            
        # Read predicted current OOD time-point
        adata_current = sc.read_h5ad(f'../predictions/{current_time_point}.h5ad')

        # Read predicted future OOD time-point
        adata_future = sc.read_h5ad(f'../predictions/{future_time_point}.h5ad')

        # Assert cells are in same order
        assert adata_current.obs['sc_cell_ids'].tolist() == adata_future.obs['sc_cell_ids'].tolist()

        # contruct adata_velocity with unspliced=future and spliced=current
        adata_velocity = adata_current.copy()
        adata_velocity.layers['spliced'] = adata_current.X.copy()
        adata_velocity.layers['unspliced'] = adata_future.X.copy()

        # Create cell_type column
        adata_velocity.obs['cell_type'] = adata_velocity.obs['predType'].copy()
        adata_velocity.obs['clusters'] = adata_velocity.obs['predType'].copy()

        # Compute neighbors
        scv.pp.neighbors(adata_velocity)

        # Compute velocity as future - current
        adata_velocity.layers['velocity'] = adata_velocity.layers['unspliced'] - adata_velocity.layers['spliced']

        # Compute velocity graph
        scv.tl.velocity_graph(adata_velocity)

        probs_mtx = scv.utils.get_transition_matrix(adata_velocity).toarray()
        
        # remove self transitions
        np.fill_diagonal(probs_mtx, 0)

        # renormalize rows
        row_sums = probs_mtx.sum(axis=1, keepdims=True)
        assert (row_sums == 0).sum() == 0
        probs_mtx = probs_mtx / row_sums
        probs_mtx = probs_mtx * 100

        mask_l2 = adata_velocity.obs['cell_type'] == 'Epi_Luminal_2Psca'

        # l2_to_l1 transition probs
        l2_to_l1 = probs_mtx[mask_l2][:, ~mask_l2].sum(axis=1)
        # l1_to_l2 transition probs
        l1_to_l2 = probs_mtx[~mask_l2][:, mask_l2].sum(axis=1)
        
        # Log mean probs results
        results_mean = {
            "current_time": current_time_point,
            "future_time": future_time_point,
            "cell_l2_to_l1_mean": l2_to_l1.mean(),
            "cell_l1_to_l2_mean": l1_to_l2.mean(),
        }

        # Log cell-level pprobs results
        results_cell = {
            "current_time": current_time_point,
            "future_time": future_time_point,
            "cell_l2_to_l1": l2_to_l1,
            "cell_l1_to_l2": l1_to_l2,
        }
        
        all_results_mean.append(results_mean)
        all_results_cell.append(results_cell)

    # To df
    df_results_mean = pd.DataFrame(all_results_mean)
    df_results_mean.to_csv('transition_mean_results.csv')
    
    transition_labels = [f"{r['current_time']}_to_{r['future_time']}" for r in all_results_cell]
    
    per_l2_outflow_df = pd.DataFrame(
        np.column_stack([r["cell_l2_to_l1"] for r in all_results_cell]),
        columns=transition_labels
    )
    per_l2_outflow_df.to_csv('ood_per_l2_cell_outflow.csv', index=False)

    per_l1_outflow_df = pd.DataFrame(
        np.column_stack([r["cell_l1_to_l2"] for r in all_results_cell]),
        columns=transition_labels
    )
    per_l1_outflow_df.to_csv('ood_per_l1_cell_outflow.csv', index=False)