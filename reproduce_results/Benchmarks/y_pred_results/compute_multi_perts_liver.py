import scanpy as sc
import numpy as np
import pandas as pd
import os

import metrics_tools as mt
import gc

def compute_metrics(
    data_name,
    method_name,
    seed_nb,
    adata_original,
    adata_predicted,
    cov_key,
    cond_key,
    ood_cov,
    control_name,
    pert_names,
    _train_median,
    degs_key,
    custom_name='',
    ):
    
    for pert_name in pert_names:
        # split?
        adata_org = adata_original.copy()
        # try:
        #     adata_org.X = adata_org.X.toarray()
        # except:
        #     pass
            
        # DEGs
        degs = adata_org.uns[degs_key][pert_name][ood_cov]
        degs_indices = [adata_org.var_names.get_loc(x) for x in degs]

        # Compute train median
        if _train_median is None:
            train_mask = ~((adata_org.obs[cond_key] == pert_name) & (adata_org.obs[cov_key] == ood_cov))
            train_median = np.median(adata_org[train_mask].X.sum(axis=1))
            if not isinstance(train_median, float):
                train_median = train_median.item()
        else:
            train_median = _train_median

        # Normalize
        sc.pp.normalize_total(adata_org, target_sum=train_median)
        sc.pp.log1p(adata_org)
        
        true_stim = adata_org[
            (adata_org.obs[cond_key] == pert_name) &
            (adata_org.obs[cov_key] == ood_cov)
        ].copy()
        true_ctrl = adata_org[
            (adata_org.obs[cond_key] == control_name) &
            (adata_org.obs[cov_key] == ood_cov)
        ].copy()

        
        true_ctrl = true_ctrl[true_ctrl.obs[f'split_Infected_{ood_cov}'] == 'train'].copy()
            
        context_agnostic = adata_org[
            (adata_org.obs[cond_key] == pert_name) &
            (adata_org.obs[cov_key] != ood_cov) & 
            (adata_org.obs[f'split_Infected_{ood_cov}'] == 'train')
        ].copy()

        pert_agnostic = adata_org[
            ~( adata_org.obs[cond_key].isin([pert_name]) ) &
            (adata_org.obs[cov_key] == ood_cov) & 
            (adata_org.obs[f'split_Infected_{ood_cov}'] == 'train')
        ].copy()

        pred_ctrl = true_ctrl.copy()
        
        if adata_predicted is not None:
            adata_pred = adata_predicted.copy()
            adata_pred = adata_pred[adata_pred.obs[f'split_Infected_{ood_cov}'] == 'train'].copy()
            if adata_pred.uns['X_normalization'] == 'count':
                print(method_name, 'normalizing by count')
                sc.pp.normalize_total(adata_pred, target_sum=train_median)
                sc.pp.log1p(adata_pred)
            
            pred_stim = adata_pred[adata_pred.obs[f'{cond_key}_pred'] == pert_name].copy()

        elif method_name == 'context-agnostic':
            pred_stim = context_agnostic.copy()
            
        elif method_name == 'perturbation-agnostic':
            pred_stim = pert_agnostic.copy()
        elif method_name == 'l1-mean':
            pred_stim = l1_mean.copy()
        else:
            raise ValueError('No prediction anchor')
            
        #pert_mean = context_agnostic.X.mean(axis=0)
        
        #assert pred_stim.shape[0] == true_ctrl.shape[0]
        # Compute metrics
        all_metrics = {}
                        
        # Correlation metrics: nested dict, format (metric_name: {n_degs: value})
        print(pred_stim.X.mean(), pred_stim.shape, pert_name)
        corr_metrics = mt.get_correlations(
            _pred_stim=pred_stim,
            _true_stim=true_stim,
            _pred_ctrl=pred_ctrl,
            _true_ctrl=true_ctrl,
            degs_indices=degs_indices,
            degs_list=[200, 100, 50, 20, 10],
                )
        all_metrics.update(corr_metrics)

        # Distance metrics: same nested structure
        dist_metrics = mt.get_distances(
            _pred_stim=pred_stim,
            _true_stim=true_stim,
            _pred_ctrl=pred_ctrl,
            _true_ctrl=true_ctrl,
            degs_indices=degs_indices,
            degs_list=[200, 100, 50, 20, 10],
            )
        all_metrics.update(dist_metrics)

        # Subset true_ctrl to only include train CTRL cells
        # true_ctrl = true_ctrl[true_ctrl.obs[split_key] == "train"].copy()

        if method_name not in ['context-agnostic', 'perturbation-agnostic'] and 'sc_cell_ids' in pred_stim.obs.columns.tolist():
            # Here sort pred_stim to have same order as true_ctrl (using obs['sc_cell_ids'])
            id_to_idx = {
                cid: i for i, cid in enumerate(pred_stim.obs['sc_cell_ids'])
            }
            order = [id_to_idx[cid] for cid in true_ctrl.obs['sc_cell_ids']]
                    
            pred_stim = pred_stim[order].copy()
    
            assert np.array_equal(
                pred_stim.obs['sc_cell_ids'].to_numpy(), 
                true_ctrl.obs['sc_cell_ids'].to_numpy()
                        )
    
            # Single-cell preservation metric (Identity preservation)
            sc_sim_nested = mt.get_sc_similarity(
                _pred_stim=pred_stim,
                _true_ctrl=true_ctrl,
            )
            sc_sim_values = sc_sim_nested['All']
            sc_degs = [200, 100, 50, 20, 10]
            for metric_name, value in sc_sim_values.items():
                all_metrics[metric_name] = {n: value for n in sc_degs}

        # Save metrics: rows = metric names, columns = DEG subset sizes
        metrics_df = pd.DataFrame(all_metrics).T
        metrics_df.index.rename("Metric", inplace=True)

        save_path = f"results/{data_name}/{method_name}"
        os.makedirs(save_path, exist_ok=True)
        
        metrics_df.to_csv(
            f"{save_path}/{custom_name}{ood_cov}_{pert_name}_{seed_nb}.csv"
        )

        del adata_org, true_ctrl, true_stim, pred_ctrl, pred_stim, context_agnostic, pert_agnostic
        gc.collect()