import os

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import anndata as ad
import yaml
from tqdm import tqdm
from hygeia.data.load_rnaseq import DatasetLoader
from scdisentangle.train.trainer import Trainer
import gc

import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'Liberation Sans'
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['axes.unicode_minus'] = False

yaml_path = '../../Benchmarks/SCDISENTANGLE/Prostate/configs/prostate.yaml'
weights_dir = '../../Benchmarks/SCDISENTANGLE/Prostate/weights/prostate/42'

ctype_palette = {
    'Epi_Luminal_1': '#d62728',
    'Epi_Luminal_2Psca': '#264653'
}
time_points = [
    'T01_Cast_Day1', 'T02_Cast_Day7', 
    'T03_Cast_Day14', 'T04_Cast_Day28',
    'T05_Regen_Day1', 'T06_Regen_Day2', 
    'T07_Regen_Day3', 'T08_Regen_Day7',
    'T09_Regen_Day14', 'T10_Regen_Day28',
    'T00'
]

l1_l2_cells = ['Epi_Luminal_1', 'Epi_Luminal_2Psca']

for time_ood in tqdm(time_points):
    #if os.path.isfile(f'predictions/{time_ood}.h5ad'):
        #print(time_ood, 'already processed')
        #continue
    print('Processing', time_ood)
    # Read original data
    adata = sc.read_h5ad('../../Datasets/preprocessed_datasets/prostate.h5ad')
    if not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.toarray()

    # Subset to train (remove OOD Luminal_2Psca at current time_ood)
    ood_mask = ~(
        (adata.obs['time'] == time_ood)
        & (adata.obs['predType'] == 'Epi_Luminal_2Psca')
    )
    if time_ood != 'T00':
        adata_train = adata[ood_mask].copy()
    else:
        adata_train = adata.copy()

    # Compute Median of training set
    _sums = adata_train.X.sum(axis=1, keepdims=True)
    data_median = np.median(_sums)
    
    # Filter cell types
    ctypes_to_filter = [
        'Epi_SV_Ionocyte', 'Epi_SV_Luminal', 'Epi_SV_Basal',
        'SymDoublet_Epi_Imm', 'SymDoublet_Str_Epi', 'SymDoublet_Str_Imm'
        ]
    adata_train = adata_train[~adata_train.obs['predType'].isin(ctypes_to_filter)]
    adata = adata[~adata.obs['predType'].isin(ctypes_to_filter)]
    
    # Read params, collapse time to time_ood
    with open(yaml_path, 'r') as stream:
        hparams = yaml.safe_load(stream)
    hparams['wandb']['wandb_log'] = False #True
    hparams['OOD']['filter_dict']['time'] = time_ood
    hparams['growing_neurons']['prior_mappers']['mappers']['time_mapper']['collapse_name'] = time_ood
    hparams['save_experiment']['apply'] = False
    hparams['save_experiment']['save_weights']['apply'] = False
    hparams['save_experiment']['save_best_weights']['apply'] = False

    if not 'sc_cell_ids' in hparams['data']['label_keys']:
        hparams['data']['label_keys'].append('sc_cell_ids')

    # Get dataloaders
    dataset = DatasetLoader(
                path=hparams['data']['file_path'],
                label_keys=hparams['data']['label_keys'],
                default_normalization=hparams['data']['default_normalization'],
                min_gene_counts=hparams['data']['min_gene_counts'],
                min_cell_counts=hparams['data']['min_cell_counts'],
                n_highly_variable_genes=hparams['data']['highly_variable'],
                use_counts = hparams['data']['use_counts'],
                subset=hparams['data']['SUBSET'],
            )

    # Dataloader cont containing all data samples
    dataloader = dataset.get_dataloader(batch_size=2000)
    
    # Train test val split
    train_dataloader, val_dataloader, test_dataloader = dataset.train_val_ood(
                train_size=hparams['data']['train_size'],
                batch_size=hparams['data']['batch_size'],
                filter_dict=hparams['OOD']['filter_dict']
            )

    # Get trainer
    trainer = Trainer(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                dataloader=dataloader,
                dataset=dataset,
                device=hparams['hardware']['device'],
                hparams=hparams,
                )
    
    # Load weights
    
    weights_path = f'{weights_dir}/{time_ood}/r2_mean_criterion_10_all/'
    if time_ood == 'T00':
        weights_path = f'{weights_dir}/T05_Regen_Day1/r2_mean_criterion_10_all/'
        
    trainer.load_weights(weights_path)
    
    adata_pred = trainer.predict(
        adata_train.copy(),
        counterfactual_dict={'time': time_ood}
    )

    adata_gt = adata.copy()

    # Normalize
    sc.pp.normalize_total(adata_pred, target_sum=data_median)
    sc.pp.log1p(adata_pred)

    sc.pp.normalize_total(adata_gt, target_sum=data_median)
    sc.pp.log1p(adata_gt)
    
    # Make PCA & preds dirs
    os.makedirs('PCA/gt', exist_ok=True)
    os.makedirs('PCA/pred', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)
    os.makedirs('ground_truth', exist_ok=True)

    # PCA:
    sc.pp.pca(adata_pred)
    sc.pp.pca(adata_gt)

    adata_pred_filtered = adata_pred[(adata_pred.obs['predType'].isin(l1_l2_cells)) & (adata_pred.obs['time_org'] == 'T00')].copy()
    try:
        adata_pred_filtered.uns.pop("predType_colors", None)
    except:
        pass
    
    fig, ax = plt.subplots(figsize=(4, 4))
    sc.pl.pca(
        adata_pred_filtered,
        save=None,
        show=False,
        color='predType',
        ax=ax,
        palette=ctype_palette
    )
    ax.set_aspect('equal')
    ax.set_box_aspect(1)
    for coll in ax.collections:
        coll.set_rasterized(True)
    
    plt.savefig(f'PCA/pred/{time_ood}.png', dpi=300)
    plt.savefig(f'PCA/pred/{time_ood}.svg', dpi=400)

    adata_gt_filtered = adata_gt[(adata_gt.obs['predType'].isin(l1_l2_cells)) & (adata_gt.obs['time'] == time_ood)]
    try:
        adata_gt_filtered.uns.pop("predType_colors", None)
    except:
        pass
    
    fig, ax = plt.subplots(figsize=(4, 4))
    sc.pl.pca(
        adata_gt_filtered,
        save=None,
        show=False,
        color='predType',
        ax=ax,
        palette=ctype_palette
    )
    ax.set_aspect('equal')
    ax.set_box_aspect(1)

    for coll in ax.collections: 
        coll.set_rasterized(True)
    
    plt.savefig(f'PCA/gt/{time_ood}.png', dpi=400)
    plt.savefig(f'PCA/gt/{time_ood}.svg', dpi=400)

    adata_pred.uns = {
        'median': data_median,
        'X_normalization': 'median',
        'train_size': adata_train.shape[0],
        'train_mean': adata_train.X.mean()
    }
    
    adata_pred_luminal = adata_pred[(adata_pred.obs['time_org'] == 'T00') & (adata_pred.obs['predType'].isin(l1_l2_cells))].copy()
    
    adata_gt = adata_gt[(adata_gt.obs['time'] == time_ood) & (adata_gt.obs['predType'].isin(l1_l2_cells))].copy()
    
    # Save predictions
    adata_pred_luminal.write_h5ad(f'predictions/{time_ood}.h5ad')
    adata_gt.write_h5ad(f'ground_truth/{time_ood}.h5ad')

    del adata_pred, adata_pred_luminal, adata_gt, adata, adata_train
    gc.collect()