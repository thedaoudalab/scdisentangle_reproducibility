"""
Computational cost benchmark (Supplementary Table 17).

Measures scDisentangle time-per-epoch and peak GPU memory across batch sizes on
the Kang PBMC dataset, and extrapolates full-training time at 521 epochs.

Run:
    export SCDIS_ROOT=/abs/path/to/reproduce_results   # set by the conda env var
    python cost.py
"""
import time
import os
import yaml
import torch
import pandas as pd

from hygeia.data.load_rnaseq import DatasetLoader
from scdisentangle.train.trainer import Trainer
from scdisentangle.train.tools import set_seed

# Config
YAML_PATH = os.environ['SCDIS_ROOT'] + '/Benchmarks/SCDISENTANGLE/Kang/configs/kang.yaml'
BATCH_SIZES = [64, 128, 256, 512, 1024]
MAX_EPOCHS = 100
SEED = 42
DEVICE_ID = 0
OOD_CELL_TYPE = 'B'

print('='*60)
print('COMPUTATIONAL COST BENCHMARK')
print('='*60)

gpu_name = torch.cuda.get_device_name(DEVICE_ID)
gpu_mem_total = torch.cuda.get_device_properties(DEVICE_ID).total_memory / 1e9
print(f'GPU: {gpu_name} ({gpu_mem_total:.0f} GB)')

results = []

for batch_size in BATCH_SIZES:
    print(f'\n--- Batch size: {batch_size} ---')

    set_seed(SEED)

    with open(YAML_PATH, 'r') as f:
        hparams = yaml.safe_load(f)

    hparams['data']['batch_size'] = batch_size
    hparams['hardware']['device'] = DEVICE_ID
    hparams['wandb']['wandb_log'] = False
    hparams['save_experiment']['apply'] = True
    hparams['save_experiment']['experiment_path'] = '/tmp/scdisentangle_benchmark/'
    hparams['save_experiment']['save_weights']['apply'] = False
    hparams['save_experiment']['save_best_weights']['apply'] = False
    hparams['OOD']['filter_dict']['cell_type'] = OOD_CELL_TYPE
    hparams['growing_neurons']['prior_mappers']['mappers']['cell_type_mapper']['collapse_name'] = OOD_CELL_TYPE
    hparams['evaluations']['context_transfer_criterion']['kwargs']['ood_covariate'] = OOD_CELL_TYPE

    os.makedirs('/tmp/scdisentangle_benchmark/', exist_ok=True)

    device = torch.device(f'cuda:{DEVICE_ID}')
    torch.cuda.reset_peak_memory_stats(DEVICE_ID)
    torch.cuda.empty_cache()

    dataset = DatasetLoader(
        path=hparams['data']['file_path'],
        label_keys=hparams['data']['label_keys'],
        default_normalization=hparams['data']['default_normalization'],
        min_gene_counts=hparams['data']['min_gene_counts'],
        min_cell_counts=hparams['data']['min_cell_counts'],
        n_highly_variable_genes=hparams['data']['highly_variable'],
        use_counts=hparams['data']['use_counts'],
        subset=hparams['data']['SUBSET'],
    )

    split_key = f'split_stimulated_{OOD_CELL_TYPE}'
    train_dl, val_dl, test_dl = dataset.train_val_ood_key(batch_size=batch_size, split_key=split_key)

    # Get full dataloader for evaluation
    full_dataloader = dataset.get_dataloader(batch_size=2000)

    n_cells = len(train_dl.dataset)
    n_genes = dataset.data.shape[1]

    trainer = Trainer(
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        test_dataloader=test_dl,
        ctrl_dataloader=None,
        dataloader=full_dataloader,
        dataset=dataset,
        device=device,
        hparams=hparams,
    )

    start = time.time()
    trainer.train(MAX_EPOCHS)
    end = time.time()

    training_time = end - start
    time_per_epoch = training_time / MAX_EPOCHS
    peak_mem = torch.cuda.max_memory_allocated(DEVICE_ID) / 1e9

    results.append({
        'batch_size': batch_size,
        'time_per_epoch_sec': round(time_per_epoch, 2),
        'peak_gpu_memory_gb': round(peak_mem, 2),
        'est_full_training_min': round(time_per_epoch * 521 / 60, 1),
    })

    print(f'Time/epoch: {time_per_epoch:.2f} sec')
    print(f'Peak GPU memory: {peak_mem:.2f} GB')

# Summary
print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f'GPU: {gpu_name}')
print(f'Dataset: Kang ({n_cells} cells, {n_genes} genes)')
print()

df = pd.DataFrame(results)
print(df.to_string(index=False))

# For appendix
print('\n' + '='*60)
print('FOR APPENDIX:')
print('='*60)
print(f'Hardware: {gpu_name}')
print(f'Dataset: Kang PBMC ({n_cells} training cells, {n_genes} genes)')
print()
print('| Batch Size | Time/Epoch (sec) | Peak GPU Memory (GB) | Est. Full Training (min) |')
print('|------------|------------------|----------------------|--------------------------|')
for r in results:
    print(f"| {r['batch_size']:>10} | {r['time_per_epoch_sec']:>16} | {r['peak_gpu_memory_gb']:>20} | {r['est_full_training_min']:>24} |")
