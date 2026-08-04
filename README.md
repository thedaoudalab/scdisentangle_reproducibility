# scDisentangle reproducibility code

Reproducibility code and analysis notebooks for *scDisentangle: sequential disentanglement for single-cell perturbation response prediction*.
The package source for `scdisentangle` and `hygeia` is under `package/`.
Everything needed to regenerate the results is under `reproduce_results/`.

> **Package and tutorials:** The maintained `scdisentangle` and `hygeia` packages with tutorial notebooks live at [thedaoudalab/scDisentangle](https://github.com/thedaoudalab/scDisentangle).


## Setup

```bash
conda create -n scdisentangle_reproduce python=3.10 -y
conda activate scdisentangle_reproduce
conda env config vars set SCDIS_ROOT=/abs/path/to/reproduce_results
conda activate scdisentangle_reproduce
export PYTHONNOUSERSITE=1
pip install -r package/requirements-scvi.txt
pip install -r package/requirements.txt
pip install -e package/hygeia -e package/scdisentangle
```


To use jupyter notebooks:
```bash
pip install ipykernel
python -m ipykernel install --user --name=scdisentangle_reproduce --display-name="Python (scdisentangle_reproduce)"
```
Then in jupyter notebook, select the Kernel Python (scdisentangle_reproduce)

## Data

Place the raw h5ad files in `reproduce_results/Datasets/original_datasets/<dataset>/`:

| Dataset | Accession / source |
|---|---|
| Kang *IFN-β* PBMC | https://drive.google.com/drive/folders/1n1SLbXha4OH7j7zZ0zZAxrj_-2kczgl8 |
| Liver *Plasmodium* | https://figshare.com/articles/dataset/spatio-temporal-infection_infected/22148900?file=39375713 |
| Norman Perturb-seq | https://zenodo.org/records/7041849 |
| Myocarditis | GEO GSE228597 |
| Prostate castration-regeneration | GEO GSE146811 |
| Hao PBMC ("Seurat") | scvi-tools (v0.20.3) |
| B-ALL ("Leukemia") | https://cellxgene.cziscience.com/collections/14dc301f-d4fb-4743-a590-aa88d5f1df1a |

Then run the matching notebook in
`reproduce_results/Datasets/preprocessing_scripts/<dataset>/`. Resulting preprocessed datasets are saved to
`Datasets/preprocessed_datasets/`. For Norman, run the three notebooks in
numeric order. For exact Norman reproducibility, download the preprocessed `h5ad` file and skip the first pre-processing notebook (See `reproduce_results/Datasets/preprocessing_scripts/Norman/README.md`).

## Reproducing the paper

All paths below are relative to `reproduce_results/`.

The workflow for every dataset is:

1. Pre-process (above).
2. Train: `Benchmarks/SCDISENTANGLE/<dataset>/train_jobs*.py`.
3. Predict: `Benchmarks/SCDISENTANGLE/<dataset>/get_predictions.py` (Norman
   uses `infer_norman_{1,2}.py`).
4. Evaluate: `Benchmarks/y_pred_results/<dataset>.ipynb`.
  - Note that this requires:
    - Either training all models + running inference.
    - Use already computed metrics, and skip the `cm.compute_metrics` cells (jump directly to plotting and statistical testing cells). We provide the pre-computed metrics in `Benchmarks/y_pred_results/results.zip`. Unzip in the same folder:
   ```bash
   cd Benchmarks/y_pred_results
   unzip results.zip && rm results.zip
   ```
      
5. Per-dataset analyses: notebooks in `Figures/<dataset>/`.

### Kang *IFN-β*: Fig. 2, Extended Data Fig. 1, Supplementary Fig. 1

```bash
cd Benchmarks/SCDISENTANGLE/Kang
python train_jobs.py
python get_predictions.py
```
- Benchmark boxplots and barplots (Fig. 2a-c; Extended Data Fig. 1): `Benchmarks/y_pred_results/Kang.ipynb`
- Violin panels (Fig. 2d; Supplementary Fig. 1): `Figures/Kang/violinplot.ipynb`

### Liver *Plasmodium*: Fig. 2e-j, Extended Data Figs. 2 and 3, Supplementary Fig. 19

- **Zone-held-out infection prediction.** 
1. Train: `Benchmarks/SCDISENTANGLE/Liver/train_jobs_infected.py`
2. Predict: `Benchmarks/SCDISENTANGLE/Liver/get_predictions.py`
3. Generate benchmark boxplots and barplots (Fig. 2e-g; Extended Data Fig. 2): `Benchmarks/y_pred_results/Liver_infected.ipynb`
4. Generate violinplots (Fig. 2h): `Figures/Liver/violinplot.ipynb`

- **Hours-post-infection trajectory.**
1. To generate UMAPs and dot plots in Fig. 2i,j and Extended Data Fig. 3 :
    - `Benchmarks/SCDISENTANGLE/Liver/train_jobs_hpi.py`
    - `Figures/Liver_HPI/get_preds.py`
    - `Figures/Liver_HPI/UMAP.ipynb`
    - `Figures/Liver_HPI/Dot_plot.ipynb`

2. To generate Supplementary Fig. 19: 
    - `Benchmarks/SCDISENTANGLE/Liver_HPI/train_jobs_hpi.py`
    - `Benchmarks/SCDISENTANGLE/Liver_HPI/get_predictions.py`
    - `Benchmarks/y_pred_results/Liver_HPI.ipynb`

### Norman Perturb-seq: Fig. 2k,l, Extended Data Figs. 4-6; Supplementary Figs. 2-8 and 18

```bash
cd Benchmarks/SCDISENTANGLE/Norman
python train_jobs_combinatorial.py   # combinatorially-seen scenario
python train_jobs_single.py          # single-only scenario
python infer_norman_1.py && python infer_norman_2.py # for inference on each scenario
```

Combinatorially-seen and single-only metrics (Extended Data Figs. 4 and 6):
`Benchmarks/y_pred_results/Norman.ipynb`.
- Combination-specific DEGs (Supplementary Figs. 2 and 6): `Benchmarks/y_pred_results/Norman_combo_specific.ipynb`.
- Distributional and perturbed-reference metrics in Supplementary Figs. 7 and 8 are generated by these two notebooks.
- Scatter plots (Fig. 2l; Extended Data Fig. 5; Supplementary Figs. 4 and 5): `Figures/Norman/scatter.ipynb`.
- Systematic variation (Fig. 2k; Supplementary Fig. 3): `Figures/Norman/systematic_variation.ipynb`.
- To generate Supplementary Fig. 18: run `Figures/Norman/Supplementary_Fig_18.ipynb`.


### Myocarditis: Fig. 3, Supplementary Figs. 9 and 10

```bash
cd Benchmarks/SCDISENTANGLE/Myocarditis
python train_jobs.py
python get_predictions.py
```

- CD8+ T subset metrics (Fig. 3b,c; Supplementary Fig. 9): `Benchmarks/y_pred_results/Myocarditis_cd8.ipynb`.
- Full-dataset metrics (Supplementary Fig. 10): `Benchmarks/y_pred_results/Myocarditis.ipynb`.
- PCA and distribution panel (Fig. 3a,d): `Figures/Myocarditis/PCA and distribution.ipynb`.

### Prostate castration-regeneration: Fig. 4, Extended Data Fig. 7, Supplementary Figs. 11–13

```bash
cd Benchmarks/SCDISENTANGLE/Prostate
python train_jobs.py
cd ../../../Figures/Prostate
python get_pred_cast_cycle.py
```

- Fig. 4d and Supplementary Fig. 11 are generated by get_pred_cast_cycle.py.
- Disentanglement and co-clustering (Fig. 4a-c,e,f; Supplementary Figs. 12 and 13): `Figures/Prostate/disentangle.ipynb`,
`Figures/Prostate/co_clustering.ipynb`.
- OOD luminal cell-state transition (Fig. 4g-j; Extended Data Fig. 7): run
`Figures/Prostate/OOD_transition/transition_ood.py` first to get transition probabilities, then the three notebooks in that folder.


### Latent structure: Fig. 1d, Fig. 5, Extended Data Figs. 8-10, Supplementary Figs. 14-17

- First run `Figures/<dataset>/disentangle.ipynb`

- Pairwise correlation and mutual information between latent factors on Kang (Fig. 1d, Extended Data Fig. 8): Run `Figures/Kang/orthogonal_factors/pairwise corr and MI.ipynb`

- For each of Kang, Hao (Seurat), and B-ALL (Leukemia), run the numbered notebooks in `Figures/<dataset>/latent_structure/` in order. This generates Fig. 5, Extended Data Figs. 9 and 10, and Supplementary Figs. 14-16.

- Perturbation-effect decomposition (Supplementary Fig. 17): Run the two notebooks under
`Figures/Kang/latent_structure/Perturbation effect across levels/` (
`Generate_data.ipynb` then `Pert effect analysis.ipynb`).

## Competitor predictions

CPA, scGEN, scDisInFact, biolord and GEARS predictions are needed for the
benchmark figures. To reproduce:

1. Install the competitor at the version listed in
   `reproduce_results/Benchmarks/competitors_versions.md`.
2. Train using their recommended hyperparameters (detailed in Supplementary Note 3).
3. Write predictions to
   `Benchmarks/<METHOD>/<DATASET>/predictions/<ood_cov>_<seed>.h5ad`
   (Norman uses `predictions/<scenario>/<pert>.h5ad`).
   format of predictions:
   - `.X`: predicted expression data (as predicted by the model), i.e. **Counts for scDisentangle, CPA, and scDisInFact**, or **log-normalized for scGen, biolord, and GEARS**. Since biolord trains on log-1e4-normalized data, rescale to median via: expm1 -> rescale to median -> log1p. For count-based models (scDisentangle, CPA, and scDisInFact), set `adata_pred.uns['X_normalization'] = 'count'`, otherwise (scGen, biolord, and GEARS) set it to `adata_pred.uns['X_normalization'] = 'log-norm'`. Normalization for count predictions is handled by the evaluation framework. All models are evaluated on the same scale (`sc.pp.normalize_total(adata_pred, target_sum=median)`, then `sc.pp.log1p(adata_pred)`). The median used for normalization is always computed on the training-set. For Norman Perturb-seq, that median is computed from the singly-perturbed and control cells (i.e. the subset that is invariant to the OOD settings; all OOD settings in our benchmarks are double-perturbations).
   - `.obs[<perturbation_name>_pred]` and `.obs[<perturbation_name>]`: Should be set to the predicted perturbation label.
   - `.obs[<perturbation_name>_org]`: Should be set to the source cell label (cells taken as input e.g. `ctrl`).
   Note that all models should take as input all control cells of the OOD context from the **training-set** (e.g. `control CD4 T` cells when the OOD context is `stimulated CD4 T cells`). **Validation-set control cells are not used for prediction**. In Norman, this corresponds to all control cells in the dataset (as the validation-set does not include control cells).

## Supplementary tables

Notebooks in `reproduce_results/Supplementary Tables/`:

- `Dataset_summary.ipynb` → Supplementary Tables 1, 4, 14-16
- `Norman splits.ipynb` → Supplementary Table 5
- `Benchmarks.ipynb` → Supplementary Tables 2, 3, 6, 7, 9–12
- `Hyperparameters.ipynb` → Supplementary Table 13
- `Benchmarks/Computational_cost/cost.py` → Supplementary Table 17

## License
MIT. See `LICENSE`

