# Norman (Perturb-seq) preprocessing

Run these notebooks in order:
1. `1) default_pp.ipynb`: filtering, HVG/DEG selection. It writes `norman.h5ad` to `../../preprocessed_datasets/` 
2. `2) create_splits.ipynb`
3. `3) per_pert_split.ipynb`

## Non-deterministic ordering

`1) default_pp.ipynb` does not produce a deterministic **gene** or
**perturbation** category ordering across runs. To reproduce paper numbers exactly, download the exact **`norman.h5ad`: [from Zenodo](https://zenodo.org/records/21543559/files/norman.h5ad?download=1)**, place it in `../../preprocessed_datasets/` and skip `1) default_pp.ipynb`.

Source: Norman et al., *Science* **365**, 786–793 (2019), via scPerturb
(Peidli et al., *Nat. Methods* **21**, 531–540, 2024; Zenodo
10.5281/zenodo.7041849). Redistributed under CC-BY 4.0.
