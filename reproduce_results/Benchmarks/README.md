Competitors' predictions should be inside `<competitor_name>/<dataset_name>/predictions/<ood_scenario>_<seed_number>.h5ad` e.g. `CD4 T_1.h5ad`
The anndata objects var_names should mirror the original data var_names, and they should contain the following obs columns:
`<pert_name>_org`: refering to the label of control samples (those taken as input to produce counterfactuals).
`<pert_name>_pred` and `<pert_name>`: refering to the label of counterfactuals.
e.g. a control sample with `condition=control` mapped to `condition=stimulated` would have `condition_org=control`, `condition_pred=stimulated`, and `condition=stimulated`.
They should additionally contain to the context obs column e.g. `cell_type`, `zone`.

