1) Run scDisInFact with their recommended default parameters (except latent dimension for shared bio = 16), and save as: `z_cs_scdisinfact.npy`

2) Run CPA with their recommended disentanglement parameters (except latent = 16), and save the latent as `CPA_latent.h5ad`

3) Run `pairwise corr and MI.ipynb`