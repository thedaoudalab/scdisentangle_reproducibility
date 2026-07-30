import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from copy import deepcopy
from scipy.cluster.hierarchy import dendrogram, leaves_list
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'Liberation Sans'
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['axes.unicode_minus'] = False

def order_linkage(Z, clusters, desired_order):
   
    n = len(clusters)
    Z_out = Z.astype(float).copy()

    label_to_idx = {c: i for i, c in enumerate(clusters)}
    pos = {label_to_idx[c]: p for p, c in enumerate(desired_order)}

    def _leaves(node):
        """all leaf indices under node"""
        if node < n:
            return [int(node)]
        row = int(node - n)
        return _leaves(int(Z_out[row, 0])) + _leaves(int(Z_out[row, 1]))

    def _min_pos(node):
        """the smallest desired position among leaves of node"""
        return min(pos[lf] for lf in _leaves(node))

    for row in range(len(Z_out)):
        left, right = int(Z_out[row, 0]), int(Z_out[row, 1])
        if _min_pos(left) > _min_pos(right):
            Z_out[row, 0], Z_out[row, 1] = Z_out[row, 1], Z_out[row, 0]

    # sanity check
    final = [clusters[i] for i in leaves_list(Z_out)]
    assert final == list(desired_order), (
        f"Desired order is not compatible with tree topology.\n"
        f"  achieved: {final}\n"
        f"  desired:  {list(desired_order)}"
    )

    return Z_out

def em_cluster(
    adata,
    n_comps,
    cell_type_key='cell_type',
    covariance_type='full',
    random_state=0
    ):

    # Get X
    X = adata.X.copy()
    
    # Init GMM Model
    gmm = GaussianMixture(
        n_components=n_comps,
        covariance_type=covariance_type,
        init_params='kmeans',
        n_init=10,
        random_state=random_state,
    )

    # Fit GMM
    gmm.fit(X)

    # Probs
    probs = gmm.predict_proba(X)
    
    return probs

def get_tree(
    adata,
    cell_type_key='cell_type',
    n_comps_scale=3,
    random_state=0,
    ):

    # To category
    adata.obs[cell_type_key] = adata.obs[cell_type_key].astype(str)
    
    # compute step
    n_unique = adata.obs[cell_type_key].nunique()
    if n_unique <= 1:
        raise ValueError("Need at least 2 cell types to compute step.")
        
    computed_step = int(16 // (n_unique - 1))

    if computed_step < 1:
        computed_step = 1
            
    step = computed_step
    print(f'using step={step}')

    # unique clusters
    clusters = adata.obs[cell_type_key].unique().tolist()

    # list of clusters to which we will be appending new merged ones (for dendrogram)
    new_clusters = deepcopy(clusters)

    # n_comps
    n_comps = int(len(clusters) * n_comps_scale)
    print(f'n_comps: {n_comps}')
    
    # mappings dictionary will be used to rename cell types into their merged clusters
    mappings = {}

    # used to log merged clusters
    merged_levels = {}

    # Init current_level to last latent level
    current_level = int(adata.obs['level'].max())

    # Init iter_step to 1 (this will be used as a distance proxy)
    iter_step = 1

    # Init rows (linkage_mtx)
    rows = []

    # loop through latent levels backwards, with step, 1 merging per iteration
    while True:
        # subset to current level
        adata_subset = adata[adata.obs['level'] == current_level].copy()

        # rename cell types to their merged cluster
        for mapping_key, mapping_value in mappings.items():
            adata_subset.obs[cell_type_key] = adata_subset.obs[cell_type_key].replace(mapping_key, mapping_value)

        # fit and predict hard labels with GMM
        probs = em_cluster(
            adata=adata_subset,
            n_comps=n_comps,
            cell_type_key=cell_type_key,
            random_state=random_state,
        )
            
        counts_df = {}
        for ct in adata_subset.obs[cell_type_key].unique().tolist():
            mask = (adata_subset.obs[cell_type_key] == ct)
            counts_df[ct] = probs[mask].mean(axis=0).squeeze().tolist()
        counts_df = pd.DataFrame(counts_df).T

        # rows already normalized to sum=1
        X = counts_df.fillna(0.0).to_numpy()
    
        # similarities (pairwise dot products)
        G = X @ X.T
    
        # exclude self
        np.fill_diagonal(G, -np.inf)
    
        # Get index of best pair
        i, j = np.unravel_index(np.argmax(G), G.shape)
    
        # Get cell type labels of best pairs
        best_pair = (counts_df.index[i], counts_df.index[j])
            
        # to list
        best_pair = list(best_pair)

        # add pair to dict
        for pair in best_pair:
            mappings[pair] = '&'.join(best_pair)

        # log merging
        merged_levels[current_level] = best_pair

        # step backward
        current_level -= step

        # Number of original samples in newly formed cluster
        n_samples = sum([_.count('&') + 1 for _ in best_pair])

        # create linkage row
        row = [new_clusters.index(best_pair[0]), new_clusters.index(best_pair[1]), float(iter_step), n_samples]
        rows.append(row)

        # add new cluster
        new_clusters.append('&'.join(best_pair))

        # Iteration step used as a distance measure
        iter_step += 1

        # Stop when nothing more to merge
        if len(counts_df) == 2:
            break

        # Or when we reach lowest latent level
        elif current_level < 0:
            break

    # create linkage matrix array
    linkage_mtx = np.array(rows)

    # log results
    tree_results = {
        'merged_levels': merged_levels,
        'linkage_mtx': linkage_mtx,
        'clusters': clusters
    }
    
    return tree_results

def plot_dendro(
    linkage_mtx, 
    clusters, 
    figsize=(14, 6),
    save_path=None,
    linewidth=2.5,
    orientation='bottom',
    label_fontsize=14,
    desired_order=None
    ):
    
    hierarchy.set_link_color_palette(['#B2182B', '#4393C3', '#1B9E77', '#762A83'])
    
    plt.figure(figsize=figsize)

    if desired_order is not None:
        linkage_mtx_ordered = order_linkage(linkage_mtx, clusters, desired_order)
        leaf_order = leaves_list(linkage_mtx_ordered)
    else:
        linkage_mtx_ordered = linkage_mtx
        leaf_order = None
        
    # dendrogram
    dendrogram(
        linkage_mtx_ordered,
        labels=clusters,
        orientation=orientation,
        above_threshold_color='#4D4D4D',
        count_sort=False,
        distance_sort=False
    )

    ax = plt.gca()

    # increase line width
    for coll in ax.collections:
        coll.set_linewidth(linewidth)

    # font and tick
    ax.tick_params(axis='both', which='major', labelsize=label_fontsize)
    
    # Save
    if save_path is not None:
        print(f'saving to {save_path}')
        plt.savefig(save_path + '.png', dpi=600, bbox_inches='tight')
        plt.savefig(save_path + '.svg', dpi=600, bbox_inches='tight')
        #plt.savefig(save_path + '.pdf', dpi=600, bbox_inches='tight')
    
    plt.show()
    return leaf_order