import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from tqdm import tqdm
import matplotlib

# Set plt params
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'Liberation Sans'
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['axes.unicode_minus'] = False

def compute_mutual_info_matrix(
    data, 
    n_neighbors=3, 
    random_state=42
    ):
    """Compute pairwise MI mtx"""
    n_samples = data.shape[0]
    n_features = data.shape[1]
    mi_matrix = np.zeros((n_features, n_features))
    
    for i in tqdm(range(n_features), desc="Computing MI"):
        for j in range(i, n_features):
            # MI is symmetric so we compute only upper triangle
            assert data[:, i].reshape(-1, 1).shape == (n_samples, 1)
            assert data[:, j].shape == (n_samples, )
            mi = mutual_info_regression(
                data[:, i].reshape(-1, 1),
                data[:, j],
                n_neighbors=n_neighbors,
                discrete_features=False,
                random_state=random_state
            )[0]
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi # sym
            
    return mi_matrix

def corr_heatmap(
    corr_matrix,
    save_path,
    method_name
    ):
    """ Plots correlation heatmap """
    
    # create fig
    plt.figure(
        figsize=(10, 8), 
        dpi=300
    )
    
    # create sns heatmap
    heatmap = sns.heatmap(
        corr_matrix,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="black",
        cbar_kws={
            'shrink': 0.8, 
            }
    )
    
    # appearance
    heatmap.set_xticklabels(heatmap.get_xticklabels(), 
                           rotation=45, 
                           horizontalalignment='right',
                           fontsize=10)
    
    heatmap.set_yticklabels(heatmap.get_yticklabels(), 
                           rotation=0, 
                           fontsize=10)
    
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)           
    plt.savefig(f'{save_path}/latent_factor_correlation_{method_name}.svg', dpi=500,bbox_inches="tight")
    plt.savefig(f'{save_path}/latent_factor_correlation_{method_name}.png', dpi=500,bbox_inches="tight")

def MI_heatmap(
    mi_matrix,
    save_path,
    method_name,
    vmax=0.3985,
    ):
    
    # create labels and mi_df
    factor_labels = [f"LF{i+1}" for i in range(mi_matrix.shape[1])]
    mi_df = pd.DataFrame(mi_matrix, index=factor_labels, columns=factor_labels)
    
    # create fig
    plt.figure(figsize=(10, 8), dpi=300)
    
    # create heatmap
    heatmap = sns.heatmap(
        mi_df,
        cmap='Blues',
        vmin=0,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        linecolor="black",
        annot=False,
        fmt=".2f",
        cbar_kws={
            'shrink': 0.8,
        }
    )
    
    # appearance
    heatmap.set_xticklabels(heatmap.get_xticklabels(), 
                           rotation=45, 
                           horizontalalignment='right',
                           fontsize=10)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), 
                           rotation=0, 
                           fontsize=10)
    
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)           
    plt.savefig(f'{save_path}/latent_factor_MI_{method_name}.svg', dpi=500, bbox_inches="tight")
    plt.savefig(f'{save_path}/latent_factor_MI_{method_name}.png', dpi=500, bbox_inches="tight")

def plot_method_scores(
    scores, 
    colors, 
    metric_label="Correlation", 
    savepath=None, 
    dpi=500,
    figsize=(3.0, 3.0)
    ):
    methods = list(scores.keys())
    values = np.array([scores[m] for m in methods])
    bar_colors = [colors[m] for m in methods]

    plt.style.use("seaborn-v0_8-white")

    fig, ax = plt.subplots(figsize=figsize) 

    bar_width = 0.7
    bars = ax.bar(
        x=np.arange(len(methods)),
        height=values,
        color=bar_colors,
        edgecolor="black",
        linewidth=1.0,
        width=bar_width,
    )

    # labels and sticks
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right",
                       fontsize=10, fontweight="bold")
    ax.set_ylabel(metric_label, fontsize=11, fontweight="bold")
    ax.set_xlabel("")  # cleaner look

    # y_limit
    ymin = 0
    ymax = max(values) * 1.15 if max(values) > 0 else 1
    ax.set_ylim(ymin, ymax)

    # remove grtids
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    # spine
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    ax.tick_params(axis="both", which="both",
                   labelsize=9, width=0.8, length=4)

    # annot bars with vars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + (ymax * 0.01),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    fig.tight_layout(pad=0.5)

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    return fig, ax