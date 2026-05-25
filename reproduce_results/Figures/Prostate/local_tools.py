import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.ticker as mticker

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'Liberation Sans'
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['axes.unicode_minus'] = False

short_mapping = {
        'T00': 'T0',
        'T01_Cast_Day1': 'C1',
        'T02_Cast_Day7': 'C7',
        'T03_Cast_Day14': 'C14',
        'T04_Cast_Day28': 'C28',
        'T05_Regen_Day1': 'R1',
        'T06_Regen_Day2': 'R2',
        'T07_Regen_Day3': 'R3',
        'T08_Regen_Day7': 'R7',
        'T09_Regen_Day14': 'R14',
        'T10_Regen_Day28': 'R28'
    }

full_mapping = {
        'T00': 'T0',
        'T01_Cast_Day1': 'Castration Day 1',
        'T02_Cast_Day7': 'Castration Day 7',
        'T03_Cast_Day14': 'Castration Day 14',
        'T04_Cast_Day28': 'Castration Day 28',
        'T05_Regen_Day1': 'Regeneration Day 1',
        'T06_Regen_Day2': 'Regeneration Day 2',
        'T07_Regen_Day3': 'Regeneration Day 3',
        'T08_Regen_Day7': 'Regeneration Day 7',
        'T09_Regen_Day14': 'Regeneration Day 14',
        'T10_Regen_Day28': 'Regeneration Day 28'
    }

def rename_time_points(
    adata, 
    obs_name='time', 
    obs_name_renamed=None,
    mapping_type='short'
):

    if mapping_type=='short':
        mapping = short_mapping
    elif mapping_type=='full':
        mapping = full_mapping
    else:
        raise ValueError('mapping_type should be either short or full')
    
    if obs_name_renamed is None:
        obs_name_renamed = obs_name + '_renamed'
    adata.obs[obs_name_renamed] = adata.obs[obs_name].map(mapping)

    return adata

def plot_gene_expression_box(
    time_labels,
    expression_vectors,
    gene_name= "",
    show_points=True,
    save_path= None,
    color_mapping= None,
    gap_after_first = True,
    gap_size = 0.8,
    y_max = None,
):
    """
    Parameters
    ----------
    time_labels : list
       labels of each time-point
    expression_vectors : list[numpy arrays]
        one np array per time point with expression values
    gene_name : str
        y-axis title
    show_points : bool
        show individual observations on top of the box
    save_path : str
    color_mapping : dict
        dict mapping time labels to color
    gap_after_first : bool
        add extra gap for first time-point
    gap_size : float
    y_max : float
        max y_scale
    """
    # Set style
    sns.set_style(
        "white",
        {
            "axes.edgecolor": "0.25",
            "axes.linewidth": 0.7,
            "grid.color": ".9",
        },
    )
    plt.figure(figsize=(7.0, 5.8), dpi=400)
    ax = plt.gca()
    
    # create first gap for control cells (T00)
    if gap_after_first and len(time_labels) > 1:
        # custom_positions
        x_positions = [0] + [i + gap_size for i in range(1, len(time_labels))]
    else:
        # evenly spaced
        x_positions = list(range(len(time_labels)))
    
    # --- prepare data
    time_position_map = dict(zip(time_labels, x_positions))
    long_df = pd.DataFrame(
        {
            "Time": np.repeat(time_labels, [len(v) for v in expression_vectors]),
            "Expression": np.concatenate(expression_vectors),
            "x_pos": np.repeat(x_positions, [len(v) for v in expression_vectors]),
        }
    )
    
    # colors
    if color_mapping is not None:
        #  create ordered list of colors based on time_labels order
        colors = [color_mapping.get(str(label), "#1f77b4") for label in time_labels]
        palette = colors
    else:
        # use default seaborn palette
        palette = "deep"
        colors = sns.color_palette(palette, len(time_labels))
    
    # plot boxplots
    box_parts = ax.boxplot(
        expression_vectors,
        positions=x_positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        showcaps=True,
        medianprops={"color": "0.3", "linewidth": 1.2},
        whiskerprops={"linewidth": 0.8, "color": "0.3"},
        boxprops={"linewidth": 0.8, "edgecolor": "0.3"},
        capprops={"linewidth": 0.8, "color": "0.3"},
    )
    
    # color the boxes
    for patch, color in zip(box_parts['boxes'], colors):
        if color_mapping is not None:
            import matplotlib.colors as mcolors
            # Make box fill color lighter
            rgb_color = mcolors.to_rgb(color)
            light_color = tuple(min(1.0, c + 0.3) for c in rgb_color)
            patch.set_facecolor(light_color)
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    if show_points:
        for i, (x_pos, expr_values, color) in enumerate(zip(x_positions, expression_vectors, colors)):
            # add jitter to x positions
            n_points = len(expr_values)
            jitter_strength = 0.25 * 0.55 
            np.random.seed(42) 
            x_jittered = np.random.uniform(
                x_pos - jitter_strength, 
                x_pos + jitter_strength, 
                n_points
            )
            
            ax.scatter(
                x_jittered,
                expr_values,
                s=3.2**2, 
                alpha=0.55,
                c=color,
                edgecolors="white",
                linewidths=0.3,
            )
    
    ax.set_xlabel("Time point", fontsize=12, labelpad=8)
    ax.set_ylabel(f"{gene_name} log normalised expression", fontsize=12, labelpad=8)
    ax.set_title(gene_name, loc="left", pad=12, fontsize=12)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(time_labels)
    ax.tick_params(axis="both", labelsize=12)
    
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    
    # spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.25")
        spine.set_linewidth(0.7)
    
    
    if gap_after_first and len(time_labels) > 1:
        ax.set_xlim(-0.5, x_positions[-1] + 0.5)
    
    # set y-max if specified
    if y_max is not None:
        ax.set_ylim(top=y_max)
    
    plt.tight_layout()
    
    # save
    if save_path:
        plt.savefig(f"{save_path}.png", dpi=400, bbox_inches="tight", facecolor="white")
        plt.savefig(f"{save_path}.svg", dpi=400, bbox_inches="tight", facecolor="white")
    
    return plt

def nature_line(ax, x, y, color, marker='o', label=None,
                markersize=6, linewidth=2, markeredgecolor='k', markeredgewidth=0.5):
    ax.plot(x, y, color=color, marker=marker, markersize=markersize,
            linewidth=linewidth, markeredgecolor=markeredgecolor,
            markeredgewidth=markeredgewidth, label=label, zorder=3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)


def phase_bg(ax, n_cast, n_total, cast_bg='#FDECEA', regen_bg='#E8F0FE'):
    ax.axvspan(-0.5, n_cast - 0.5, color=cast_bg, zorder=0)
    ax.axvspan(n_cast - 0.5, n_total - 0.5, color=regen_bg, zorder=0)


def phase_labels(ax, n_cast, n_total):
    yl = ax.get_ylim()
    ax.text((n_cast - 1) / 2, yl[1] * 0.95, 'Castration',
            ha='center', fontsize=8, fontstyle='italic', color='#B71C1C')
    ax.text(n_cast + (n_total - n_cast - 1) / 2, yl[1] * 0.95, 'Regeneration',
            ha='center', fontsize=8, fontstyle='italic', color='#0D47A1')


def nature_legend(ax, title=None):
    leg = ax.legend(title=title, frameon=False, fontsize=8)
    if title:
        leg.get_title().set_fontweight('bold')
        leg.get_title().set_fontsize(9)
    for t in leg.get_texts():
        t.set_fontweight('bold')

def plot_method_scores(scores, colors, metric_label="", savepath=None, dpi=300):
  
    methods = list(scores.keys())
    values = np.array([scores[m] for m in methods])
    bar_colors = [colors[m] for m in methods]

    plt.style.use("seaborn-v0_8-white")

    fig, ax = plt.subplots(figsize=(3.0, 3.0)) 

    # bars
    bar_width = 0.7
    bars = ax.bar(
        x=np.arange(len(methods)),
        height=values,
        color=bar_colors,
        edgecolor="black",
        linewidth=1.0,
        width=bar_width,
    )

    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right",
                       fontsize=10, fontweight="bold")
    ax.set_ylabel(metric_label, fontsize=11, fontweight="bold")
    ax.set_xlabel("")  # cleaner look

    ymin = 0
    ymax = max(values) * 1.15 if max(values) > 0 else 1
    ax.set_ylim(ymin, ymax)

    # Remove gridlines
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    # Clean spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    # Tick params
    ax.tick_params(axis="both", which="both",
                   labelsize=9, width=0.8, length=4)

    # Annotate bars with values
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