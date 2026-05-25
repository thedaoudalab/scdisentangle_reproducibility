import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'Liberation Sans'
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['axes.unicode_minus'] = False

def violin_plot(
    adata,
    variable_of_interest,
    label_column_name,
    x_name,
    y_name,
    title,
    violin_color='Slategray',
    label_color_dict=None,
    figsize=(7.08, 5),
    avg_point_size=12,
    dpi=300,
    save_path=None,
    inner='quart',
    rotation=45,
    despine=True
    ):

    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi

    avg_font_size = avg_point_size
    min_font_size = avg_point_size - 1
    max_font_size = avg_point_size + 1

    # Set font properties
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = avg_point_size

    # Relevant Colors: 'Black' 'Cadetblue' #'Darkcyan' #'Royalblue' #'Slategray'
    adata_copy = adata.copy()
    # Verify gene exists
    if variable_of_interest not in adata_copy.var_names:
        raise ValueError(f"{variable_of_interest} not found in adata_cat.var_names")

    # Extract expression values
    _X = adata_copy[:, variable_of_interest].X
    
    adata_copy.obs[f'expression_{variable_of_interest}'] = _X.toarray().flatten()

    # Sort labels for consistent plotting
    sorted_labels = list(label_color_dict) #sorted(adata_copy.obs[label_column_name].unique())

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    if label_color_dict is not None:
        #Plot each label individually with its color
        for label in sorted_labels:
            subset = adata_copy.obs[adata_copy.obs[label_column_name] == label]
            sns.violinplot(
                data=subset,
                x=label_column_name,
                y=f'expression_{variable_of_interest}',
                order=sorted_labels,#[label],
                color=label_color_dict.get(label, violin_color),
                inner=inner,#'quartile',
                linewidth=1.2,
                ax=ax
            )
    else:
        sns.violinplot(
            data=adata_copy.obs,
            x=label_column_name,
            y=f'expression_{variable_of_interest}',
            order=sorted_labels,
            color=violin_color,
            inner='quartile',
            linewidth=1.2,
            ax=ax
        )

    # Beautification (Nature Biotech guidelines)
    if despine:
    	sns.despine(trim=True, offset=10)

    ax.set_title(title, fontsize=avg_font_size)
    ax.set_xlabel(x_name, fontsize=avg_font_size)
    ax.set_ylabel(y_name, fontsize=avg_font_size)

    # Adjust tick labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='center', fontsize=avg_font_size)

    # Adjust layout neatly
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f'{save_path}.png', dpi=300)
        plt.savefig(f'{save_path}.svg', dpi=300)

    plt.show()

def plot_data(
    data_dict,
    plot_type='boxplot',
    y_name=None,
    x_name=None,
    title=None,
    save_path=None,
    legend=True,
    show_grid=True,
    figsize=(7.08, 5),
    dpi=300,
    avg_point_size=12,
    colors=None,
    y_min=None,                # ← NEW PARAMETER
    sort_order=None,
    stars=None
):
    """
    Generates a boxplot or barplot from a nested dictionary with Nature Biotech styling,
    including error bars (barplot only) and consistent method ordering.

    Parameters
    ----------
    data_dict : dict
        {method: {'group1': [vals], 'group2': [vals], ...}}
    plot_type : {'boxplot', 'barplot'}, default 'boxplot'
    y_name, x_name : str or None
        Axis labels.
    title : str or None
        Plot title (non‑bold, per NB style).
    save_path : str or None
        Path (without extension) to save figure.
    legend : bool, default True
        Whether to display the legend.
    show_grid : bool, default True
        Whether to show horizontal grid lines.
    figsize : tuple, default (7.08, 5)
        Figure size in inches.
    dpi : int, default 300
        Resolution.
    avg_point_size : int, default 7
        Base font size in points.
    colors : dict or None
        Optional mapping {method: color}.
    y_min : float or None, default None
        If provided, forces the y‑axis lower limit to this value.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    # ── Styling ──────────────────────────────────────────────────────────────────
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = avg_point_size

    # ── Prepare tidy DataFrame ───────────────────────────────────────────────────
    records = [
        {'Method': m, 'Group': g, 'Value': v}
        for m, sub in data_dict.items()
        for g, vals in sub.items()
        for v in vals
    ]
    df = pd.DataFrame(records)

    if sort_order is None:
    	sort_order = sorted(df['Group'].unique(), key=lambda x: (len(x), x), reverse=True)
    else:
    	sort_order = sort_order
    df['Group'] = pd.Categorical(df['Group'], categories=sort_order, ordered=True)

    if colors:
        method_order = list(colors.keys())
        palette = colors
    else:
        method_order = sorted(df['Method'].unique())
        palette = sns.color_palette("Set2")

    df['Method'] = pd.Categorical(df['Method'], categories=method_order, ordered=True)

    # ── Plot ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == 'boxplot':
        sns.boxplot(
            x='Group', y='Value', hue='Method', data=df,
            palette=palette, ax=ax, hue_order=method_order,
            showfliers=False
        )

    elif plot_type == 'barplot':
        grouped = df.groupby(['Method', 'Group'], observed=True)['Value']
        #stats = grouped.agg(['mean', 'std']).reset_index()
        #stats.rename(columns={'mean': 'Value', 'std': 'Value_std'}, inplace=True)
        stats = grouped.agg(['mean', 'std', 'count']).reset_index()
        stats['Value_sem'] = stats['std'] / np.sqrt(stats['count'])
        stats.rename(columns={'mean': 'Value'}, inplace=True)

        stats['Method'] = pd.Categorical(stats['Method'], categories=method_order, ordered=True)
        stats['Group'] = pd.Categorical(stats['Group'], categories=sort_order, ordered=True)

        sns.barplot(
            x='Group', y='Value', hue='Method', data=stats,
            palette=palette, ax=ax, hue_order=method_order, ci=None
        )

        # Centered error bars
        for patch, (_, row) in zip(ax.patches, stats.iterrows()):
            bar_x = patch.get_x() + patch.get_width() / 2
            ax.errorbar(
                bar_x, row['Value'], yerr=row['Value_sem'],
                fmt='none', ecolor='black', elinewidth=1, capsize=3, capthick=1
            )
    else:
        raise ValueError("Unsupported plot_type. Choose 'boxplot' or 'barplot'.")

    # ── Axis labels & ticks ──────────────────────────────────────────────────────
    ax.set_xlabel(x_name or "Number of DEGs", fontsize=avg_point_size)
    ax.set_ylabel(y_name or "R² Mean", fontsize=avg_point_size)

    if y_min is not None:                       # ← APPLY NEW PARAMETER
        ax.set_ylim(bottom=y_min)

    if title:
        ax.set_title(title, fontsize=avg_point_size + 1)

    ax.tick_params(axis='both', labelsize=avg_point_size, labelcolor='black')
    plt.xticks(fontsize=avg_point_size)
    plt.yticks(fontsize=avg_point_size)

    # Legend
    if legend:
        leg = ax.legend(title='Method', title_fontsize=avg_point_size, fontsize=avg_point_size)
        leg.get_title().set_fontweight('bold')
    else:
        ax.legend_.remove()

    # Grid & aesthetics
    if show_grid:
    	ax.grid(show_grid, which='major', axis='y', linestyle='--', linewidth=0.5, color='grey')
    ax.set_axisbelow(True)
    sns.despine(trim=True, offset=10)
    plt.tight_layout()

    # ── Significance stars (scDisentangle vs others) ────────────────────────────
    if stars is not None:
        # stars is expected as stars[method][group] = star_string
        y_min_ax, y_max_ax = ax.get_ylim()
        y_range = y_max_ax - y_min_ax if y_max_ax > y_min_ax else 1.0

        offset_frac = 0.02          # vertical offset above whisker / bar top
        extra_margin_frac = 0.03    # extra headroom so stars are never clipped
        star_fontsize = max(avg_point_size - 2, 6) / 1.6

        star_y_max = y_max_ax

        if plot_type == 'boxplot':
            # --- Compute upper whisker (Q3 + 1.5*IQR, clipped to data) per (Method, Group)
            grouped_vals = df.groupby(['Method', 'Group'], observed=True)['Value']
            whisker_tops = {}

            for (m, g), vals in grouped_vals:
                vals = np.asarray(vals.dropna(), dtype=float)
                if vals.size == 0:
                    continue
                whisker_top = vals.max()             # <- use absolute max
                whisker_tops[(m, g)] = float(whisker_top)

            # --- Compute x-positions analytically (Seaborn default width=0.8, dodge=True)
            n_groups = len(sort_order)
            n_methods = len(method_order)
            total_width = 0.8
            box_width = total_width / n_methods

            for g_idx, group in enumerate(sort_order):
                for m_idx, method in enumerate(method_order):
                    star = stars.get(method, {}).get(group, "")
                    if not star:
                        continue

                    whisker_top = whisker_tops.get((method, group), None)
                    if whisker_top is None:
                        continue

                    # Center of the box for this (group, method)
                    left = g_idx - total_width / 2.0
                    x_center = left + (m_idx + 0.5) * box_width
                    y = whisker_top + offset_frac * y_range

                    ax.text(
                        x_center, y, star,
                        ha='center', va='bottom',
                        fontsize=star_fontsize,
                        color='black'
                    )
                    star_y_max = max(star_y_max, y)

        elif plot_type == 'barplot':
            # Use underlying data maxima per (Method, Group) so stars sit above highest point
            grouped_vals = df.groupby(['Method', 'Group'], observed=True)['Value']
            tops = grouped_vals.max().reset_index().rename(columns={'Value': 'top'})
            top_dict = {(row['Method'], row['Group']): float(row['top']) for _, row in tops.iterrows()}

            for patch, (_, row) in zip(ax.patches, df.groupby(['Method', 'Group'], observed=True).size().reset_index().iterrows()):
                method = row['Method']
                group = row['Group']
                star = stars.get(method, {}).get(group, "")
                if not star:
                    continue

                top = top_dict.get((method, group), None)
                if top is None:
                    continue

                bar_x = patch.get_x() + patch.get_width() / 2.0
                y = top + offset_frac * y_range  # above highest value

                ax.text(
                    bar_x, y, star,
                    ha='center', va='bottom',
                    fontsize=star_fontsize,   # already smaller
                    color='black'
                )
                star_y_max = max(star_y_max, y)

        # Ensure stars are not clipped at the top of the axis
        if star_y_max > y_max_ax:
            ax.set_ylim(top=star_y_max + extra_margin_frac * y_range)
    # Save
    if save_path:
        plt.savefig(f"{save_path}.png", dpi=dpi)
        plt.savefig(f"{save_path}.svg", dpi=dpi)

    plt.show()
    return df
    
def plot_data_refined(
    data_dict,
    plot_type='boxplot',
    y_name=None,
    x_name=None,
    title=None,
    save_path=None,
    legend=True,
    show_grid=False,          # default now matches the other plot (no grid)
    figsize=(7.08, 5),
    dpi=300,
    avg_point_size=12,
    colors=None,
    y_min=None,               # NEW PARAMETER
    sort_order=None,
    stars=None
):
    """
    Generates a boxplot or barplot from a nested dictionary with Nature Biotech styling,
    including error bars (barplot only) and consistent method ordering.

    Parameters
    ----------
    data_dict : dict
        {method: {'group1': [vals], 'group2': [vals], ...}}
    plot_type : {'boxplot', 'barplot'}, default 'boxplot'
    y_name, x_name : str or None
        Axis labels.
    title : str or None
        Plot title (non‑bold, per NB style).
    save_path : str or None
        Path (without extension) to save figure.
    legend : bool, default True
        Whether to display the legend.
    show_grid : bool, default False
        Whether to show horizontal grid lines.
    figsize : tuple, default (7.08, 5)
        Figure size in inches.
    dpi : int, default 300
        Resolution.
    avg_point_size : int, default 12
        Base font size in points.
    colors : dict or None
        Optional mapping {method: color}.
    y_min : float or None, default None
        If provided, forces the y‑axis lower limit to this value.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    # ── Styling ──────────────────────────────────────────────────────────────────
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = avg_point_size

    # ── Prepare tidy DataFrame ───────────────────────────────────────────────────
    records = [
        {'Method': m, 'Group': g, 'Value': v}
        for m, sub in data_dict.items()
        for g, vals in sub.items()
        for v in vals
    ]
    df = pd.DataFrame(records)

    if sort_order is None:
        sort_order = sorted(df['Group'].unique(), key=lambda x: (len(x), x), reverse=True)
    df['Group'] = pd.Categorical(df['Group'], categories=sort_order, ordered=True)

    if colors:
        method_order = list(colors.keys())
        palette = colors
    else:
        method_order = sorted(df['Method'].unique())
        palette = sns.color_palette("Set2")

    df['Method'] = pd.Categorical(df['Method'], categories=method_order, ordered=True)

    # ── Plot ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == 'boxplot':
        sns.boxplot(
            x='Group', y='Value', hue='Method', data=df,
            palette=palette, ax=ax, hue_order=method_order,
            showfliers=False
        )

    elif plot_type == 'barplot':
        grouped = df.groupby(['Method', 'Group'], observed=True)['Value']
        #stats = grouped.agg(['mean', 'std']).reset_index()
        #stats.rename(columns={'mean': 'Value', 'std': 'Value_std'}, inplace=True)
        stats = grouped.agg(['mean', 'std', 'count']).reset_index()
        stats['Value_sem'] = stats['std'] / np.sqrt(stats['count'])
        stats.rename(columns={'mean': 'Value'}, inplace=True)

        stats['Method'] = pd.Categorical(stats['Method'], categories=method_order, ordered=True)
        stats['Group'] = pd.Categorical(stats['Group'], categories=sort_order, ordered=True)

        sns.barplot(
            x='Group', y='Value', hue='Method', data=stats,
            palette=palette, ax=ax, hue_order=method_order, ci=None
        )

        # Centered error bars
        for patch, (_, row) in zip(ax.patches, stats.iterrows()):
            bar_x = patch.get_x() + patch.get_width() / 2
            ax.errorbar(
                bar_x, row['Value'], yerr=row['Value_sem'],
                fmt='none', ecolor='black', elinewidth=1, capsize=3, capthick=1
            )
    else:
        raise ValueError("Unsupported plot_type. Choose 'boxplot' or 'barplot'.")

    # ── Axis labels & ticks ──────────────────────────────────────────────────────
    ax.set_xlabel(x_name or "Number of DEGs",
                  fontsize=avg_point_size + 1,
                  )
    ax.set_ylabel(y_name or "R² Mean",
                  fontsize=avg_point_size + 1,
                  )

    if y_min is not None:
        ax.set_ylim(bottom=y_min)

    if title:
        # Title non‑bold per NB style, but slightly larger
        ax.set_title(title, fontsize=avg_point_size + 2, fontweight='normal')

    # Tick params (bolder, similar to the other plot)
    ax.tick_params(axis='both', which='both',
                   labelsize=avg_point_size,
                   width=0.8, length=4, color='black')
    

    # Legend
    if legend:
        leg = ax.legend(title='Method',
                        title_fontsize=avg_point_size,
                        fontsize=avg_point_size)
        
    else:
        if ax.get_legend() is not None:
            ax.legend_.remove()

    # Grid & aesthetics (match barplot style)
    if show_grid:
        ax.yaxis.grid(True, which='major', linestyle='--',
                      linewidth=0.5, color='grey', alpha=0.6)
        ax.xaxis.grid(False)
    else:
        ax.grid(False)

    ax.set_axisbelow(True)
    sns.despine(trim=True, offset=10)

    plt.tight_layout()

    # ── Significance stars (scDisentangle vs others) ────────────────────────────
    if stars is not None:
        # stars is expected as stars[method][group] = star_string
        y_min_ax, y_max_ax = ax.get_ylim()
        y_range = y_max_ax - y_min_ax if y_max_ax > y_min_ax else 1.0

        offset_frac = 0.02          # vertical offset above whisker / bar top
        extra_margin_frac = 0.03    # extra headroom so stars are never clipped
        star_fontsize = max(avg_point_size - 2, 6) / 1.6

        star_y_max = y_max_ax

        if plot_type == 'boxplot':
            # --- Compute upper whisker (here: use absolute max) per (Method, Group)
            grouped_vals = df.groupby(['Method', 'Group'], observed=True)['Value']
            whisker_tops = {}

            for (m, g), vals in grouped_vals:
                vals = np.asarray(vals.dropna(), dtype=float)
                if vals.size == 0:
                    continue
                whisker_top = vals.max()
                whisker_tops[(m, g)] = float(whisker_top)

            # --- Compute x-positions analytically (Seaborn default width=0.8, dodge=True)
            n_groups = len(sort_order)
            n_methods = len(method_order)
            total_width = 0.8
            box_width = total_width / n_methods

            for g_idx, group in enumerate(sort_order):
                for m_idx, method in enumerate(method_order):
                    star = stars.get(method, {}).get(group, "")
                    if not star:
                        continue

                    whisker_top = whisker_tops.get((method, group), None)
                    if whisker_top is None:
                        continue

                    # Center of the box for this (group, method)
                    left = g_idx - total_width / 2.0
                    x_center = left + (m_idx + 0.5) * box_width
                    y = whisker_top + offset_frac * y_range

                    ax.text(
                        x_center, y, star,
                        ha='center', va='bottom',
                        fontsize=star_fontsize,
                        color='black'
                    )
                    star_y_max = max(star_y_max, y)

        elif plot_type == 'barplot':
            # Use underlying data maxima per (Method, Group) so stars sit above highest point
            grouped_vals = df.groupby(['Method', 'Group'], observed=True)['Value']
            tops = grouped_vals.max().reset_index().rename(columns={'Value': 'top'})
            top_dict = {(row['Method'], row['Group']): float(row['top']) for _, row in tops.iterrows()}

            for patch, (_, row) in zip(
                ax.patches,
                df.groupby(['Method', 'Group'], observed=True).size().reset_index().iterrows()
            ):
                method = row['Method']
                group = row['Group']
                star = stars.get(method, {}).get(group, "")
                if not star:
                    continue

                top = top_dict.get((method, group), None)
                if top is None:
                    continue

                bar_x = patch.get_x() + patch.get_width() / 2.0
                y = top + offset_frac * y_range  # above highest value

                ax.text(
                    bar_x, y, star,
                    ha='center', va='bottom',
                    fontsize=star_fontsize,
                    color='black'
                )
                star_y_max = max(star_y_max, y)

        # Ensure stars are not clipped at the top of the axis
        if star_y_max > y_max_ax:
            ax.set_ylim(top=star_y_max + extra_margin_frac * y_range)

    # Save
    if save_path:
        plt.savefig(f"{save_path}.png", dpi=dpi, bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", dpi=dpi, bbox_inches='tight')

    plt.show()
    return df
