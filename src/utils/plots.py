import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.ticker as mtick
import seaborn as sns
from utils.subtypes import process_value
import warnings

def visualize_motion_fc_characteristics(clinical_data, significant_clinical_variables, label='Our subtype'):
    labels = clinical_data[label]
    visualize_data = np.zeros((len(significant_clinical_variables), 3))
    # prepare data for visualization
    for clinical_i, clinical_variable in enumerate(significant_clinical_variables):
        clinical_variable_data = clinical_data[clinical_variable]
        # clean data
        clinical_variable_data = np.array([process_value(value) for value in clinical_variable_data])
        keep_indices = np.logical_not(np.logical_or(np.isinf(clinical_variable_data), np.isnan(clinical_variable_data)))
        clinical_variable_data_filtered = list(clinical_variable_data[keep_indices])
        labels_filtered = list(np.array(labels)[keep_indices])

        # find mean values based on subtypes
        clinical_variable_mean = np.mean(clinical_variable_data_filtered)
        subtype_data = {0: [], 1: [], 2: []}
        for i, subtype in enumerate(labels_filtered):
            column_value = clinical_variable_data_filtered[i]
            subtype_data[subtype].append(column_value)
        subtype_1_data = np.array(subtype_data[0])
        subtype_2_data = np.array(subtype_data[1])
        subtype_3_data = np.array(subtype_data[2])
        
        # normalize
        visualize_data[clinical_i][0] = (np.mean(subtype_1_data) - clinical_variable_mean) / np.std(clinical_variable_data_filtered)
        visualize_data[clinical_i][1] = (np.mean(subtype_2_data) - clinical_variable_mean) / np.std(clinical_variable_data_filtered)
        visualize_data[clinical_i][2] = (np.mean(subtype_3_data) - clinical_variable_mean) / np.std(clinical_variable_data_filtered)
    
    # create plot showing variable means per subtype
    xticklabels = ['Subtype I', 'Subtype II', 'Subtype III']
    heatmap_palette = ["#66c2a5", "#fc8d62", "#8da0cb"]
    heatmap_height=12
    fig, ax = plt.subplots(figsize=(22,heatmap_height))
    sns.heatmap(visualize_data, cmap=sns.diverging_palette(17, 240, as_cmap=True, sep=20), yticklabels=significant_clinical_variables,
            xticklabels=xticklabels, ax=ax, vmin=-max(abs(visualize_data.min()), abs(visualize_data.max())), 
            vmax=max(abs(visualize_data.min()), abs(visualize_data.max())), cbar_kws={"shrink": 0.4, 'anchor':(0, 0), 'aspect': 10,})

    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=22)
    ax.collections[0].colorbar.ax.tick_params(labelsize=16)
    for label, color in zip(ax.get_xticklabels(), heatmap_palette):
        label.set_color(color)
        label.set_weight('bold')
        label.set_alpha(1)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=18)
    
    return fig

def visualize_motion_fc_characteristics_boxplot(clinical_data, significant_clinical_variables, significant_clinical_variables_p_values):

    def p_value_formatter(p_value):
        if p_value < 0.001:
            coefficient, exponent = "{:.2e}".format(p_value).split('e')
            return f'{coefficient} x $10^{{{int(exponent)}}}$'
        else:
            return "{:.3f}".format(p_value)

    def subtype_rename(group_numbers):
        rename_map = {0: 'I', 1: 'II', 2: 'III'}
        return [rename_map[num] for num in group_numbers]

    num_rows=6
    num_cols=4
    subtype_colorpalette = ["#66c2a5", "#fc8d62", "#8da0cb"]
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20,28))
    warnings.filterwarnings("ignore", message=".*You passed a edgecolor.*") # ignore this warning

    for clinical_i, clinical_variable in enumerate(significant_clinical_variables):
        row = clinical_i // num_cols
        col = clinical_i % num_cols
        clinical_variable_data = clinical_data.loc[:, [clinical_variable, 'Our subtype']]
        clinical_variable_data['Subtype'] = subtype_rename(clinical_data['Our subtype'])

        axes[row][col].set_title(f'$p = ${p_value_formatter(significant_clinical_variables_p_values[clinical_i])}', fontsize=16)

        processed_data = np.array([process_value(value) for value in clinical_variable_data[clinical_variable]])
        keep_indices = np.logical_not(np.logical_or(np.isinf(processed_data), np.isnan(processed_data)))
        
        clinical_variable_data_filtered = clinical_variable_data[keep_indices]
        
        sns.boxplot(data=clinical_variable_data_filtered, x='Subtype', y=clinical_variable, order=['I', 'II', 'III'], ax=axes[row][col], palette=subtype_colorpalette, saturation=1, linewidth=3.5, showcaps=False,medianprops={'linewidth': 6, 'solid_capstyle': 'butt', 'zorder': 1}, zorder=1)

        sns.pointplot(data=clinical_variable_data_filtered, x='Subtype', y=clinical_variable, order=['I', 'II', 'III'], ax=axes[row][col], color='gold', ci=None, markers="x", linestyles='dashed', scale=1.3, zorder=2)

        axes[row][col].xaxis.label.set_size(18)
        axes[row][col].tick_params(axis='x', labelsize=16)
        axes[row][col].yaxis.label.set_size(18)
        axes[row][col].tick_params(axis='y', labelsize=14)

        sns.despine(ax=axes[row][col])

    fig.delaxes(axes[num_rows-1][num_cols-1])
    plt.tight_layout(pad=6.0)

    #warnings.resetwarnings()

    return fig

def create_per_patient_plot_categorical(x, y, hue, data, person_bounds, hue_order, palette, alpha=0.9, n_std=1.0):
    ax = sns.scatterplot(x=x, y=y, hue=hue, data=data, alpha=alpha, hue_order=hue_order, palette=palette,s=30, edgecolor='black', linewidth=1)
    for start, end in person_bounds:
        if end - start == 1: continue
        x_sub = data[x].to_list()[start:end]
        y_sub = data[y].to_list()[start:end]
        cov = np.cov(x_sub, y_sub)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipsecolor = palette[hue_order.index(data[hue].to_list()[start])]
        ellipse = Ellipse((0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2, 
            alpha=0.8,
            facecolor=ellipsecolor,
            edgecolor='black')
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x_sub)

        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y_sub)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        
        ellipse.set_transform(transf + ax.transData)

        ax.add_patch(ellipse)
    return ax

def create_per_patient_plot_numerical(x, y, hue, data, person_bounds, alpha=0.9, n_std=1.0, palette='viridis'):
    ax = sns.scatterplot(x=x, y=y, hue=hue, data=data, alpha=alpha, palette=palette, s=30, edgecolor='black', linewidth=1)
    norm = plt.Normalize(min(data[hue]), max(data[hue]))
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    for start, end in person_bounds:
        x_sub = data[x].to_list()[start:end]
        y_sub = data[y].to_list()[start:end]
        cov = np.cov(x_sub, y_sub)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipsecolor = sm.to_rgba(sum(data[hue].to_list()[start:end]) / len(data[hue].to_list()[start:end]))
        ellipse = Ellipse((0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2, 
            alpha=.8,
            facecolor=ellipsecolor,
            edgecolor='black'
            )
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x_sub)

        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y_sub)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        
        ellipse.set_transform(transf + ax.transData)

        ax.add_patch(ellipse)

    sm.set_array([])

    # Remove the legend and add a colorbar
    ax.get_legend().remove()
    ax.figure.colorbar(sm, label=hue)
    return ax

def create_hist_random_wcss_scores(random_groups_wcss_scores, our_approach_score_wcss, conventional_approach_score_wcss, td_pigd_approach_score_wcss, width_ratio=6):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 4), gridspec_kw={'width_ratios': [1, width_ratio]})
    
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True

    td_pigd_color = '#bebada'
    clinical_subtype_color = '#1f78b4'
    our_subtype_color = '#fc8d62'

    sns.histplot(random_groups_wcss_scores, ax=ax2, stat='probability', alpha=1, color='#8C8C8C', edgecolor='black')
    # Our subtyping score much lower wcss score, two axs needed and break plot
    ax1.axvline(our_approach_score_wcss, 0, 1, linewidth=3, color=our_subtype_color, linestyle='--')
    ax2.axvline(conventional_approach_score_wcss, 0, 1, linewidth=3, color=clinical_subtype_color, linestyle='--')
    ax2.axvline(td_pigd_approach_score_wcss, 0, 1, linewidth=3, color=td_pigd_color, linestyle='--')

    ax1.text(our_approach_score_wcss + .1, .045, 'Our\nsubtyping', horizontalalignment='left', color=our_subtype_color, fontsize=15, alpha=1, weight='bold')
    ax2.text(conventional_approach_score_wcss + .1, .045, 'Clinical variable\nbased subtyping', horizontalalignment='left', color=clinical_subtype_color, fontsize=15, alpha=1, weight='bold')
    ax2.text(td_pigd_approach_score_wcss - .1, .045, 'TD/PIGD\nsubtyping', horizontalalignment='right', color=td_pigd_color, fontsize=15, alpha=1, weight='bold')
    ax1.set_ylim(0, 0.06)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='y', length=0)
    ax1.set_ylabel('% of random subgroups', fontsize=20)
    fig.text(0.5, -0.04, '$\sigma$ score', ha='center', fontsize=20)

    # make range of axs the same
    min_x, max_x = ax2.get_xlim()
    range_padding = (max_x - min_x) / 5 / 2
    ax1.set_xlim(our_approach_score_wcss - range_padding, our_approach_score_wcss + range_padding)

    sns.despine()
    # hide the spines between ax and ax2
    ax2.spines['left'].set_visible(False)

    # add diagonal lines signifying a break in the x-axis
    d = .03  # how big to make the diagonal lines breaking the x axis
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d/width_ratio, +d/width_ratio), (-d, +d), **kwargs)

    fig.tight_layout()
    return fig