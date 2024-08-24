import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lin_gen_model import get_secondary_data

def get_mean_predB(path):
    """
    Gets the mean predicted FC for a resection. In the path definition,
    'mean_predB' may need to be changed to 'predB'.
    """
    path = path + 'mean_predB'
    return get_secondary_data(path)

def make_avg_plots(name_indx, path):
    """
    Creates plots showing the average predicted FC for a specific region
    after virtual resection.
    """
   # Instantiate the output paths.
    out_path = path + 'fig1'
    mean_predB = get_mean_predB(path)

    #Declare the reiong indices.
    regions = {'Thalamus': (0, 7), 
            'Caudate': (1, 8), 
            'Putamen': (2, 9),
            'Pallidum': (3, 10),
            'Accumbens': (6, 13), 
            'Amygdala': (5, 12), 
            'Hypocampus': (4, 11),
            'Sommatosensory':list(range(25, 30)) + list(range(75, 82)), 
            'Visual Cortex': list(range(16, 24)) + list(range(67, 74)), 
            'DAN': list(range(31, 38)) + list(range(83, 89)), 
            'SAN': list(range(39, 45)) + list(range(90, 94)), 
            'Limbic': list(range(46, 48)) + list(range(95, 96)), 
            'Cont': list(range(49, 52)) + list(range(97, 105)), 
            'DMN': list(range(53, 65)) + list(range(106, 116))}
    
    # Instantiate this list to declare the order in which to iterate over the regions.
    labels = ['Thalamus', 'Caudate', 'Putamen', 'Pallidum', 'Accumbens', 'Amygdala',\
            'Hypocampus','Sommatosensory', 'Visual Cortex', 'DAN', 'SAN', 'Limbic', 'Cont', \
            'DMN']

     # Create zero matrices to then edit.
    num_regions = len(labels)
    avg_matrix = np.zeros(shape=(num_regions, num_regions))

    # For each region, take the average and add it to the mean matrix.
    for (i, region1) in enumerate(labels):
        for (j, region2) in enumerate(labels):
            region1_indices = regions[region1]
            region2_indices = regions[region2]
            num_indices = len(region1_indices) * len(region2_indices)
            
            mean = 0
            for index1 in region1_indices:
                for index2 in region2_indices:
                    mean += mean_predB[index1][index2]

            mean = mean / num_indices
            avg_matrix[i][j] = mean / 2 # divide by two because the algorithm double counts
    
    # Create the plot.
    sns.color_palette("vlag", n_colors=8, as_cmap=True)
    ax = sns.heatmap(avg_matrix, center=0, cmap="vlag", square=True, \
                xticklabels=labels, yticklabels=labels, linewidth=0.5, linecolor='black')
    ax.set_title(labels[name_indx])
    plt.savefig(out_path, bbox_inches='tight', dpi=500)
    plt.close()

    # Diagonal values matrix.
    diags = np.diagonal(avg_matrix).reshape((num_regions, 1))
    sns.color_palette("vlag", n_colors=8, as_cmap=True)
    ax = sns.heatmap(diags, center=0, cmap="vlag", square=False, \
                xticklabels=labels, yticklabels=labels, linewidth=0.5, linecolor='black')
    ax.set_title(f'{labels[name_indx]} Diagonal')
    plt.savefig(out_path + "diag", bbox_inches='tight', dpi=500)
    plt.close()

    #Positive  matrix with diagonal values = 0.
    np.fill_diagonal(avg_matrix, 0)
    sns.color_palette("vlag", n_colors=8, as_cmap=True)
    ax = sns.heatmap(avg_matrix, center=0, cmap="vlag", square=True, \
                xticklabels=labels, yticklabels=labels, linewidth=0.5, linecolor='black')
    ax.set_title(f'{labels[name_indx]}, Diag = 0')
    plt.savefig(out_path + "0", bbox_inches='tight', dpi=500)
    plt.close()

def make_signed_plots(region_name, name_indx):
    """
    Creates resection plots showing only the negative or positive predicted FC for
    each region/ system.
    """
    # Instantiate the output paths.
    out_path = f'group_level/log10_SC/virt_resects_log10_SC/{region_name}/fig1'
    mean_predB = get_mean_predB(region_name)

    #Declare the reiong indices.
    regions = {'Thalamus': (0, 7), 
            'Caudate': (1, 8), 
            'Putamen': (2, 9), 
            'Pallidum': (3, 10),
            'Accumbens': (6, 13), 
            'Amygdala': (5, 12), 
            'Hypocampus': (4, 11),
            'Sommatosensory':list(range(25, 30)) + list(range(75, 82)), 
            'Visual Cortex': list(range(16, 24)) + list(range(67, 74)), 
            'DAN': list(range(31, 38)) + list(range(83, 89)), 
            'SAN': list(range(39, 45)) + list(range(90, 94)), 
            'Limbic': list(range(46, 48)) + list(range(95, 96)), 
            'Cont': list(range(49, 52)) + list(range(97, 105)), 
            'DMN': list(range(53, 65)) + list(range(106, 116))}
    
    # Instantiate this list to declare the order in which to iterate over the regions.
    labels = ['Thalamus', 'Caudate', 'Putamen', 'Pallidum', 'Accumbens', 'Amygdala',\
            'Hypocampus','Sommatosensory', 'Visual Cortex', 'DAN', 'SAN', 'Limbic', 'Cont', \
            'DMN']
    
    # Create zero matrices to then edit.
    num_regions = len(labels)
    pos_avg_matrix = np.zeros(shape=(num_regions, num_regions))
    neg_avg_matrix = np.zeros(shape=(num_regions, num_regions))

    # For each region, take the average of positives and negatives and add those values to the corresponding mean matrices.
    for (i, region1) in enumerate(labels):
        for (j, region2) in enumerate(labels):
            region1_indices = regions[region1]
            region2_indices = regions[region2]
            num_indices = len(region1_indices) * len(region2_indices)

            pos_mean = 0
            neg_mean = 0
            for index1 in region1_indices:
                for index2 in region2_indices:
                    if mean_predB[index1][index2] > 0:
                        pos_mean += mean_predB[index1][index2]
                    elif mean_predB[index1][index2] < 0:
                        neg_mean += mean_predB[index1][index2]

            pos_mean = pos_mean / num_indices
            neg_mean = neg_mean / num_indices
            pos_avg_matrix[i][j] = pos_mean / 2
            neg_avg_matrix[i][j] = neg_mean / 2
    
    # Create the plots.
    # Positive matrix.
    sns.color_palette("vlag", n_colors=8, as_cmap=True)
    ax = sns.heatmap(pos_avg_matrix,center=0, cmap="vlag", square=True, \
                xticklabels=labels, yticklabels=labels, linewidth=0.5, linecolor='black')
    ax.set_title(f'{labels[name_indx]} [Positive]')
    plt.savefig(out_path + "pos", bbox_inches='tight', dpi=500)
    plt.close()

    # Positive diagonal values matrix.
    pos_diags = np.diagonal(pos_avg_matrix).reshape((num_regions, 1))
    sns.color_palette("vlag", n_colors=8, as_cmap=True)
    ax = sns.heatmap(pos_diags,center=0, cmap="vlag", square=False, \
                xticklabels=labels, yticklabels=labels, linewidth=0.5, linecolor='black')
    ax.set_title(f'{labels[name_indx]} Diagonal [Positive]')
    plt.savefig(out_path + "pos_diag", bbox_inches='tight', dpi=500)
    plt.close()

    #Positive  matrix with diagonal values = 0.
    np.fill_diagonal(pos_avg_matrix, 0)
    sns.color_palette("vlag", n_colors=8, as_cmap=True)
    ax = sns.heatmap(pos_avg_matrix, center=0, cmap="vlag", square=True, \
                xticklabels=labels, yticklabels=labels, linewidth=0.5, linecolor='black')
    ax.set_title(f'{labels[name_indx]} [Positive], Diag = 0')
    plt.savefig(out_path + "pos0", bbox_inches='tight', dpi=500)
    plt.close()

    # Negative matrix.
    sns.color_palette("vlag", n_colors=8, as_cmap=True)
    ax = sns.heatmap(neg_avg_matrix,center=0, cmap="vlag", square=True, \
                xticklabels=labels, yticklabels=labels, linewidth=0.5, linecolor='black')
    ax.set_title(f'{labels[name_indx]} [Negative]')
    plt.savefig(out_path + "neg", bbox_inches='tight', dpi=500)
    plt.close()

    # Negative diagonal values matrix.
    neg_diags = np.diagonal(neg_avg_matrix).reshape((num_regions, 1))
    sns.color_palette("vlag", n_colors=8, as_cmap=True)
    ax = sns.heatmap(neg_diags, center=0, cmap="vlag", square=False, \
                xticklabels=labels, yticklabels=labels, linewidth=0.5, linecolor='black')
    ax.set_title(f'{labels[name_indx]} Diagonal [Negative]')
    plt.savefig(out_path + "neg_diag", bbox_inches='tight', dpi=500)
    plt.close()

    # Negative matrix with diagaonl values = 0.
    np.fill_diagonal(neg_avg_matrix, 0)
    sns.color_palette("vlag", n_colors=8, as_cmap=True)
    ax = sns.heatmap(neg_avg_matrix,center=0, cmap="vlag", square=True, \
                xticklabels=labels, yticklabels=labels, linewidth=0.5, linecolor='black')
    ax.set_title(f'{labels[name_indx]} [Negative], Diag = 0')
    plt.savefig(out_path + "neg0", bbox_inches='tight', dpi=500)
    plt.close()

def main():
    """
    PATH should be set as the resections folder you are creating figures for. The figures
    are outputted to the resections folder.
    """
    regions = ['Thalamus', 'Caudate', 'Putamen', 'Pallidum', 'Accumbens', 'Amygdala',\
            'Hypocampus','Sommatosensory', 'Visual Cortex', 'DAN', 'SAN', 'Limbic', 'Cont', \
            'DMN']
    
    # For each region, make the signed and average plots
    for (name_indx, region) in enumerate(regions):
        PATH = f"subject_level/virt_resects_log10_SC/{region}/"
        make_signed_plots(region, name_indx)
        make_avg_plots(name_indx, PATH)

main()