import numpy as np
import matplotlib.pyplot as plt

from data_manager import heatmaps, check_paths, get_secondary_data


def create_histograms(matrix_stack):
    """
    Creats a histogram to visualize the distribution of predicted FC between
    some subset of regions. To choose the regions for which yousee this
    distribution, edit regions_i, and regions_j. Regions should be selected
    that have non-zero predicted FC.
    """
    # Creates a figure and axis object.
    fig, axs = plt.subplots(8, 8, figsize=(20, 20))

    # Add some extra padding to the top
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Iterates over each region and adds a histogram to each the corrosponding subplot.
    regions_i = [1, 7, 16, 77, 89, 114, 46, 34] # A selection of regions
    regions_j = [55, 4, 99, 3, 65, 45, 88, 24] # A selection of regions
    for i in range(len(regions_i)):
        for j in range(len(regions_j)):
            region_stack = matrix_stack[:, i, j]
            ax = axs[i, j]
            ax.hist(region_stack, bins=15, alpha=0.5)

    # Add column names
    for i, name in enumerate(regions_j):
        axs[0, i].set_title(str(name))  # Set title for top row of histograms
    # Add row names
    for i, name in enumerate(regions_j):
        axs[i, 0].set_ylabel(str(name), rotation=90, ha="center")  # Set ylabel for first column
    return fig

def keep_indices(matrix, indices_to_keep):
    """
    Takes in a matrix, and returns a new matrix, that keeps only the entries
    involving the indices in indices_to_keep.
    """
    kept_matrix= np.zeros(np.shape(matrix))
    for indx in indices_to_keep:
        kept_matrix[indx, :] = matrix[indx, :]
        kept_matrix[:, indx] = matrix[:, indx]
    return kept_matrix

def resection_output(predicted_fc_stack, path, resected_rules=None, histograms=False):
    """ Outputs histograms for pred FC and mean predicted FC, and optionally
    rules distributions and resected_rules or resected X."""
    # If histograms is true, output predB histograms.
    if histograms:
        create_histograms(predicted_fc_stack)
        plt.savefig(path + 'predB_distributions', dpi=300)
        plt.close()

    # Outputs mean predicted FC heatmap
    mean_predicted_fc = np.mean(predicted_fc_stack, axis=0)
    heatmaps(mean_predicted_fc, path + 'mean_predB', 'predfc')
    np.savetxt(path + 'mean_predB', mean_predicted_fc)

    # If resected_rules is provided, output the heatmap
    if resected_rules:
        # Outputs resected rules heatmap
        heatmaps(resected_rules, path + 'resected_rules', 'rules')

def group_level_resections(results_dir, out_dir, regions_dict, resect_rules=False):
    """
    Computes group level resections. By default, resections are done on SC,
    but if resect_rules is set to True, resections will be carried out on 
    the rule set.
    """
    # Instantiate the ids and read in the group rules
    subject_ids = [num + 1 for num in range(50)]
    rules = get_secondary_data(results_dir + 'fitted_O/fitted_ONone')

    # Iterates over each set of ranges
    for (region, indxs) in regions_dict.items():
        # Define the output path and create it if it doesn't already exist
        out_path = out_dir + f'{region}/'
        check_paths(out_path)

        # If resect_rules is True, make rule resections
        if resect_rules:
            resected_rules = keep_indices(rules, indxs)
        
        # List of predicted FC matrices that will be stacked into one 3d array later
        predicted_fc_to_stack = []

        # Iterate over each subject and add the resected rules and new prediction to the stacks.
        for subject_id in subject_ids:
            sc = get_secondary_data(results_dir + f'X/X{subject_id}')

            # If resect_rules is False, make SC resections, and make predictions for FC. Else use resected rules.
            if not resect_rules:
                resected_sc = keep_indices(sc, indxs)
                predicted_fc = resected_sc @ rules @ resected_sc
            else:
                predicted_fc = sc @ resected_rules @ sc

            # Append the predicted FC to the list.
            predicted_fc_to_stack.append(predicted_fc)

        # Create the 3d array and output a subset of histograms, mean predB, resected rules, resected X, depending on settings.
        predicted_fc_stack = np.stack(predicted_fc_to_stack)

        if resect_rules:
            resection_output(predicted_fc_stack, out_path, resected_rules=rules)
        else:
            resection_output(predicted_fc_stack, out_path)

def subject_level_predictions(results_dir, out_dir, regions_dict, resect_rules=False):
    """
    Computes subject level resections. By default, resections are done on SC,
    but if resect_rules is set to True, resections will be carried out on 
    the rule set.
    """
    # Iterate over each subjects' individually fit rule set.
    subject_ids = [1, 2, 3, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29]

    # Iterate over each of the regions.
    for (region, indxs) in regions_dict.items():
        # List of predicted FC matrices that will be stacked into one 3d array later
        predicted_fc_to_stack = []

        # iterate over each subject's id
        for subject_id in subject_ids:
            # Read in the SC and rules matrix for the given subject id    
            sc = get_secondary_data(results_dir + f'X/X{subject_id}')
            rules = get_secondary_data(results_dir + f'fitted_O/fitted_O{subject_id}')

            # Make sure output path exists.
            out_path = out_dir + f'{region}/'
            check_paths(out_path)

            # If resected_rules is True, resect the rules. Else resect X. Predict FC.
            if resect_rules:
                resected_rules = keep_indices(rules, indxs)
                predicted_fc = sc @ resected_rules @ sc
            else:
                resected_sc = keep_indices(sc, indxs)
                predicted_fc = resected_sc @ rules @ resected_sc
            
            # Add predicted B to the list
            predicted_fc_to_stack.append(predicted_fc)
        
        # Create the 3d array and output a subset of histograms, mean predB, resected rules, resected X, depending on settings.
        predicted_fc_stack = np.stack(predicted_fc_to_stack)

        if resect_rules:
            resection_output(predicted_fc_stack, out_path)
        else:
            resection_output(predicted_fc_stack, out_path)

def main():
    """
    Performs the resections, and outputs the results. 
    """
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
        'DMN': list(range(53, 65)) + list(range(106, 116))
    }

    # These paths need to be set!!
    results_dir = 'pre_resection/matrices/'
    outdir = 'sl_resects/'

    subject_level_predictions(results_dir, outdir, regions, resect_rules=True)
    #group_level_resections(RESULTS_DIR, OUT_DIR, regions)

if __name__ == "__main__":
    main()