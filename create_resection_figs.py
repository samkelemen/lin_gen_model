import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

from data_manager import get_secondary_data


class CreateResectionFigs:
    def __init__(self, path):
        self.path = path

    def get_mean_predicted_fc(self):
        """
        Gets the mean predicted FC matrix for a resection. In the path definition,
        'mean_predB' may need to be changed to 'predB'.
        """
        path = self.path + 'mean_predB'
        return get_secondary_data(path)

    def plot_matrix(self, matrix, title, output_path, labels, diag_info=None):
        """
        Plots given resection matrix.
        """
        if not diag_info:
            cmap='vlag'
            ax = sns.heatmap(matrix, center=0, cmap="vlag", square=True, xticklabels=labels, yticklabels=labels, linewidth=0.5, linecolor='black')
        elif diag_info == 'neg':
            cmap = mcolors.LinearSegmentedColormap.from_list("", ["#2F4F7F", "white"])
            ax = sns.heatmap(matrix, vmin=-1, cmap=cmap, square=True, yticklabels=labels, linewidth=0.5, linecolor='black')
            plt.xticks(ticks=[], labels=[])
        elif diag_info == 'pos':
            cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "#8B0A1A"])
            ax = sns.heatmap(matrix, vmin=0, cmap=cmap, square=True, yticklabels=labels, linewidth=0.5, linecolor='black')
            plt.xticks(ticks=[], labels=[])

        ax.set_title(title)
        plt.savefig(output_path, bbox_inches='tight', dpi=500)
        plt.close()

    def make_avg_plots(self, name_indx, regions, set_diag_to_zero=False):
        """
        Creates plots showing the average predicted FC for a specific region
        after virtual resection.
        """
    # Instantiate the output paths.
        output_path = self.path + 'fig1'
        mean_predicted_fc= self.get_mean_predicted_fc()
        
        # Instantiate this list to declare the order in which to iterate over the regions.
        labels = ['Thalamus', 'Caudate', 'Putamen', 'Pallidum', 'Accumbens', 'Amygdala',\
                'Hypocampus','Sommatosensory', 'Visual Cortex', 'DAN', 'SAN', 'Limbic', 'Cont', \
                'DMN']

        # Create zero matrices to then edit.
        num_regions = len(labels)
        avg_matrix = np.zeros(shape=(num_regions, num_regions))

        # Iterate over the indices of the subregions in region1 and region2
        for (i, region1) in enumerate(labels):
            for (j, region2) in enumerate(labels):
                # For both regions in the pair, define the indices in that region
                region1_indices = regions[region1]
                region2_indices = regions[region2]

                # Define the total number of values that will be counted.
                num_indices = len(region1_indices) * len(region2_indices)
                
                # Set the sum to 0 to start.
                region_sum = 0

                # For each pair of region indices, add the predicted B value to the mean for the regions.
                for index1 in region1_indices:
                    for index2 in region2_indices:
                        # If we don't want to include diagonals, here we specify not to add them to the sum
                        if set_diag_to_zero and index1 == index2:
                            num_indices -= 1
                        # Otherwise, add the values to the sum
                        else:
                            region_sum += mean_predicted_fc[index1][index2]

                # Calculate the mean
                mean = region_sum / num_indices

                # Set the (i, j)th value of the avg_matrix to the mean
                avg_matrix[i][j] = mean

        # Create the average matrix plot
        self.plot_matrix(avg_matrix, labels[name_indx], output_path, labels )

        # Diagonal values matrix.
        diags = np.diagonal(avg_matrix).reshape((num_regions, 1))
        self.plot_matrix(diags, f'{labels[name_indx]} Diagonal', output_path + "diag", labels)

        # Matrix with diagonal values = 0.
        np.fill_diagonal(avg_matrix, 0)
        self.plot_matrix(avg_matrix, f'{labels[name_indx]}, Diag = 0', output_path + "0", labels)

    def make_signed_plots(self, regions, name_indx, set_diag_to_zero=False):
        """
        Creates resection plots showing only the negative or positive predicted FC for
        each region/ system.
        """
        # Instantiate the output paths.
        output_path = self.path + 'fig1'
        mean_predicted_fc = self.get_mean_predicted_fc()
        
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
                # For both regions in the pair, define the indices in that region
                region1_indices = regions[region1]
                region2_indices = regions[region2]

                # Define the total number of values that will be counted.
                num_indices = len(region1_indices) * len(region2_indices)

                # Instantiate the sums at 0 to start
                pos_sum = 0
                neg_sum = 0

                # Iterate over the indices of the subregions in region1 and region2
                for index1 in region1_indices:
                    for index2 in region2_indices:
                        # If we don't want to include diagonals, here we specify not to add them to the sums
                        if set_diag_to_zero and index1 == index2:
                            num_indices -= 1
                        # Otherwise, add the negative, positive values to the negative, positive sums
                        else:
                            # Add positive values to positive sum
                            if mean_predicted_fc[index1][index2] > 0:
                                pos_sum += mean_predicted_fc[index1][index2]
                            # Add negative values to negative sum
                            elif mean_predicted_fc[index1][index2] < 0:
                                neg_sum += mean_predicted_fc[index1][index2]

                # Calculate the mean of positive and negative values
                pos_mean = pos_sum / num_indices
                neg_mean = neg_sum / num_indices

                # Set the (i, j)th element of the matrices to the means
                pos_avg_matrix[i][j] = pos_mean
                neg_avg_matrix[i][j] = neg_mean

        # Output the plots.
        # Positive matrix
        self.plot_matrix(pos_avg_matrix, f'{labels[name_indx]} [Positive]', output_path + "pos", labels)
        
        # Positive diagonal values matrix
        pos_diags = np.diagonal(pos_avg_matrix).reshape((num_regions, 1))
        self.plot_matrix(pos_diags, f'{labels[name_indx]} Diagonal [Positive]', output_path + "pos_diag", labels, diag_info='pos')
        
        # Positive matrix with diagonal values = 0
        np.fill_diagonal(pos_avg_matrix, 0)
        self.plot_matrix(pos_avg_matrix, f'{labels[name_indx]} [Positive], Diag = 0', output_path + "pos0", labels)
        
        # Negative matrix
        self.plot_matrix(neg_avg_matrix, f'{labels[name_indx]} [Negative]', output_path + "neg", labels)
        
        # Negative diagonal values matrix
        neg_diags = np.diagonal(neg_avg_matrix).reshape((num_regions, 1))
        self.plot_matrix(neg_diags, f'{labels[name_indx]} Diagonal [Negative]', output_path + "neg_diag", labels, diag_info='neg')
        
        # Negative matrix with diagonal values = 0
        np.fill_diagonal(neg_avg_matrix, 0)
        self.plot_matrix(neg_avg_matrix, f'{labels[name_indx]} [Negative], Diag = 0', output_path + "neg0", labels)


def main():
    """
    PATH should be set as the resections folder you are creating figures for. The figures
    are outputted to the resections folder.
    """
    regions = ['Thalamus', 'Caudate', 'Putamen', 'Pallidum', 'Accumbens', 'Amygdala',\
            'Hypocampus','Sommatosensory', 'Visual Cortex', 'DAN', 'SAN', 'Limbic', 'Cont', \
            'DMN']
    
    regions_dict = {'Thalamus': (0, 7), 
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
    
    # For each region, make the signed and average plots
    for (name_indx, region_name) in enumerate(regions):
        # Define the subject level output paths
        path = f"subject_level_results/standard_SC/sl_X_resects_standard/{region_name}/"
        path_for_zero_diag = f"subject_level_results/standard_SC/sl_X_resects_standard_diag0/{region_name}/"
        
        sl_resector = CreateResectionFigs(path)
        sl_diag_0_resector = CreateResectionFigs(path_for_zero_diag)

        # Create the plots keeping the diagonal values
        sl_resector.make_signed_plots(regions_dict, name_indx)
        sl_resector.make_avg_plots(name_indx, regions_dict, path)

        # Create the plots with diagonal values set to 0
        sl_diag_0_resector.make_signed_plots(regions_dict, name_indx, set_diag_to_zero=True)
        sl_diag_0_resector.make_avg_plots(name_indx, regions_dict, set_diag_to_zero=True)
        """
        # Define group level output paths
        path = f"group_results/standard_SC/X_resects_standard/{region_name}/"
        path_for_zero_diag = f"group_results/standard_SC/X_resects_standard_diag0/{region_name}/"
        
        gl_resector = CreateResectionFigs(path)
        gl_diag_0_resector = CreateResectionFigs(path_for_zero_diag)

        # Create the plots keeping the diagonal values
        gl_resector.make_signed_plots(regions_dict, name_indx)
        gl_resector.make_avg_plots(name_indx, regions_dict)

        # Create the plots with diagonal values set to 0
        gl_diag_0_resector.make_signed_plots(regions_dict, name_indx, set_diag_to_zero=True)
        gl_diag_0_resector.make_avg_plots(name_indx, regions_dict, set_diag_to_zero=True)
        """
if __name__ == '__main__':
    main()