import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path


def check_paths(paths):
    """Checks if the paths exist, and if not, creates them."""
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                Path(path).mkdir(parents=True, exist_ok=True)
    elif isinstance(paths, str):
        Path(paths).mkdir(parents=True, exist_ok=True)

def heatmaps(matrix, fname, type, bounded=False):
    """Creates heatmaps for the matrices."""
    if (type != 'rules') and (type != 'fc') and (type != 'sc') and (type != 'predfc') and (type != 'pval'):
        raise RuntimeError()
    
    ax = plt.axes()
    # Rules Matrix.
    if type == 'rules':
        sns.heatmap(matrix, center=0, cmap='vlag', square=True, ax=ax)
        ax.set_title('Fitted O')
    # Functional Connectivity Matrix, B.
    elif type == 'fc':
        sns.heatmap(matrix, vmin=-1, vmax=1, center=0, cmap='vlag', square=True, ax=ax)
        ax.set_title('B')
    # Predicted Functional Connectivity Matrix, predicted_B.
    elif type == 'predfc':
        if bounded:
            sns.heatmap(matrix, center=0, vmin=-1, vmax=1, cmap='vlag', square=True, ax=ax)
        else:
            sns.heatmap(matrix, center=0, cmap='vlag', square=True, ax=ax)
        ax.set_title('Predicted B')
    # Structural Connectivity Matrix, X.
    elif (type == 'sc'):
        sns.heatmap(matrix, cmap='Greys', square=True, ax=ax)
        ax.set_title('X')
    elif (type == 'pval'):
        sns.heatmap(matrix, vmin=0, vmax=1, cmap='Greys', square=True, ax=ax)
        ax.set_title('P Values')
        
    plt.tight_layout()
    plt.savefig(fname, dpi=500)
    plt.close()

def get_schaefer100_data(path):
        """Base function for getting scaefer data. Used by get_schaefer100_fc and get_schaefer100_sc"""
        data = []
        with open(path, 'r') as file:
            for line in file:
                row = [float(value) for value in line.strip().split(',')]
                data.append(row)
        data = np.array(data)
        data_complement = np.copy(data.T)
        np.fill_diagonal(data_complement, 0)
        data += data_complement
        return data

def get_schaefer100_fc(id):
    """Gets the functional connnectome matrix, B."""
    id = f'00{id}' if id < 10 else f'0{id}'
    path = f'data/micapipe/micapipe_MICs_v1.1/micapipe/sub-HC{id}/ses-01/'
    fc_path = path + f'func/sub-HC{id}_ses-01_space-fsnative_atlas-schaefer100_desc-fc.txt'
    sfc100 = get_schaefer100_data(fc_path)
    np.fill_diagonal(sfc100, 1) # Sets diagonal values to 0.
    return sfc100

def get_schaefer100_sc(id):
    """Gets the structural connectome matrix, X."""
    id = f'00{id}' if id < 10 else f'0{id}'
    path = f'data/micapipe/micapipe_MICs_v1.1/micapipe/sub-HC{id}/ses-01/'
    sc_path = path + f'dwi/sub-HC{id}_ses-01_space-dwinative_atlas-schaefer100_desc-sc.txt'
    ssc100 = get_schaefer100_data(sc_path)
    return ssc100

def remove_medial_wall(matrix):
    matrix = np.delete(matrix, 65, 0)
    matrix = np.delete(matrix, 65, 1)
    matrix = np.delete(matrix, 14, 0)
    matrix = np.delete(matrix, 14, 1)
    return matrix

def get_secondary_data(path):
    """
    Reads matrices created by our scripts (Not from the 
    dataset).
    """
    data = []
    with open(path, 'r') as file:
        for line in file:
            row = [float(value) for value in line.strip().split(' ')]
            data.append(row)
    data = np.array(data)
    return data


class Writer_Reader:
    def __init__(self, base_path, log=False, log_base=10):
        self.base_path = base_path if base_path.endswith('/') else base_path + '/'
        self.output_path = base_path + f'output_log{log_base}' if log else base_path + 'output_standard'

    def get_secondary_data(self, path):
        """
        Reads matrices created by our scripts (Not from the 
        dataset).
        """
        data = []
        with open(path, 'r') as file:
            for line in file:
                row = [float(value) for value in line.strip().split(' ')]
                data.append(row)
        data = np.array(data)
        return data

    def _output_matrix(self, matrix, id, matrix_type, heatmap_type):
        """
        Outputs a matrix to a file.
        """
        heatmap_path, matrix_path = self.output_path + f'heatmaps/{matrix_type}/', self.output_path + f'matrices/{matrix_type}/'
        check_paths([heatmap_path, matrix_path])

        # Write the heatmaps and numpy array to files.
        np.savetxt(f'{matrix_path}{matrix_type}{id}', matrix)
        matrix = remove_medial_wall(matrix) # Remove the medial wall before plotting
        heatmaps(matrix, f'{heatmap_path}{matrix_type}{id}', heatmap_type)

    def output_X(self, X, id):
        """
        Outputs the X matrix for a single subject.
        """
        self._output_matrix(X, id, 'X', 'sc')

    def output_B(self, B, id):
        """
        Outputs the B matrix for a single subject.
        """
        self._output_matrix(B, id, 'B', 'fc')

    def output_predicted_B(self, predicted_B, id):
        """
        Outputs the predicted B matrix for a single subject.
        """
        self._output_matrix(predicted_B, id, 'predicted_B', 'predfc')

    def output_sig_predicted_B(self, sig_predicted_B, id):
        """
        Outputs the predicted B matrix for a single subject.
        """
        self._output_matrix(sig_predicted_B, id, 'Sig_predicted_B', 'predfc')

    def output_group_null_rules(self, null_rules, null_num):
        """
        Outputs group null rules.
        """
        path = self.output_path + 'null_rules'
        check_paths(path)
        np.savetxt(f'{path}/null_rules_{null_num}', null_rules)

    def output_group_sig_rules(self, group_sig_rules):
        """
        Outputs the significant rules for each subject.
        """
        self._output_matrix(group_sig_rules, '', 'sig_rules', 'rules')

    def output_sl_sig_rules(self, group_sig_rules, id):
        """
        Outputs the significant rules for each subject.
        """
        self._output_matrix(group_sig_rules, '', f'sig_rules{id}', 'rules')

    def output_sl_rules(self, rules, id):
        """
        Outputs the fitted rules for a single subject.
        """
        self._output_matrix(rules, id, 'fitted_O', 'rules')

    def output_group_rules(self, rules):
        """
        Outputs the fitted rules for the group.
        """
        self._output_matrix(rules, '', 'fitted_O', 'rules')

    def output_group_pval_matrix(self, group_rule_pvals):
        """
        Outputs the pval matrix for each subject.
        """
        self._output_matrix(group_rule_pvals, '', 'pval_matrix', 'pval')

    def get_predicted_B(self, id):
        path = self.output_path + f'matrices/predictedB/predicted_B{id}'
        return self.get_secondary_data(path)
    
    def get_fc_null(self, id, null_num):
        path = self.base_path + f'fc_nulls/{id}/null_B_{null_num}'
        print(path, flush=True)
        if not os.path.exists(path):
            print(f'fc null for id: {id}, null_num: {null_num} not found.')
            return None
        data = self.get_secondary_data(path)
        return data
    
    def get_group_null_rules(self, null_num):
        path = self.base_path + f'rule_nulls/null_rules_{null_num}'
        if not os.path.exists(path):
            print(f'rule null for null_num {null_num} not found.', flush=True)
            return None
        data = self.get_secondary_data(path)
        return data
    
    def get_sl_null_rules(self, id, null_num):
        path = self.base_path + f'rule_nulls/{id}/null_rules_{null_num}'
        if not os.path.exists(path):
            print(f'rule null for id {id} null_num {null_num} not found.', flush=True)
            return None
        data = self.get_secondary_data(path)
        return data
