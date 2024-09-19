import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat

from pathlib import Path


def check_paths(paths):
    """Checks if the paths exist, and if not, creates them."""
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                Path(path).mkdir(parents=True, exist_ok=True)
    elif isinstance(paths, str):
        Path(paths).mkdir(parents=True, exist_ok=True)

def heatmaps(matrix, fname, matrix_type, bounded=False):
    """Creates heatmaps for the matrices."""
    if (matrix_type != 'rules') and (matrix_type != 'fc') and (matrix_type != 'sc') \
          and (matrix_type != 'predfc') and (matrix_type != 'pval'):
        raise RuntimeError()

    # Create the axes for the plot
    ax = plt.axes()
    # Rules Matrix.
    if matrix_type == 'rules':
        sns.heatmap(matrix, center=0, cmap='vlag', square=True, ax=ax)
        ax.set_title('Fitted O')
    # Functional Connectivity Matrix, B.
    elif matrix_type == 'fc':
        sns.heatmap(matrix, vmin=-1, vmax=1, center=0, cmap='vlag', square=True, ax=ax)
        ax.set_title('B')
    # Predicted Functional Connectivity Matrix, predicted_B.
    elif matrix_type == 'predfc':
        if bounded:
            sns.heatmap(matrix, center=0, vmin=-1, vmax=1, cmap='vlag', square=True, ax=ax)
        else:
            sns.heatmap(matrix, center=0, cmap='vlag', square=True, ax=ax)
        ax.set_title('Predicted B')
    # Structural Connectivity Matrix, X.
    elif matrix_type == 'sc':
        sns.heatmap(matrix, cmap='Greys', square=True, ax=ax)
        ax.set_title('X')
    elif matrix_type == 'pval':
        sns.heatmap(matrix, vmin=0, vmax=1, cmap='Greys', square=True, ax=ax)
        ax.set_title('P Values')

    plt.tight_layout()
    plt.savefig(fname, dpi=500)
    plt.close()

def get_schaefer100_data(path):
    """
    Base function for getting scaefer data. Used by get_schaefer100_fc 
    and get_schaefer100_sc
    """
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            row = [float(value) for value in line.strip().split(',')]
            data.append(row)
    data = np.array(data)
    data_complement = np.copy(data.T)
    np.fill_diagonal(data_complement, 0)
    data += data_complement
    return data

def get_schaefer100_fc(subject_id):
    """Gets the functional connnectivity matrix."""
    subject_id = f'00{subject_id}' if subject_id < 10 else f'0{subject_id}'
    path = f'data/micapipe/micapipe_MICs_v1.1/micapipe/sub-HC{subject_id}/ses-01/'
    fc_path = path + f'func/sub-HC{subject_id}_ses-01_space-fsnative_atlas-schaefer100_desc-fc.txt'
    sfc100 = get_schaefer100_data(fc_path)
    np.fill_diagonal(sfc100, 1) # Sets diagonal values to 0.
    return sfc100

def get_schaefer100_sc(subject_id):
    """Gets the structural connectivity matrix."""
    subject_id = f'00{subject_id}' if subject_id < 10 else f'0{subject_id}'
    path = f'data/micapipe/micapipe_MICs_v1.1/micapipe/sub-HC{subject_id}/ses-01/'
    sc_path = path + f'dwi/sub-HC{subject_id}_ses-01_space-dwinative_atlas-schaefer100_desc-sc.txt'
    ssc100 = get_schaefer100_data(sc_path)
    return ssc100
    
def get_pre_resection_sc(subject_id):
    """Gets the pre-resection structural connectivity matrix."""
    subject_id = f'0{subject_id}' if subject_id < 10 else subject_id
    return loadmat(f'derivatives_preop/TVB/sub-PAT{subject_id}/ses-preop/SCthrAn.mat')['SCthrAn']

def get_post_resection_sc(subject_id):
    """Gets the post-resection structural connectvity matrix."""
    subject_id = f'0{subject_id}' if subject_id < 10 else subject_id
    return loadmat(f'derivatives_postop/TVB/sub-PAT{subject_id}/ses-postop/SCthrAn.mat')['SCthrAn']

def get_pre_resection_fc(subject_id):
    """Gets the pre-resection functional connectivity matrix."""
    subject_id = f'0{subject_id}' if subject_id < 10 else subject_id
    return loadmat(f'derivatives_preop/TVB/sub-PAT{subject_id}/ses-preop/FC.mat')['FC_cc_DK68']

def get_post_resection_fc(subject_id):
    """Gets the post-resection functional connectivity matrix."""
    subject_id = f'0{subject_id}' if subject_id < 10 else subject_id
    return loadmat(f'derivatives_postop/TVB/sub-PAT{subject_id}/ses-postop/FC.mat')['FC_cc_DK68']

def remove_medial_wall(matrix):
    """
    Remove medial wall from the inputted matrix.
    """
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
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            row = [float(value) for value in line.strip().split(' ')]
            data.append(row)
    data = np.array(data)
    return data


class WriterReader:
    """
    Read and write processed output.
    """
    def __init__(self, base_path, tumor_ds=False):
        self.base_path = base_path if base_path.endswith('/') else base_path + '/'
        self.output_path = base_path
        self.tumor_ds = tumor_ds

    def get_secondary_data(self, path):
        """
        Reads matrices created by our scripts (Not from the 
        dataset).
        """
        data = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                row = [float(value) for value in line.strip().split(' ')]
                data.append(row)
        data = np.array(data)
        return data

    def _output_matrix(self, matrix, subject_id, matrix_type, heatmap_type):
        """
        Outputs a matrix to a file.
        """
        heatmap_path, matrix_path = self.output_path + f'heatmaps/{matrix_type}/', \
            self.output_path + f'matrices/{matrix_type}/'
        check_paths([heatmap_path, matrix_path])

        # Write the heatmaps and numpy array to files.
        np.savetxt(f'{matrix_path}{matrix_type}{subject_id}', matrix)
        if not self.tumor_ds:
            matrix = remove_medial_wall(matrix) # Remove the medial wall before plotting
        heatmaps(matrix, f'{heatmap_path}{matrix_type}{subject_id}', heatmap_type)

    def output_sc(self, X, subject_id):
        """
        Outputs the X matrix for a single subject.
        """
        self._output_matrix(X, subject_id, 'X', 'sc')

    def output_fc(self, fc_matrix, subject_id):
        """
        Outputs the fc matrix for a single subject.
        """
        self._output_matrix(fc_matrix, subject_id, 'B', 'fc')

    def output_predicted_fc(self, predicted_fc, subject_id):
        """
        Outputs the predicted fc matrix for a single subject.
        """
        self._output_matrix(predicted_fc, subject_id, 'predicted_B', 'predfc')

    def output_sig_predicted_fc(self, sig_predicted_fc, subject_id):
        """
        Outputs the predicted fc matrix for a single subject.
        """
        self._output_matrix(sig_predicted_fc, subject_id, 'Sig_predicted_B', 'predfc')

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

    def output_sl_sig_rules(self, group_sig_rules, subject_id):
        """
        Outputs the significant rules for each subject.
        """
        self._output_matrix(group_sig_rules, subject_id, f'sig_rules', 'rules')

    def output_sl_rules(self, rules, subject_id):
        """
        Outputs the fitted rules for a single subject.
        """
        self._output_matrix(rules, subject_id, 'fitted_O', 'rules')

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

    def get_predicted_fc(self, subject_id):
        """
        Loads predicted fc from the output file.
        """
        path = self.output_path + f'matrices/predictedB/predicted_B{subject_id}'
        return self.get_secondary_data(path)

    def get_fc_null(self, subject_id, null_num):
        """
        Loads the given fc null from the output file.
        """
        path = self.base_path + f'fc_nulls/{subject_id}/fc_null_{null_num}'
        if not os.path.exists(path):
            print(f'fc null for subject_id:{subject_id}, null_num: {null_num} not found.')
            return None
        data = self.get_secondary_data(path)
        return data

    def get_group_null_rules(self, null_num):
        """
        Loads the given group level null rules from the output file.
        """
        path = self.base_path + f'rule_nulls/gl/gl_rule_nulls_{null_num}'
        if not os.path.exists(path):
            print(path)
            print(f'rule null for null_num {null_num} not found.', flush=True)
            return None
        data = self.get_secondary_data(path)
        return data

    def get_sl_null_rules(self, subject_id, null_num):
        """
        Loads the given subject level null rules from the output file.
        """
        path = self.base_path + f'rule_nulls/sl/{subject_id}/rule_nulls_{null_num}'
        if not os.path.exists(path):
            print(f'rule null for subject_id{subject_id} \
                  null_num {null_num} not found.', flush=True)
            return None
        data = self.get_secondary_data(path)
        return data