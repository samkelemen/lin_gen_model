"""
This file handles all input/output for the program.
"""

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

def heatmaps(matrix, title, colored, bounded=False):
    """Creates heatmaps for the matrices."""
    # Create the axes for the plot
    ax = plt.axes()
    ax.set_title(title)
    # Define the color map
    cmap = 'valg' if colored else 'Greys'
    # Create the heatmap
    if bounded:
        sns.heatmap(matrix, vmin=-1, vmax=1, center=0, cmap=cmap, square=True, ax=ax)
    else:
        sns.heatmap(matrix, center=0, cmap=cmap, square=True, ax=ax)
    # Return the plot
    return ax


####################### Shaefer dataset functions ##########################

def get_schaefer100_data(path):
    """
    Base function for getting scaefer data. Used by get_schaefer100_fc 
    and get_schaefer100_sc
    """
    data = np.genfromtxt(path, delimiter=',')
    data_complement = np.copy(data.T)
    np.fill_diagonal(data_complement, 0)
    data += data_complement
    return data

def get_schaefer100_fc(base_dir, subject_id):
    """Gets the functional connnectivity matrix."""
    subject_id = f'{subject_id}{3:}' # Sets  the subject id to 3 digits
    path = base_dir + f'data/micapipe/micapipe_MICs_v1.1/micapipe/sub-HC{subject_id}/ses-01/'
    fc_path = path + f'func/sub-HC{subject_id}_ses-01_space-fsnative_atlas-schaefer100_desc-fc.txt'
    sfc100 = get_schaefer100_data(fc_path)
    np.fill_diagonal(sfc100, 0) # Sets diagonal values to 0.
    return sfc100

def get_schaefer100_sc(base_dir, subject_id):
    """Gets the structural connectivity matrix."""
    subject_id = f'00{subject_id}' if subject_id < 10 else f'0{subject_id}'
    path = base_dir + f'data/micapipe/micapipe_MICs_v1.1/micapipe/sub-HC{subject_id}/ses-01/'
    sc_path = path + f'dwi/sub-HC{subject_id}_ses-01_space-dwinative_atlas-schaefer100_desc-sc.txt'
    ssc100 = get_schaefer100_data(sc_path)
    return ssc100

def remove_medial_wall(matrix):
    """
    Remove medial wall from the inputted matrix.
    """
    matrix = np.delete(matrix, 65, 0)
    matrix = np.delete(matrix, 65, 1)
    matrix = np.delete(matrix, 14, 0)
    matrix = np.delete(matrix, 14, 1)
    return matrix

##################### Resection dataset functions ###########################

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
