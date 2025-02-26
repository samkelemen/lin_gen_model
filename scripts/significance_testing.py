"""
Contains function for calulcating significant rules.
"""
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.stats import wilcoxon
import bct

from inout import check_paths


def fc_randomization(subject_id: int, fc_matrix: NDArray[np.float64], num_nulls :int =100):
    """
    Performs network randomization on the FC matrix for the given subject,
    preserving degree and strength distributions, using bct.null_model_und_sign.
    """
    print(f"Generating nulls for subject {subject_id}...", flush=True)

    # If correct directories do not exist, create them.
    out_dir = f"fc_nulls/{subject_id}/"
    check_paths(out_dir)

    # create the specified number of nulls
    for null_num in range(num_nulls):
        # Create null fc matrix
        randomized_fc = bct.null_model_und_sign(fc_matrix, bin_swaps=10, wei_freq=1)[0]

        # Output the 
        outfile = f'fc_nulls/{subject_id}/null_B_{null_num}'
        np.savetxt(outfile, randomized_fc)
        
        print(f"Null number {null_num} computed.", flush=True)

def calc_sig_rules(null_rules_stack:list[NDArray[np.float64]], rules:NDArray[np.float64], pval_cutoff:float=0.05):
    """
    Finds the significant rules for a single subject.
    """
    rule_dimensions = np.shape(rules)
    pval_matrix = np.zeros(shape=rule_dimensions)
    for i in range(rule_dimensions[0]):
        for j in range(rule_dimensions[1]):
            null_population = null_rules_stack[:, i, j]
            # For wilcoxon(sc - y), if sc - y is a zero vector, wilcoxon does not work. Setting pval to 0 or 1 works here, since the rule val is 0.
            if np.all((null_population - rules[i][j]) == 0):
                pvalue = 1
            else:
                pvalue = wilcoxon(null_population - rules[i][j])[1]
            pval_matrix[i][j] = pvalue

    # Apply FDR correction. The numpy array must be reshaped to 1d for FDR and put back into its original shape after.
    pval_matrix = pval_matrix.reshape(1, -1)
    pval_matrix = stats.false_discovery_control(pval_matrix)
    pval_matrix = pval_matrix.reshape(rule_dimensions)

    # For each rule, if the associated p value is greater than the cutoff, set that rule to 0.       
    for i in range(rule_dimensions[0]):
        for j in range(rule_dimensions[1]):
            if pval_matrix[i][j] > pval_cutoff:
                rules[i, j] = np.nan # Setting non significant rules to NaN
    sig_rules = rules
    return sig_rules, pval_matrix
