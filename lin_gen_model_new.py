import numpy as np
import os
import math
import seaborn as sns
import time
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import wilcoxon
from scipy import stats
from data_manager import get_schaefer100_fc, get_schaefer100_sc, Writer_Reader


class Subject:
    def __init__(self, id, log=False, log_base=10):
        if id == None:
            self.id = None
            self.X = None
            self.B = None
            self.b, self.K = None, None
        else:
            # Read in the SC matrix, X.
            self.id = id
            self.X = get_schaefer100_sc(id)

            if log:
                # Define lambda function to put X to log base 10.
                log_func = lambda x: math.log(x + 1, log_base)
                vlog_func = np.vectorize(log_func)
                self.X = vlog_func(self.X)

            # Read in the FC matrix, B.
            self.B = get_schaefer100_fc(id)
            self.b, self.K = self.symmetric_modification()
    
    def symmetric_modification(self):
        """Removes the duplicate elements, in b, inherent in any symmetric matrix.

        Takes the FC and SC matrix as B and X. It then flattens B, and removes
        the elements in B that are duplicates. In other words, it keeps only the values of the 
        upper triangular portion of the matrix, as the data aross the diagonal is
        redundant, as the matrix is symmetric. This allows training to speed up, and ensures
        that our fitted rules can be symmetric. Then, we take the kroneckor product of X
        with itself and remove the associated columns and rows that were removed from the
        flattened B, so that they align when mutliplied together, as follows: kron(X, X) vec(b).
        """
        # Computes K, b.
        b = self.B.flatten()
        K = np.kron(self.X, self.X)
        dim_X = np.shape(self.B)[0] # The number of brain regions.
        len_b = dim_X ** 2

        computed_cols = []
        to_del = []
        for col in range(len_b):
            # The columns and rows to delete from K and b are computed. K is altered to maintain the original equation.
            current_column = (col) // dim_X
            current_row = (col) % dim_X

            if current_column != current_row:
                add_col = int(current_row * dim_X + current_column)
                if add_col not in computed_cols:
                    """ The columns that are multiplied by the same element in b are added togther.
                    Because matrix multiplication is linear, this allows our compacted equation
                    to equal the original equation. The original equation cannot be
                    recovered afterwards, but the solution to this equation then gives the solution
                    to the original equation.
                    """
                    to_del.append(add_col)
                    K[:, col] += K[:, add_col]
                computed_cols.append(col)

        K = np.delete(np.delete(K, to_del, 1), to_del, 0) # Deletes the proper rows, then columns.          
        b = np.delete(b, to_del) # Deletes the duplicate elements.
        return b, K
    
    def inverse_symmetric_modification(self, flat_O, mat_size=None):
        """Inverses the effects of symmetric modification."""
        shape_O = np.shape(self.X) if self.X is not None else mat_size
        dim_O = shape_O[0]
        O = np.zeros(shape=shape_O)

        indexes_1d = []
        for i in range(dim_O):
            # Creates list of indexes in upper right triangular portion of matrix of size (dim, dim).
            temp_indexes = [(i * dim_O + j + i) for j in range(dim_O - i)]
            indexes_1d += temp_indexes

        indexes_2d = []
        for index in indexes_1d:
            column = (index) // dim_O
            row = (index) % dim_O
            indexes_2d.append([column, row])

        for (index, index_2d) in enumerate(indexes_2d):
            O[index_2d[0], index_2d[1]] = flat_O[index]

        O_complement = np.copy(O.T)
        np.fill_diagonal(O_complement, 0)
        O += O_complement
        return O

    def calc_predicted_B(self, rules):
        """Makes a prediction for a single subject."""
        return  self.X @ rules @ self.X

class Group(Writer_Reader):
    def __init__(self, ids, out_path, log=False, log_base=10):
        out_path = out_path if out_path.endswith('/') else out_path + '/'
        super().__init__(out_path, log=log, log_base=log_base)
        self.subjects = [Subject(id) for id in ids]
        self.K_stack, self.b_stack = self.stack_data()

    def stack_data(self):
        """Stacks the data for training."""
        Ks_to_stack = [subject.K for subject in self.subjects]
        bs_to_stack = [subject.b for subject in self.subjects]

        K_stack = np.vstack(Ks_to_stack)
        b_stack = np.hstack(bs_to_stack)
        return K_stack, b_stack
    
    def train_group(self):
        """Trains the model with algebraic linear regression and returns the fitted rules."""
        flat_rules = np.linalg.pinv(self.K_stack) @ self.b_stack
        subject_methods = Subject(None)
        return subject_methods.inverse_symmetric_modification(flat_rules, mat_size=np.shape(self.subjects[0].X))
    
    def fc_null_symmetric_mod(self, fc_null):
        """Helper function for gen_fc_nulls: modifies fc_null for null_training."""
        null_b = fc_null.flatten()
        dim_B = np.shape(fc_null)[0]
        len_b = dim_B ** 2

        computed_cols = []
        to_del = [] # Indices to delete from null_b.
        for col in range(len_b):
            current_column = (col) // dim_B
            current_row = (col) % dim_B
            
            if current_column != current_row:
                add_col = int(current_row * dim_B + current_column)
                if add_col not in computed_cols:
                    to_del.append(add_col)
                computed_cols.append(col)

        null_b = np.delete(null_b, to_del) # Deletes the duplicate elements.
        return null_b

    def gen_null_rules(self):
        """Fits and outputs null rules."""
        # Instantiate to use Subjects' methods.
        subject_methods = Subject(None)

        # Make stack of null fcs, one for each subject.
        for null_num in range(0, 100):
            start_time = time.time() # Start time to track how long each iter takes.

            fc_nulls_to_stack = []
            Ks_to_stack = []

            # Add the fc null for each null num to the list to then stack.
            for subject in self.subjects:
                Ks_to_stack.append(subject.K)
                fc_null = self.fc_null_symmetric_mod(self.get_null_fc(subject.id, null_num))
                fc_nulls_to_stack.append(fc_null)
            fc_nulls_stack = np.hstack(fc_nulls_to_stack)
            K_stack = np.vstack(Ks_to_stack)

            # Train the null rule set for the null_num.
            flat_null_rules = np.linalg.pinv(K_stack) @ fc_nulls_stack
            null_rules = subject_methods.inverse_symmetric_modification(flat_null_rules, mat_size=np.shape(subject.X))

            # Writes the results to text file.
            self.output_group_null_rules(null_rules, null_num)

            # Print time taken to the log.
            print(f'Null Rules {null_num} found in: {time.time() - start_time} seconds', flush=True)

    def calc_sig_rules(self, rules, pval_cutoff=0.05):
        """Finds the significant rules for a single subject."""
        # Gathers all the null rules.
        rules_to_stack = []
        number_of_nulls = 100
        for null_num in range(number_of_nulls):
            try:
                null_rules = self.get_group_null_rules(null_num)
                if rules is None:
                    print(f'null_num_{null_num} is none!')
                else:
                    rules_to_stack.append(null_rules)
            except:
                print(f'null_num_{null_num} is missing!')

        # Stacks null rules matrices in 3d array.
        stacked_rules = np.stack(rules_to_stack)

        rule_dimensions = np.shape(rules)
        pval_matrix = np.zeros(shape=rule_dimensions)
        for i in range(rule_dimensions[0]):
            for j in range(rule_dimensions[1]):
                null_population = stacked_rules[:, i, j]
                # For wilcoxon(x - y), if x - y is a zero vector, wilcoxon does not work. Setting pval to 0 or 1 works here, since the rule val is 0.
                if np.all((null_population - rules[i][j]) == 0):
                    pvalue = 0
                else:
                    statistic, pvalue = wilcoxon(null_population - rules[i][j])
                pval_matrix[i][j] = pvalue

        # Apply FDR correction. The numpy array must be reshaped to 1d for FDR and put back into its original shape after.
        pval_matrix = pval_matrix.reshape(1, -1)
        pval_matrix = stats.false_discovery_control(pval_matrix)
        pval_matrix = pval_matrix.reshape(rule_dimensions)

        # For each rule, if the associated p value is greater than the cutoff, set that rule to 0.       
        for i in range(rule_dimensions[0]):
            for j in range(rule_dimensions[1]):
                if pval_matrix[i][j] > pval_cutoff:
                    rules[i, j] = 0
        sig_rules = rules
        return sig_rules, pval_matrix

    def calc_predicted_Bs(self, rules):
        """Makes predictions for each subject."""
        predicted_Bs = []
        for subject in self.subjects:
            predicted_B = subject.calc_predicted_B(rules)
            predicted_Bs.append(predicted_B)
        return predicted_Bs

    def output_Xs(self):
        """Outputs the X matrices for each subject."""
        for subject in self.subjects:
            self.output_X(subject.X, subject.id)

    def output_Bs(self):
        """Outputs the B matrices for each subject."""
        for subject in self.subjects:
            self.output_B(subject.B, subject.id)

    def output_predicted_Bs(self, rules):
        """Outputs the predicted B matrices for each subject."""
        for subject in self.subjects:
            predicted_fc = subject.calc_predicted_B(rules)
            self.output_predicted_B(predicted_fc, subject.id)

def main():
    """
    Train the group and output the results.
    """
    # Define the subject ids.
    num_subjects = 1
    ids = [num for num in range(1, num_subjects + 1)]

    # Instantiate the group.
    out_path = 'test.'
    group = Group(ids, out_path)

    # Train the model.
    rules = group.train_group()

    # Generate the null rules and calculate significant rules.
    group.gen_null_rules()
    sig_rules, pval_matrix = group.calc_sig_rules(rules)

    # Output the matrices.
    group.output_Xs()
    group.output_Bs()
    group.output_predicted_Bs(rules)
    group.output_group_rules(rules)
    group.output_group_sig_rules(sig_rules)
    group.output_group_pval_matrix(pval_matrix) 


if __name__ == '__main__':
    main()