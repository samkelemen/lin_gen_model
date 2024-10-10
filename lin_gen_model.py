import math
import time
import numpy as np
import sklearn

from scipy.stats import wilcoxon
from scipy import stats
import sklearn.model_selection
from sklearn.linear_model import Lasso
from statsmodels.regression.linear_model import OLS
from data_manager import get_schaefer100_fc, get_schaefer100_sc, WriterReader
import data_manager


class Subject:
    def __init__(self, subject_id, tumor_ds=False, pre_resection=True, log=False, log_base=10):
        self.subject_id = subject_id

        # Read in the SC matrix
        if not tumor_ds:
            self.sc_matrix = get_schaefer100_sc(subject_id)
        else:
            if pre_resection:
                self.sc_matrix = data_manager.get_pre_resection_sc(subject_id)
            else:
                self.sc_matrix = data_manager.get_post_resection_sc(subject_id)

        if log:
            # Define lambda function to put sc to log base 10
            log_func = lambda x: math.log(x + 1, log_base)
            vlog_func = np.vectorize(log_func)
            self.sc_matrix = vlog_func(self.sc_matrix)

        # Read in the FC matrix
        if not tumor_ds:
            self.fc_matrix = get_schaefer100_fc(subject_id)
        else:
            if pre_resection:
                self.fc_matrix = data_manager.get_pre_resection_fc(subject_id)
            else:
                self.fc_matrix = data_manager.get_post_resection_fc(subject_id)

        self.flat_fc_matrix, self.sc_kron_matrix = self.symmetric_modification()
    
    def symmetric_modification(self):
        """Removes the duplicate elements, in fc and sc for more efficient training."""
        # Computes K, fc.
        flat_fc_matrix = self.fc_matrix.flatten()
        sc_kron_matrix = np.kron(self.sc_matrix, self.sc_matrix)
        dim_sc_matrix = np.shape(self.fc_matrix)[0] # The number of brain regions.
        len_b = dim_sc_matrix ** 2

        computed_cols = []
        to_del = []
        for col in range(len_b):
            # The columns and rows to delete from K and fc are computed. K is altered to maintain the original equation.
            current_column = (col) // dim_sc_matrix
            current_row = (col) % dim_sc_matrix

            if current_column != current_row:
                add_col = int(current_row * dim_sc_matrix + current_column)
                if add_col not in computed_cols:
                    to_del.append(add_col)
                    sc_kron_matrix[:, col] += sc_kron_matrix[:, add_col]
                computed_cols.append(col)

        sc_kron_matrix = np.delete(np.delete(sc_kron_matrix, to_del, 1), to_del, 0) # Deletes the proper rows, then columns.          
        flat_fc_matrix = np.delete(flat_fc_matrix, to_del) # Deletes the duplicate elements.
        return flat_fc_matrix, sc_kron_matrix
    
    def inverse_symmetric_modification(self, flat_rules, mat_size=None):
        """Inverses the effects of symmetric modification."""
        rules_shape = np.shape(self.sc_matrix) if self.sc_matrix is not None else mat_size
        dim_rules = rules_shape[0]
        rules = np.zeros(shape=rules_shape)

        indexes_1d = []
        for i in range(dim_rules):
            # Creates list of indexes in upper right triangular portion of matrix of size (dim, dim).
            temp_indexes = [(i * dim_rules + j + i) for j in range(dim_rules - i)]
            indexes_1d += temp_indexes

        indexes_2d = []
        for index in indexes_1d:
            column = (index) // dim_rules
            row = (index) % dim_rules
            indexes_2d.append([column, row])

        for (index, index_2d) in enumerate(indexes_2d):
            rules[index_2d[0], index_2d[1]] = flat_rules[index]

        rules_complement = np.copy(rules.T)
        np.fill_diagonal(rules_complement, 0)
        rules += rules_complement
        return rules

    def calc_predicted_fc(self, rules):
        """Makes a prediction for a single subject."""
        return  self.sc_matrix @ rules @ self.sc_matrix

class GroupLevelModel(WriterReader):
    def __init__(self, subject_ids, out_path, tumor_ds=False, pre_resection=True, log=False, log_base=10):
        out_path = out_path if out_path.endswith('/') else out_path + '/'
        super().__init__(out_path, tumor_ds=tumor_ds)
        self.subjects = [Subject(subject_id, tumor_ds=tumor_ds, pre_resection=pre_resection) for subject_id in subject_ids]
        self.K_stack, self.b_stack = self.stack_data()

    def stack_data(self):
        """Stacks the data for training."""
        Ks_to_stack = [subject.sc_kron_matrix for subject in self.subjects]
        bs_to_stack = [subject.flat_fc_matrix for subject in self.subjects]

        K_stack = np.vstack(Ks_to_stack)
        b_stack = np.hstack(bs_to_stack)
        return K_stack, b_stack
    
    def train_group(self):
        """Trains the model with algebraic linear regression and returns the fitted rules."""
        flat_rules = np.linalg.pinv(self.K_stack) @ self.b_stack
        return self.inverse_symmetric_modification(flat_rules, mat_size=np.shape(self.subjects[0].sc_matrix))
    
    def fc_null_symmetric_mod(self, fc_null):
        """Helper function for gen_fc_nulls: modifies fc_null for null_training."""
        null_b = fc_null.flatten()
        dim_fc = np.shape(fc_null)[0]
        len_b = dim_fc ** 2

        computed_cols = []
        to_del = [] # Indices to delete from null_b.
        for col in range(len_b):
            current_column = (col) // dim_fc
            current_row = (col) % dim_fc
            
            if current_column != current_row:
                add_col = int(current_row * dim_fc + current_column)
                if add_col not in computed_cols:
                    to_del.append(add_col)
                computed_cols.append(col)

        null_b = np.delete(null_b, to_del) # Deletes the duplicate elements.
        return null_b
    
    def inverse_symmetric_modification(self, flat_rules, mat_size=None):
        """Inverses the effects of symmetric modification."""
        rules_shape = np.shape(self.subjects[0].sc_matrix) if self.subjects[0].sc_matrix is not None else mat_size
        dim_rules = rules_shape[0]
        rules = np.zeros(shape=rules_shape)

        indexes_1d = []
        for i in range(dim_rules):
            # Creates list of indexes in upper right triangular portion of matrix of size (dim, dim).
            temp_indexes = [(i * dim_rules + j + i) for j in range(dim_rules - i)]
            indexes_1d += temp_indexes

        indexes_2d = []
        for index in indexes_1d:
            column = (index) // dim_rules
            row = (index) % dim_rules
            indexes_2d.append([column, row])

        for (index, index_2d) in enumerate(indexes_2d):
            rules[index_2d[0], index_2d[1]] = flat_rules[index]

        rules_complement = np.copy(rules.T)
        np.fill_diagonal(rules_complement, 0)
        rules += rules_complement
        return rules

    def calc_sig_rules(self, rules, pval_cutoff=0.05):
        """Finds the significant rules for a single subject."""
        # Gathers all the null rules.
        rules_to_stack = []
        number_of_nulls = 100
        for null_num in range(number_of_nulls):
            null_rules = self.get_group_null_rules(null_num)
            if rules is None:
                print(f'null_num_{null_num} is none!')
            else:
                rules_to_stack.append(null_rules)

        # Stacks null rules matrices in 3d array.
        stacked_rules = np.stack(rules_to_stack)

        rule_dimensions = np.shape(rules)
        pval_matrix = np.zeros(shape=rule_dimensions)
        for i in range(rule_dimensions[0]):
            for j in range(rule_dimensions[1]):
                null_population = stacked_rules[:, i, j]
                # For wilcoxon(sc - y), if sc - y is a zero vector, wilcoxon does not work. Setting pval to 0 or 1 works here, since the rule val is 0.
                if np.all((null_population - rules[i][j]) == 0):
                    pvalue = 0
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
                    rules[i, j] = 0
        sig_rules = rules
        return sig_rules, pval_matrix

    def calc_predicted_fcs(self, rules):
        """Makes predictions for each subject."""
        predicted_fcs = []
        for subject in self.subjects:
            predicted_fc = subject.calc_predicted_fc(rules)
            predicted_fcs.append(predicted_fc)
        return predicted_fcs

    def output_scs(self):
        """Outputs the sc matrices for each subject."""
        for subject in self.subjects:
            self.output_sc(subject.sc_matrix, subject.subject_id)

    def output_fcs(self):
        """Outputs the fc matrices for each subject."""
        for subject in self.subjects:
            self.output_fc(subject.fc_matrix, subject.subject_id)

    def output_predicted_fcs(self, rules):
        """Outputs the predicted fc matrices for each subject."""
        for subject in self.subjects:
            predicted_fc = subject.calc_predicted_fc(rules)
            self.output_predicted_fc(predicted_fc, subject.subject_id)

class SubjectLevelModel(Subject, WriterReader):
    def __init__(self, subject_id, output_path, tumor_ds=False, pre_resection=True, log=False, log_base=10):
        Subject.__init__(self, subject_id, tumor_ds=tumor_ds, pre_resection=pre_resection, log=log, log_base=log_base)
        WriterReader.__init__(self, output_path, tumor_ds=tumor_ds)

    def calc_sig_rules(self, rules, pval_cutoff=0.05):
        """Finds the significant rules for a single subject."""
        # Gathers all the null rules.
        rules_to_stack = []
        for null_num in range(100):
            try:
                null_rules = self.get_sl_null_rules(self.subject_id, null_num)
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
                # For wilcoxon(sc - y), if sc - y is a zero vector, wilcoxon does not work. Setting pval to 0 or 1 works here, since the rule val is 0.
                if np.all((null_population - rules[i][j]) == 0):
                    pvalue = 0
                else:
                    statistic, pvalue = stats.wilcoxon(null_population - rules[i][j])
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
        return sig_rules

    def calc_alpha_grid(self):
        """
        Calculates alpha grid described in paper: [paper name here]
        """
        inner_products = []
        N = len(self.flat_fc_matrix)
        for row in self.sc_kron_matrix.T:
            inner_products.append(np.dot(row, self.flat_fc_matrix))
        alpha_max = max(inner_products) / N
        alpha_min = 0.0001 * alpha_max
        alphas = np.logspace(math.log(alpha_min), math.log(alpha_max), num=100, base=math.exp(1))
        return alphas

    def train_lasso_bic(self, alpha_vals):
        """
        Fit the model with Lasso and BIC for selection.
        """
        # Initialize the best BIC and corresponding alpha value
        best_bic = np.inf
        best_alpha = None
        best_model = None

        K = self.sc_kron_matrix
        fc = self.flat_fc_matrix

        # Loop over the alpha values
        for alpha in alpha_vals:
            try:
                # Fit a Lasso model using scikit-learn's Lasso function
                lasso_model = Lasso(alpha=alpha, fit_intercept=False)
                lasso_model.fit(K, fc)
                
                # Get the coefficients of the Lasso model
                beta_opt = lasso_model.coef_
                
                # Create a new sc matrix with only the selected features (non-zero coefficients)
                selected_features = np.where(beta_opt != 0)[0]
                K_selected = K[:, selected_features]
                
                # Fit an OLS model using statsmodels to get the BIC
                ols_model = OLS(K, K_selected).fit()
                bic_value = ols_model.bic
                
                # Check if this alpha value results in a lower BIC
                if bic_value < best_bic:
                    best_bic = bic_value
                    best_alpha = alpha
                    best_model = lasso_model

                print(f"Trained with alpha = {alpha}.", flush=True)
            except:
                pass
            
        # Convert the model output to O matrix and return the best alpha and rule set.
        rules = self.inverse_symmetric_modification(best_model.coef_)
        return best_alpha, rules

def train_sl(subject_id, output_path):
    # Instantiate the subject model.
    subject_model = SubjectLevelModel(subject_id, output_path, tumor_ds=True, pre_resection=False)

    # Calculate the alpha grid to be used for training.
    alpha_grid = subject_model.calc_alpha_grid()

    # Train the model.
    best_alpha, rules = subject_model.train_lasso_bic(alpha_grid)
    print(f'Best alpha for id {subject_id}: {best_alpha}')

    # Output the fitted rules
    subject_model.output_sl_rules(rules, subject_model.subject_id)

    # Output predicted FC.
    predicted_fc = subject_model.calc_predicted_fc(rules)
    subject_model.output_predicted_fc(predicted_fc, subject_model.subject_id)

    # Calculate significant rules
    sig_rules = subject_model.calc_sig_rules(rules)
    subject_model.output_sl_sig_rules(sig_rules, subject_id)

    # Output predicted FC with significant rules
    sig_predicted_fc = subject_model.calc_predicted_fc(sig_rules)
    subject_model.output_sig_predicted_fc(sig_predicted_fc, subject_model.subject_id)

    # Output SC and FC matrices
    subject_model.output_fc(subject_model.fc_matrix, subject_model.subject_id)
    subject_model.output_sc(subject_model.sc_matrix, subject_model.subject_id)

def train_gl():
    """
    Train the group and output the results.
    """
    # Define the subject ids.
    subject_ids = [1, 2, 3, 5, 6, 7, 8, 10, 13, 15, 16, 17, 20, 24, 25, 26, 28]

    # Instantiate the group.
    out_path = 'post_resection/'
    group = GroupLevelModel(subject_ids, out_path, tumor_ds=True, pre_resection=False)

    # Train the model.
    rules = group.train_group()

    # Generate the null rules and calculate significant rules.
    sig_rules, pval_matrix = group.calc_sig_rules(rules)

    # Output the matrices.
    group.output_scs()
    group.output_fcs()
    group.output_predicted_fcs(rules)
    group.output_group_rules(rules)
    group.output_group_sig_rules(sig_rules)
    group.output_group_pval_matrix(pval_matrix) 


if __name__ == '__main__':
    train_gl()
    train_sl()
