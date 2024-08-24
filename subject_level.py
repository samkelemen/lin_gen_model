import numpy as np
from scipy import stats
import sklearn
import math
import time

import sklearn.model_selection
from sklearn.linear_model import Lasso
from statsmodels.regression.linear_model import OLS

from data_manager import Writer_Reader
from lin_gen_model import Subject

class SubjectLevelModel(Subject, Writer_Reader):
    def __init__(self, id, output_path, log=False, log_base=10):
        Subject.__init__(self, id, log=log, log_base=log_base)
        Writer_Reader.__init__(self, output_path, log=log, log_base=log_base)

    def calc_sig_rules(self, rules, pval_cutoff=0.05):
        """Finds the significant rules for a single subject."""
        # Gathers all the null rules.
        rules_to_stack = []
        for null_num in range(100):
            try:
                null_rules = self.get_sl_null_rules(self.id, null_num)
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
                    statistic, pvalue = stats.wilcoxon(null_population - rules[i][j])
                pval_matrix[i][j] = pvalue

        # Apply FDR correction. The numpy array must be reshaped to 1d for FDR and put back into its original shape after.
        pval_matrix = pval_matrix.reshape(1, -1)
        pval_matrix = stats.false_discovery_control(pval_matrix)
        pval_matrix = pval_matrix.reshape((116, 116))

        # For each rule, if the associated p value is greater than the cutoff, set that rule to 0.       
        for i in range(rule_dimensions[0]):
            for j in range(rule_dimensions[1]):
                if pval_matrix[i][j] > pval_cutoff:
                    rules[i, j] = 0
                    print(f'{i}, {j}', flush=True)
        sig_rules = rules
        return sig_rules

    def calc_alpha_grid(self):
        """
        Calculates alpha grid described in paper: [paper name here]
        """
        inner_products = []
        N = len(self.b)
        for row in self.K.T:
            inner_products.append(np.dot(row, self.b))
        alpha_max = max(inner_products) / N
        alpha_min = 0.0001 * alpha_max
        alphas = np.logspace(math.log(alpha_min), math.log(alpha_max), num=100, base=math.exp(1))
        return alphas

    def train_with_ridge(self, alphas):
        """Trains the model with ridge regression."""
        model = sklearn.linear_model.RidgeCV(alphas=alphas, fit_intercept=False)
        model.fit(self.K, self.b)
        flat_rules = model.coef_
        alpha = model.alpha_
        rules = self.inverse_symmetric_modification(flat_rules)
        return rules, alpha

    def train_with_lassoBIC(self, alpha_vals):
        # Initialize the best BIC and corresponding alpha value
        best_bic = np.inf
        best_alpha = None
        best_model = None

        K = self.K
        b = self.b

        # Loop over the alpha values
        for alpha in alpha_vals:
            try:
                # Fit a Lasso model using scikit-learn's Lasso function
                lasso_model = Lasso(alpha=alpha, fit_intercept=False)
                lasso_model.fit(K, b)
                
                # Get the coefficients of the Lasso model
                beta_opt = lasso_model.coef_
                
                # Create a new X matrix with only the selected features (non-zero coefficients)
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
        rules = self.inverse_symmetric_modification(best_model.coef_, X)
        return best_alpha, rules

def main(id, output_path):
    # Instantiate the subject model.
    subject_model = SubjectLevelModel(id, output_path)

    # Calculate the alpha grid to be used for training.
    alpha_grid = subject_model.calc_alpha_grid()

    # Train the model.
    best_alpha, rules = subject_model.train_with_lassoBIC(alpha_grid)

    # Output the fitted rules
    subject_model.output_sl_rules(rules, subject_model.id)

    # Output predicted FC.
    predicted_B = subject_model.calc_predicted_B(rules)
    subject_model.output_predicted_B(predicted_B, subject_model.id)

    # Calculate significant rules
    sig_rules = subject_model.calc_sig_rules(rules)

    # Output predicted FC with significant rules
    sig_predicted_B = subject_model.calc_predicted_B(sig_rules)
    subject_model.output_sig_predicted_B(sig_predicted_B, subject_model.id)

    # Output SC and FC matrices
    subject_model.output_B(subject_model.B, subject_model.id)
    subject_model.output_X(subject_model.X, subject_model.id)
