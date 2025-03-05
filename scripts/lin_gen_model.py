"""
Contains the functions to train the model.
"""
from collections.abc import Callable
from typing import Any
import math
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from scipy import stats


def algebraic_linear_regression(X: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Fits a linear regression model using algebraic linear regression.
    """
    return np.linalg.pinv(X) @ y

def calc_alpha_grid(X: NDArray[np.float64], y: NDArray[np.float64], alpha_min_max:tuple[float, float]=None, num_alphas: int=100) -> NDArray[np.float64]:
    """
    Calculates alpha grid as described in: 'Regularization Paths for 
    Generalized Linear Models via Coordinate Descent' by Friedman, 
    Hastie, and Tibshirani. 
    """
    if not alpha_min_max:
        inner_products = []
        N = len(y)
        for row in X.T:
            inner_products.append(np.dot(row, y))
        alpha_max = max(inner_products) / N
        alpha_min = 0.0001 * alpha_max
        alpha_max = 5 * alpha_min # This was experientially found to be a better upper bound, as the selected alpha never came close to 5 * min
    else:
        alpha_min = alpha_min_max[0]
        alpha_max = alpha_min_max[1]

    alphas = np.logspace(math.log(alpha_min), math.log(alpha_max), num=num_alphas, base=math.exp(1))
    return alphas

def lasso_regression(X: NDArray[np.float64], y: NDArray[np.float64], alpha: float) -> tuple[NDArray[np.float64] | Any, Any]:
    """
    Fits a Lasso regression model using scikit-learn's Lasso function.
    """
    # Fit a Lasso model using scikit-learn's Lasso function
    lasso_model = Lasso(alpha=alpha, fit_intercept=False) #, selection='random', max_iter=1000)
    lasso_model.fit(X, y)

    # Get the coefficients of the Lasso model
    beta = lasso_model.coef_

    return beta, lasso_model

def aic_selection(X: NDArray[np.float64], y: NDArray[np.float64], train_func: Callable, alpha_vals: list[float], log_path: str=None) -> tuple[float, NDArray[np.float64] | Any]:
    """
    Fits a Lasso regression model for each alpha value and selects the one with the lowest BIC.
    """
    # Small constant to avoid log(0)
    eps = 1e-10

    # Initialize the best BIC and corresponding alpha value
    best_aic = np.inf
    best_alpha = None
    best_model = None

    # Loop over the alpha values
    for alpha in alpha_vals:
        beta, curr_model = train_func(X, y, alpha)

        # Calculate AIC
        predy = X @ beta
        residuals = y - predy
        RSS = np.sum(residuals**2)

        k = np.count_nonzero(beta)
        n = X.shape[0] # number of samples

        aic_value = 2*k  + n*np.log(RSS + eps / n)

        # Set the model with the lowest aic as the best model
        if aic_value < best_aic:
            best_aic = aic_value
            best_alpha = alpha
            best_model = curr_model

        # Calculate correlation metrics for log file and terminal output
        pearson_correlation = r2_score(y, predy)
        spearman_correlation = stats.spearmanr(y, predy).statistic

        # Output to log file
        if log_path:
            with open(log_path, "a+") as log_file:
                log_file.write(f"Trained with alpha = {alpha}. AIC = {aic_value}, pearson_c = {pearson_correlation}, spearman_c = {spearman_correlation}")

        # Output to terminal
        print(f"Trained with alpha = {alpha}. AIC = {aic_value}, pearson_c = {pearson_correlation}, spearman_c = {spearman_correlation}", flush=True)

    return best_alpha, best_model.coef_

class Subject:
    """
    Represents a single subject.
    """
    def __init__(self, subject_id: int, sc: NDArray[np.float64], fc: NDArray[np.float64], transform: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]) -> None:
        self.subject_id = subject_id
        self.sc = sc
        self.fc = fc
        self.transformed_sc, self.transformed_fc = transform(sc, fc)

    def calc_predicted_fc(self, rules: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Make prediction for the subject with the given rules.
        """
        return self.sc @ rules @ self.sc

class GroupLevelModel:
    """
    Represents a group of subjects.
    """
    def __init__(self, subjects: list[int]) -> None:
        self.subjects = subjects

    def _preprocess_data(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Preprocess the data for training.
        """
        transformed_scs = [subject.transformed_sc for subject in self.subjects]
        transformed_fcs = [subject.tranformed_fc for subject in self.subjects]

        transformed_sc_stack = np.vstack(transformed_scs)
        transformed_fc_stack = np.hstack(transformed_fcs)
        return transformed_sc_stack, transformed_fc_stack

    def train_group(self, train_func: Callable[[NDArray[np.float64], NDArray[np.float64]], Any]) -> Any:
        """
        Train the group model with algebraic linear regression.
        """
        transformed_sc_stack, transformed_fc_stack = self._preprocess_data()
        return train_func(transformed_sc_stack, transformed_fc_stack)
