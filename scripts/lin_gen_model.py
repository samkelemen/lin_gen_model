"""
Contains the functions to train the model.
"""
from collections.abc import Callable
from typing import Any
import math
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Lasso
from statsmodels.regression.linear_model import OLS
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS


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

def bic_selection(X: NDArray[np.float64], y: NDArray[np.float64], train_func: Callable, alpha_vals: list[float], priors: NDArray[np.float64]=None) -> tuple[float, NDArray[np.float64] | Any]:
    """
    Fits a Lasso regression model for each alpha value and selects the one with the lowest BIC.
    """
    # Initialize the best BIC and corresponding alpha value
    best_bic = np.inf
    best_alpha = None
    best_model = None

    # Loop over the alpha values
    for alpha in alpha_vals:
        beta, curr_model = train_func(X, y, priors, alpha) if priors else train_func(X, y, alpha)

        # Create a new sc matrix with only the selected features (non-zero coefficients)
        selected_features = np.where(beta != 0)[0]
        X_selected = X[:, selected_features]

        # Fit an OLS model using statsmodels to get the BIC
        num_selected_features = np.shape(X_selected)[0]

        if num_selected_features > 0:
            ols_model = OLS(X, X_selected).fit()
            bic_value = ols_model.bic

            # Check if this alpha value results in a lower BIC
            if bic_value < best_bic:
                best_bic = bic_value
                best_alpha = alpha
                best_model = curr_model

            print(f"Trained with alpha = {alpha}. BIC = {bic_value}", flush=True)
        else:
            print(f"Trained with alpha = {alpha}. No features selected.", flush=True)

        # Check if this alpha value results in a lower BIC
        if bic_value < best_bic:
            best_bic = bic_value
            best_alpha = alpha
            best_model = curr_model

        print(f"Trained with alpha = {alpha}.", flush=True)

    # Convert the model output to O matrix and return the best alpha and rule set.
    if priors:
        posterior_samples = best_model.get_samples()
        best_solution = posterior_samples['beta'].mean(dim=0).numpy()
    else:
        best_solution = best_model.coef_
    return best_alpha, best_solution

def bayesian_lasso_regression(X: NDArray[np.float64], y: NDArray[np.float64], beta_true: NDArray[np.float64], alpha: int) -> tuple[NDArray[np.float64] | Any, NDArray[np.float64] | Any]:
    """
    Fit a Bayesian GLM using Pyro with a Lasso penalty.

    Parameters
    ----------
    X : array-like
        Feature matrix (n_samples, n_features).
    y : array-like
        Response vector (n_samples,).
    beta_true : array-like
        Known true coefficients (n_features,).
    alpha : float
        The regularization strength for the Lasso penalty.

    Returns
    -------
    posterior_samples : pyro trace object
        Contains the posterior samples of the model parameters.
    """
    # Define the model in Pyro
    def model(X, y):
        # Define the Laplace prior for the coefficients (Lasso)
        beta = pyro.sample('beta', dist.Laplace(torch.tensor(beta_true), torch.tensor(alpha)))
        
        # Likelihood: Linear model y = X * beta + noise
        mu = torch.matmul(X, beta)
        sigma = pyro.sample('sigma', dist.HalfNormal(torch.tensor(0.2)))  # Noise standard deviation (sigma)
        
        # Likelihood function for observed data (Normal distribution)
        with pyro.plate('data', len(y)):
            pyro.sample('obs', dist.Normal(mu, sigma), obs=y)
    
    # Using NUTS for MCMC sampling
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=100)
    mcmc.run(torch.tensor(X), torch.tensor(y))
    
    # Extract posterior samples for beta
    posterior_samples = mcmc.get_samples()['beta']
    
    # Calculate the mean of the posterior samples for each coefficient
    opt_beta = posterior_samples.mean(dim=0).numpy()

    # Return the optimal beta and the model
    return opt_beta, mcmc

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
