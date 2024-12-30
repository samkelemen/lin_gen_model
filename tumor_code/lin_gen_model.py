"""
Contains the functions to train the model.
"""

import math
import numpy as np
import pymc as pm
from sklearn.linear_model import Lasso
from statsmodels.regression.linear_model import OLS
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

from transformations import symmetric_modification, inverse_symmetric_modification


def algebraic_linear_regression(X, y):
    """
    Fits a linear regression model using algebraic linear regression.
    """
    return np.linalg.pinv(X) @ y

def calc_alpha_grid(X, y):
    """
    Calculates alpha grid as described in: 'Regularization Paths for 
    Generalized Linear Models via Coordinate Descent' by Friedman, 
    Hastie, and Tibshirani.
    """
    inner_products = []
    N = len(y)
    for row in X.T:
        inner_products.append(np.dot(row, y))
    alpha_max = max(inner_products) / N
    alpha_min = 0.0001 * alpha_max
    alphas = np.logspace(math.log(alpha_min), math.log(alpha_max), num=100, base=math.exp(1))
    return alphas

def lasso_regression(X, y, alpha):
    """
    Fits a Lasso regression model using scikit-learn's Lasso function.
    """
    # Fit a Lasso model using scikit-learn's Lasso function
    lasso_model = Lasso(alpha=alpha, fit_intercept=False)
    lasso_model.fit(X, y)

    # Get the coefficients of the Lasso model
    beta_opt = lasso_model.coef_
    return beta_opt, lasso_model

def bic_selection(X, y, train_func, alpha_vals, priors=None):
    """
    Fits a Lasso regression model for each alpha value and selects the one with the lowest BIC.
    """
    # Initialize the best BIC and corresponding alpha value
    best_bic = np.inf
    best_alpha = None
    best_model = None

    # Loop over the alpha values
    for alpha in alpha_vals:
        try:
            if priors is not None:
                beta_opt, curr_model = train_func(X, y, priors, alpha)
            else:
                beta_opt, curr_model = train_func(X, y, alpha)

            # Create a new sc matrix with only the selected features (non-zero coefficients)
            selected_features = np.where(beta_opt != 0)[0]
            X_selected = X[:, selected_features]

            # Fit an OLS model using statsmodels to get the BIC
            ols_model = OLS(X, X_selected).fit()
            bic_value = ols_model.bic

            # Check if this alpha value results in a lower BIC
            if bic_value < best_bic:
                best_bic = bic_value
                best_alpha = alpha
                best_model = curr_model

            print(f"Trained with alpha = {alpha}.", flush=True)
        except: # pylint: disable=bare-excep
            print(f"Training with alpha, {alpha}, failed. Moving to next alpha value.", flush=True)


    # Convert the model output to O matrix and return the best alpha and rule set.
    best_solution = best_model.coef_
    return best_alpha, best_solution

def bayesian_lasso_regression(X, y, beta_true, alpha):
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
    mcmc = MCMC(nuts_kernel, num_samples=2000, warmup_steps=1000)
    mcmc.run(torch.tensor(X), torch.tensor(y))
    
    # Extract posterior samples for beta
    posterior_samples = mcmc.get_samples()['beta']
    
    # Calculate the mean of the posterior samples for each coefficient
    opt_beta = posterior_samples.mean(dim=0).numpy()

    # Return the optimal beta and the model
    return opt_beta, mcmc


def bayesian_lasso_regression_pymc(X, y, beta_true, alpha):
    """
    Fit a Bayesian GLM using PyMC3 with a Lasso penalty.
    
    Parameters
    ----------
    X : array-like
        Feature matrix (n_samples, n_features).
    y : array-like
        Response vector (n_samples,).
    beta_true : array-like
        Known true coefficients (n_features,).
    lambda_value : float
        The regularization strength for the Lasso penalty.

    Returns
    -------
    posterior_samples : pymc3 trace object
        Contains the posterior samples of the model parameters.
    """
    # Define the model in PyMC
    with pm.Model() as model:
        # Define the Laplace prior for the coefficients (Lasso)
        beta = pm.Laplace('beta', mu=beta_true, b=alpha, shape=len(beta_true))

        # Likelihood: Linear model y = X * beta + noise
        mu = pm.math.dot(X, beta)
        sigma = pm.HalfNormal('sigma', sigma=0.2)#Noise stanadard deviation (sigma)

        # Likelihood function for observed data (Normal distribution)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

        # Sampling: Run MCMC to draw posterior samples
        trace = pm.sample(2000, tune=1000, return_inferencedata=True)

    # Extract posterior samples for beta
    posterior_samples = trace.posterior['beta']

    # Calculate the mean of the posterior samples for each coefficient
    means = []

    # Iterate over the parameters in the trace and calculate the mean of each
    for param in trace.posterior.data_vars:
        # Calculate the mean of the posterior samples for this parameter
        means.append(np.mean(trace.posterior[param].values))

    # Set the mean of the posterior samples to beta
    opt_beta = np.array(means)

    # Return the optimal beta and the model
    return opt_beta, model

class Subject:
    """Represents a single subject."""
    def __init__(self, subject_id, sc, fc, transform):
        self.subject_id = subject_id
        self.sc = sc
        self.fc = fc
        self.transformed_sc, self.transformed_fc = transform(sc, fc)

    def calc_predicted_fc(self, rules):
        """Make prediction for the subject with the given rules."""
        return self.sc @ rules @ self.sc

class GroupLevelModel:
    """Represents a group of subjects."""
    def __init__(self, subjects):
        self.subjects = subjects

    def _preprocess_data(self):
        """Preprocess the data for training."""
        transformed_scs = [subject.transformed_sc for subject in self.subjects]
        transformed_fcs = [subject.tranformed_fc for subject in self.subjects]

        transformed_sc_stack = np.vstack(transformed_scs)
        transformed_fc_stack = np.hstack(transformed_fcs)
        return transformed_sc_stack, transformed_fc_stack

    def train_group(self, train_func):
        """Train the group model with algebraic linear regression."""
        transformed_sc_stack, transformed_fc_stack = self._preprocess_data()
        return train_func(transformed_sc_stack, transformed_fc_stack)