"""
Test file for lin_gen_model.py
"""
import numpy as np

from lin_gen_model import Subject, GroupLevelModel, \
    calc_alpha_grid, bic_selection, symmetric_modification, \
    inverse_symmetric_modification, \
    bayesian_lasso_regression, algebraic_linear_regression


# Get the group rules.
group_rules = np.random.randint(1, 42, (42, 42))

# Get only the upper triangular part of the matrix, as a vector
priors = group_rules[np.triu_indices(np.shape(group_rules)[0], k = 0)]

# Get the sc and fc matrices
sc = np.random.randint(1, 10, (42, 42))
fc = np.random.randint(-100, 100, (42, 42)) * 1/100
subject_id = 1

# We need the data to be floats for torch
sc = sc.astype(np.float64)
fc = fc.astype(np.float64)
priors = priors.astype(np.float64)

# Instantiate the subject object.
subject = Subject(subject_id, sc, fc, symmetric_modification)

# Train the model.
alpha_grid = calc_alpha_grid(subject.transformed_sc, subject.transformed_fc)
alpha, rules = bic_selection(subject.transformed_sc, subject.transformed_fc, \
                                bayesian_lasso_regression, alpha_grid, priors)
