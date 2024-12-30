"""
The main module of the project.
"""
import os
import numpy as np

from lin_gen_model import Subject, GroupLevelModel, \
    calc_alpha_grid, bic_selection, symmetric_modification, \
    inverse_symmetric_modification, \
    bayesian_lasso_regression, algebraic_linear_regression
import inout


def train_group(subject_ids, output_path):
    """
    Trains a group of subjects.
    """
    # Create a list of subject objects.
    subjects = []
    for subject_id in subject_ids:
        # Load the data.
        sc = inout.get_pre_resection_sc(subject_id)
        fc = inout.get_pre_resection_fc(subject_id)
        dim = sc.shape[0]
        # Instantiate a subject object.
        subject = Subject(subject_id, sc, fc, symmetric_modification)
        # Add the subject to the list.
        subjects.append(subject)

    # Create a group-level model object.
    group_model = GroupLevelModel(subjects)

    # Train the group model.
    rules = inverse_symmetric_modification(group_model.train_group(algebraic_linear_regression), \
                                           dim)

    # Save the model.
    np.savetxt(f"{output_path}/rules", rules)

    # Return the rules.
    return rules


def main():
    """
    Main function.
    """
    # pylint: disable=invalid-name
    TUMOR_IDS = (1, 2, 3, 5, 6, 7, 8, 10, 13, 15, 16, 17, 20, 24, 25, 26, 28)
    #SHAEFER_IDS = [1, 2, 3, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, \
    #          19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 31]

    # Get the group rules.
    group_rules = np.loadtxt("pre_resection_train_output/gl/matrices/fitted_O")

    # Train the subjects individually.
    for subject_id in TUMOR_IDS[:1]: # pylint: disable=invalid-name
        # Load the data.
        sc = inout.get_pre_resection_sc(subject_id)
        fc = inout.get_pre_resection_fc(subject_id)

        # Instantiate a subject object.
        subject = Subject(subject_id, sc, fc, symmetric_modification)

        # Train the model.
        alpha_grid = calc_alpha_grid(subject.transformed_sc, subject.transformed_fc)
        alpha, rules = bic_selection(subject.transformed_sc, subject.transformed_fc, \
                                     bayesian_lasso_regression, alpha_grid, group_rules)

        print(f"Optimal alpha for subject {subject_id} is {alpha}.")

        # Save the model.
        np.savetxt(f"bayesian_model/rules{subject_id:{2}}", rules)

if __name__ == "__main__":
    main()
