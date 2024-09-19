import numpy as np
import bct

from lin_gen_model import Subject, GroupLevelModel
from data_manager import check_paths
    

class GenerateNulls(GroupLevelModel):
    """
    Use to generate group and subject level null fc matrices and null
    rule matrices. Input subject_ids, number of desired fc nulls per
    subject, and the output base directory.
    """
    def __init__(self, subject_ids, num_nulls, outpath, tumor_ds=False, pre_resection=True, log=False, log_base=10):
        super().__init__(subject_ids, outpath, tumor_ds=tumor_ds, pre_resection=pre_resection, log=log, log_base=log_base)
        self.fc_nulls_path = self.output_path + 'fc_nulls/'
        self.rule_nulls_path = self.output_path + 'rule_nulls/'
        self.num_nulls = num_nulls
    
    def fc_randomization(self, subject_id, subject_ids_dict):
        """
        Randomizes the FC matrix for the given subject id. Creates 100 of these
        randomized matrices and outputs them to the given directory.
        """
        # Define the given subject. One is added because subjects start at 1
        subject = self.subjects[subject_ids_dict[subject_id]]

        # instantiate the output directories
        fc_nulls_outdir = self.fc_nulls_path + f'{subject_id}/'
        # If output directories do not exist, create them
        check_paths(fc_nulls_outdir)

        # Repeat fc randomization 100 times
        for null_num in range(self.num_nulls):
            # Randomize the fc matrix for the subject
            randomized_data, R = bct.null_model_und_sign(subject.fc_matrix, bin_swaps=10, wei_freq=1)

            # Save the randomized fc matrix to the ouput directory
            np.savetxt(fc_nulls_outdir + f'fc_null_{null_num}', randomized_data)
            print(f'null num {null_num}')

    def calculate_sl_null_rules(self, subject_id, subject_ids_dict):
        """
        Calculates subject level null_rules from the fc nulls and outputs them.
        """
        # Define the given subject.
        subject = self.subjects[subject_ids_dict[subject_id]]

        # Define inverse of subject SC matrix to be used later
        sc_matrix_inv = np.linalg.pinv(subject.sc_matrix)

        # instantiate the output directories
        rule_nulls_path = self.rule_nulls_path + f'{subject_id}/'
        # If output directories do not exist, create them
        check_paths(rule_nulls_path)

        # Iterate over loop once for each null fc matrix for the given subject
        for null_num in range(self.num_nulls):
            # Load the null fc matrix
            null_fc = self.get_fc_null(subject.subject_id, null_num)

            # Calculate the null rules
            print(f"null_rules {null_num}")
            null_rules = sc_matrix_inv @ null_fc @ sc_matrix_inv

            # Save the null rules to the output directory
            np.savetxt(rule_nulls_path + f'rule_nulls_{null_num}', null_rules)

    def calculate_gl_null_rules(self):
        """
        Calculates and outputs group level null rules.
        """
        # instantiate the output directories
        null_rules_path = self.rule_nulls_path
        # If output directories do not exist, create them
        check_paths(null_rules_path)

        # Iterate over each over each of the null numbers
        for null_num in range(self.num_nulls):
            # Instantiate two lists to hold the fc nulls of the null number for every subject 
            fc_nulls_to_stack = []

            # Add the fc null for each null num to the list to then stack.
            for subject in self.subjects:
                fc_null = self.fc_null_symmetric_mod(self.get_fc_null(subject.subject_id, null_num))
                fc_nulls_to_stack.append(fc_null)
            # Stack the fc nulls
            fc_nulls_stack = np.hstack(fc_nulls_to_stack)

            # Train the null rule set for the null_num.
            flat_null_rules = np.linalg.pinv(self.K_stack) @ fc_nulls_stack
            null_rules = self.inverse_symmetric_modification(flat_null_rules, mat_size=np.shape(self.subjects[1].sc_matrix))

            # Write the results to text file.
            np.savetxt(null_rules_path + f'gl_rule_nulls_{null_num}', null_rules)
            
def main(pre_resection):
    # Define the output directory

    ####################### TEMPORARY CODE #########################################
    if pre_resection:
        path = 'pre_resection/'
    else:
        path = 'post_resection/'
    ########################## TEMPORARY CODE ######################################

    # Define the ids for the subjects 
    subject_ids_dict = dict(zip([1, 2, 3, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 31], list(range(25))))

    # Instantiate the GenerateNulls object to generate the null rules.
    null_generator = GenerateNulls(subject_ids_dict.keys(), 100, path, tumor_ds=True, pre_resection=True)

    # Generate the subject level null rules for each subject and save to the output directory
    for subject_id in subject_ids_dict.keys():
        #null_generator.fc_randomization(subject_id, subject_ids_dict)
        #null_generator.calculate_sl_null_rules(subject_id, subject_ids_dict)
        null_generator.calculate_gl_null_rules()

if __name__ == "__main__":
    main(False)
