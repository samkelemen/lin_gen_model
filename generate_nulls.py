import numpy as np
import bct

from lin_gen_model import Subject, Group
from data_manager import check_paths
    

class GenerateNulls(Group):
    """
    Use to generate group and subject level null fc matrices and null
    rule matrices. Input subject_ids, number of desired fc nulls per
    subject, and the output base directory.
    """
    def __init__(self, subject_ids, num_nulls, outpath, log=False, log_base=10):
        super().__init__(subject_ids, outpath, log=log, log_base=log_base)
        self.fc_nulls_path = self.output_path + 'fc_nulls/'
        self.rule_nulls_path = self.output_path + 'rule_nulls/'
        self.num_nulls = num_nulls
    
    def fc_randomization(self, subject_id):
        """
        Randomizes the FC matrix for the given subject id. Creates 100 of these
        randomized matrices and outputs them to the given directory.
        """
        # Define the given subject. One is added because subjects start at 1
        subject = self.subjects[subject_id - 1]

        # instantiate the output directories
        fc_nulls_outdir = self.fc_nulls_path + f'{subject_id}/'
        # If output directories do not exist, create them
        check_paths(fc_nulls_outdir)

        # Repeat fc randomization 100 times
        for null_num in range(self.num_nulls):
            # Randomize the fc matrix for the subject
            randomized_data, R = bct.null_model_und_sign(subject.B, bin_swaps=10, wei_freq=1)

            # Save the randomized fc matrix to the ouput directory
            np.savetxt(fc_nulls_outdir + f'fc_null_{null_num}', randomized_data)

    def calculate_sl_null_rules(self, subject_id):
        """
        Calculates subject level null_rules from the fc nulls and outputs them.
        """
        # Define the given subject. One is added because subjects start at 1
        subject = self.subjects[subject_id -1]

        # Define inverse of subject SC matrix to be used later
        sc_matrix_inv = np.linalg.pinv(subject.X)

        # instantiate the output directories
        rule_nulls_path = self.rule_nulls_path + f'{subject_id}/'
        # If output directories do not exist, create them
        check_paths(rule_nulls_path)

        # Iterate over loop once for each null fc matrix for the given subject
        for null_num in range(self.num_nulls):
            # Load the null fc matrix
            null_fc = self.get_fc_null(subject.subject_id, null_num)

            # Calculate the null rules
            print(np.shape(sc_matrix_inv), flush=True)
            print(np.shape(null_fc), flush=True)
            null_rules = sc_matrix_inv @ null_fc @ sc_matrix_inv

            # Save the null rules to the output directory
            np.savetxt(rule_nulls_path + f'rule_nulls_{null_num}', null_rules)

    def calculate_gl_null_rules(self):
        """
        Calculates and outputs group level null rules.
        """
        # Instantiate to use Subjects' methods.
        subject_methods = Subject(None)

        # instantiate the output directories
        null_rules_path = self.rule_nulls_path
        # If output directories do not exist, create them
        check_paths(null_rules_path)

        # Iterate over each over each of the null numbers
        for null_num in range(null_num):
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
            null_rules = subject_methods.inverse_symmetric_modification(flat_null_rules, mat_size=np.shape(self.subjects[1].X))

            # Write the results to text file.
            np.savetxt(null_rules_path + f'rule_nulls_{null_num}', null_rules)
            
def main():
    # Define the output directory
    path = 'sl/'
    # Define the ids for the subjects
    subject_ids = [num for num in range(1, 51)]
    # Instantiate the GenerateNulls object to generate the null rules.
    null_generator = GenerateNulls(subject_ids, 100, path)
    
    # Generate the subject level null rules for each subject and save to the output directory
    for subject_id in subject_ids:
        null_generator.calculate_sl_null_rules(subject_id)

if __name__ == '__main__':
    main()
