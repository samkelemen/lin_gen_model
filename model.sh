#!/bin/bash
#SBATCH --job-name=lin_gen
#SBATCH --partition=sixhour
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16g
#SBATCH --time=0-06:00:00
#SBATCH --output=logs_subject_level/%j.log

pwd; hostname; date

echo "Running on $SLURM_CPUS_PER_TASK cores"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load conda
conda activate /kuhpc/work/pleskac/s442k213/projects/lin_gen_model/lin_gen_env

export id=$1

#!/kuhpc/work/pleskac/s442k213/projects/lin_gen_model/lin_gen_env 
python - <<END_SCRIPT
from lin_gen_model import main2
import os

# Instantiate id.
id = os.environ['id']
id = int(id.replace('\r', ''))

# Train subjects [id]
main2(id, 'post_resection/')


END_SCRIPT

echo "Done!"