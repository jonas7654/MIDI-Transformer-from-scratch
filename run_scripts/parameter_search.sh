#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 5G
#SBATCH --cpus-per-task 1
#SBATCH --time 500:00
#SBATCH -p exercise-eml
#SBATCH -o logs/parameter_search.log


# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

echo "Running Parameter Search (with wandb sweeps)"


python ../CUPY/models/GoePT/train_wandb_sweeps.py
                                     