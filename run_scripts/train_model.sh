#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 16G
#SBATCH --cpus-per-task 1
#SBATCH --time 1500:00
#SBATCH -p exercise-eml
#SBATCH -o logs/model_py_GPU.log


# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

echo "Running Training (with CUPY)"

python ../CUPY/models/GoePT/train.py 
                                     