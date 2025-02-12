#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --gres gpu:0
#SBATCH --mem 32G
#SBATCH --time 1000:00
#SBATCH -p exercise-eml
#SBATCH -o logs/tranpose.log


# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

current_dir=($pwd)

echo "Pre-processing raw midi files"
cd ../CUPY/models/utils/
python3 collect_mem_2.py
cd $current_dir
