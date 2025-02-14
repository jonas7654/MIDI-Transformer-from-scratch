#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 16G
#SBATCH --cpus-per-task 1
#SBATCH --time 2880:00
#SBATCH -p exercise-eml
#SBATCH -o logs/fine_tuning.log


# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

echo "Running Training (with CUPY)"

python ../CUPY/models/GoePT/fine_tune_model.py \
    --weights "/csghome/hpdc04/Transformer_Code/checkpoints/avid-energy-21_5100.json" \
    --vocab-file "/csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/tokenizer_REMI_8192_FULL_True.json"
                                     