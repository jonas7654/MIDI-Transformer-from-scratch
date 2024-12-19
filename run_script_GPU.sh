#!/bin/bash
#SBATCH --gres gpu:2
#SBATCH --mem 5G
#SBATCH --cpus-per-task 1
#SBATCH --time 30:00
#SBATCH -p exercise-eml
#SBATCH -o model_py_GPU.log


# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

echo "Running Training (with CUPY)"
python CUPY/models/GoePT/model.py --data-dir /csghome/hpdc04/Transformer_Code/CUPY/datasets/tokenized --vocab-file /csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/goe_pt/goe_pt_tokenizer.json --checkpoint-dir /csghome/hpdc04/Transformer_Code/checkpoints

