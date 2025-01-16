#!/bin/bash
#SBATCH --gres gpu:2
#SBATCH --mem 5G
#SBATCH --cpus-per-task 1
#SBATCH --time 500:00
#SBATCH -p exercise-eml
#SBATCH -o logs/model_py_GPU.log


# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

# load model parameters
source /csghome/hpdc04/Transformer_Code/run_scripts/parameters.sh

echo "Running Training (with CUPY)"

echo "context length: $context_length"
echo "epochs: $epochs"
echo "batch size: $batch_size"
echo "eval_interval: $eval_interval"
echo "learning rate: $lr"
echo "number of heads: $n_heads"


python ../CUPY/models/GoePT/train.py \
    --data-dir /csghome/hpdc04/Transformer_Code/CUPY/models/datasets/tokenized \
    --vocab-file /csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/tokenizer.json \
    --checkpoint-dir /csghome/hpdc04/Transformer_Code/checkpoints \
    --context-length "$context_length" \
    --epochs "$epochs" \
    --batch-size "$batch_size" \
    --eval-interval "$eval_interval" \
    --lr "$lr" \
    --n-heads "$n_heads"
                                     