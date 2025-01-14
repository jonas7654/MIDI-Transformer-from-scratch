#!/bin/bash
#SBATCH --gres gpu:2
#SBATCH --mem 5G
#SBATCH --cpus-per-task 1
#SBATCH --time 500:00
#SBATCH -p exercise-eml
#SBATCH -o model_py_GPU.log


# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

context_length=128
epochs=32
batch_size=32 # default is 16
eval_interval=1 # How frequent should the validation loss be calculated and checkpoints be saved

echo "Running Training (with CUPY)"

echo "context length: $context_length"
echo "epochs: $epochs"
echo "batch size: $batch_size"
echo "eval_interval: $eval_interval"


python ../CUPY/models/GoePT/model.py \
    --data-dir /csghome/hpdc04/Transformer_Code/CUPY/models/datasets/tokenized \
    --vocab-file /csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/tokenizer.json \
    --checkpoint-dir /csghome/hpdc04/Transformer_Code/checkpoints \
    --context-length "$context_length" \
    --epochs "$epochs" \
    --batch-size "$batch_size" \
    --eval-interval "$eval_interval"
                                     