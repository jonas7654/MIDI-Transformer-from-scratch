source /csghome/hpdc04/Transformer_Code/run_scripts/parameters.sh

echo "context length: $context_length"
echo "epochs: $epochs"
echo "batch size: $batch_size"
echo "eval_interval: $eval_interval"
echo "learning rate: $lr"
echo "log interval: $log_interval"
echo "dropout rate: $dropout"

srun --mem=5G --cpus-per-task=1 --gres gpu:2 --partition=exercise-eml --pty python ../CUPY/models/GoePT/train.py \
    --data-dir /csghome/hpdc04/Transformer_Code/CUPY/models/datasets/tokenized \
    --vocab-file /csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/tokenizer.json \
    --checkpoint-dir /csghome/hpdc04/Transformer_Code/checkpoints \
    --context-length "$context_length" \
    --epochs "$epochs" \
    --batch-size "$batch_size" \
    --eval-interval "$eval_interval" \
    --lr "$lr" \
    --log-interval "$log_interval" \
    --n-heads "$n_heads" \
    --dropout "$dropout"

