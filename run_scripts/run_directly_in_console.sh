source parameters.sh

echo "context length: $context_length"
echo "epochs: $epochs"
echo "batch size: $batch_size"
echo "eval_interval: $eval_interval"
echo "learning rate: $lr"

srun --mem=5G --cpus-per-task=1 --gres gpu:2 --partition=exercise-eml --pty python ../CUPY/models/GoePT/model.py \
    --data-dir /csghome/hpdc04/Transformer_Code/CUPY/models/datasets/tokenized \
    --vocab-file /csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/tokenizer.json \
    --checkpoint-dir /csghome/hpdc04/Transformer_Code/checkpoints \
    --context-length "$context_length" \
    --epochs "$epochs" \
    --batch-size "$batch_size" \
    --eval-interval "$eval_interval" \
    --lr "$lr"

