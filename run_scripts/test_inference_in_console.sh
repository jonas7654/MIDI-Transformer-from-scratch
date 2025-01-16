source parameters.sh
srun --mem=5G --cpus-per-task=1 --gres gpu:2 --partition=exercise-eml --pty python ../CUPY/models/GoePT/Inference.py \
    --weights "/csghome/hpdc04/Transformer_Code/checkpoints/goe_pt_iter_75.json" \
    --vocab-file "/csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/tokenizer.json" \
    --input "/csghome/hpdc04/Transformer_Code/test_folder" \
    --context-length $context_length
