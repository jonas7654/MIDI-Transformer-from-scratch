srun --mem=12G --cpus-per-task=1 --gres gpu:1 --partition=exercise-eml --pty python ../CUPY/models/GoePT/Inference.py \
    --weights "/csghome/hpdc04/Transformer_Code/checkpoints/fancy-snowflake-51_60.json" \
    --vocab-file "/csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/tokenizer_1024.json" \
    --input "/csghome/hpdc04/Transformer_Code/test_folder" \
    --b 200 \
    --save-dir "/csghome/hpdc04/Transformer_Code/predicted_midi_files"
