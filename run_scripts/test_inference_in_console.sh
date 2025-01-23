srun --mem=12G --cpus-per-task=1 --gres gpu:1 --partition=exercise-eml --pty python ../CUPY/models/GoePT/Inference.py \
    --weights "/csghome/hpdc04/Transformer_Code/checkpoints/likely-durian-145_150.json" \
    --vocab-file "/csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/tokenizer_Structured_1024.json" \
    --input "/csghome/hpdc04/Transformer_Code/test_folder" \
    --b 200 \
    --save-dir "/csghome/hpdc04/Transformer_Code/predicted_midi_files" \
    --manually-set-sos-eos-trunc true
