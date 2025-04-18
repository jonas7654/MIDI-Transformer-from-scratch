p=$1
temperature=$2

srun --mem=12G --cpus-per-task=1 --gres gpu:1 --partition=exercise-eml --pty python ../CUPY/models/GoePT/Inference.py \
    --weights "/csghome/hpdc04/Transformer_Code/checkpoints/loyal-chocolate-387_4125.json" \
    --vocab-file "/csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/tokenizer_REMI_4096_FULL_False.json" \
    --input "/csghome/hpdc04/Transformer_Code/Inference/input_files/" \
    --b 256\
    --save-dir "/csghome/hpdc04/Transformer_Code/Inference/output_files/" \
    --manually-set-sos-eos-trunc true \
    --p $p \
    --temperature $temperature
