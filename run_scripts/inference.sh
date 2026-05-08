p=$1
temperature=$2

srun --mem=12G --cpus-per-task=1 --gres gpu:1 --partition=brook --pty python ../CUPY/models/GoePT/Inference.py \
  --weights "../checkpoints_fine_tuning/fine_tuning_dedicated-quiver-27_650_650_0.json" \
  --vocab-file "../CUPY/models/tokenizers/tokenizer_REMI_4096_FULL_False.json" \
  --input "../Inference/input_files/" \
  --b 256 --save-dir "../Inference/output_files/" \
  --manually-set-sos-eos-trunc true \
  --p $p \
  --temperature $temperature
