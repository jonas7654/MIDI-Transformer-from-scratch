#!/bin/bash



# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

current_dir=($pwd)

echo "Generating Tokenizer with BPE"
cd ../CUPY/models/utils/
python3 generate_tokenizer.py --vocab_sizes 512 1024 2048 4096 8192 10000 12000 14000 15000
cd $current_dir
