#!/bin/bash



# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

current_dir=($pwd)

echo "Generating Tokenizer with BPE"
cd ../CUPY/models/utils/
python3 generate_tokenizer.py --vocab_sizes 4096
cd $current_dir
