#!/bin/bash



# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

current_dir=($pwd)

echo "Generating Training, Validation and Test tokenized datasets"
cd ../CUPY/models/utils/
python3 generate_sets_ONLY_TESTING.py
cd $current_dir
