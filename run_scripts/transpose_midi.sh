#!/bin/bash



# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

current_dir=($pwd)

echo "Pre-processing raw midi files"
cd ../CUPY/models/utils/
python3 transp_in_memory.py
cd $current_dir
