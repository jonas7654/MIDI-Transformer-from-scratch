#!/bin/bash



# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

current_dir=($pwd)

cd ../CUPY/models/utils/
python3 token_to_event.py 
cd $current_dir
