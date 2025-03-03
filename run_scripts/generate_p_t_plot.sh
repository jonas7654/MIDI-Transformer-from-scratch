p=$1
temperature=$2

srun --mem=12G --cpus-per-task=1 --gres gpu:1 --partition=exercise-eml --pty python ../CUPY/models/utils/top_p_T_plot.py 
