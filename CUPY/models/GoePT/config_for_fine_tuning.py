# config.py
from miditok import REMI, Structured

# Paths
data_dir = "/csghome/hpdc04/Transformer_Code/CUPY/models/datasets/tokenized/"  # Path to the dataset directory
checkpoint_dir = "/csghome/hpdc04/Transformer_Code/checkpoints/"  # Path to save model checkpoints

tokenizer_name = REMI
tokenizer_name_str = tokenizer_name.__name__
# Full specifies if we should use the tokenizer which was trained on the clean + unclean midi dataset
FULL = False #:NOTE Only important for training not for generating datasets via the run scripts



# Special tokens for the tokenizer
manually_set_sos_eos_trunc = True #:NOTE At the moment we do not have a TRUNC token!

# Training parameters

# REGULARIZATION
regularization = False
reg_alpha = 2.5
#

# Relative Attention
relative_attention = True
#####

#batch_size = 8  # Default batch size
epochs = 10000  # Number of epochs to train
gradient_accumulation_steps = 32  # Steps for gradient accumulation
learning_rate = 0.0005 # Initial learning rate
dropout_rate = 0.3 # Default dropout rate
seed = 1  # Random seed

# Logging
log_interval = 50  # Number of steps between log updates
eval_interval = 50  # Number of steps between evaluations
eval_iters = 100 