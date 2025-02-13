# config.py
from miditok import REMI, Structured

# Paths
data_dir = "/csghome/hpdc04/Transformer_Code/CUPY/models/datasets/tokenized/"  # Path to the dataset directory
checkpoint_dir = "/csghome/hpdc04/Transformer_Code/checkpoints/"  # Path to save model checkpoints

tokenizer_name = REMI
tokenizer_name_str = tokenizer_name.__name__
vo_size = 8192
# Full specifies if we should use the tokenizer which was trained on the clean + unclean midi dataset
FULL = True #:NOTE Only important for training not for generating datasets via the run scripts

vocab_file = f"/csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/tokenizer_{tokenizer_name_str}_{vo_size}_FULL_{FULL}.json"  # Path to the tokenizer vocabulary file


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

batch_size = 12  # Default batch size
context_length = 256 # Sequence context length
epochs = 10000  # Number of epochs to train
gradient_accumulation_steps = 32  # Steps for gradient accumulation
learning_rate = 0.0001 # Initial learning rate
dropout_rate = 0.3 # Default dropout rate
n_layer = 4  # Number of layers in the transformer
n_embd = 1024  # Embedding size
n_heads = 16 # Number of attention heads
seed = 1  # Random seed

# Logging
log_interval = 50  # Number of steps between log updates
eval_interval = 50  # Number of steps between evaluations
eval_iters = 100 