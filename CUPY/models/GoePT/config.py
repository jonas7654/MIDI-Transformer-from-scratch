# config.py
from miditok import REMI, Structured

# Paths
data_dir = "/csghome/hpdc04/Transformer_Code/CUPY/models/datasets/tokenized/"  # Path to the dataset directory
checkpoint_dir = "/csghome/hpdc04/Transformer_Code/checkpoints/"  # Path to save model checkpoints

tokenizer_name = REMI
tokenizer_name_str = tokenizer_name.__name__
vo_size = 2048

"""
The following section is only important after generating a tokenizer
"""
vocab_file = f"/csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/tokenizer_{tokenizer_name_str}_{vo_size}.json"  # Path to the tokenizer vocabulary file


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

batch_size = 10  # Default batch size
context_length = 384 # Sequence context length
epochs = 500  # Number of epochs to train
gradient_accumulation_steps = 32  # Steps for gradient accumulation
learning_rate = 0.0005 # Initial learning rate
dropout_rate = 0.2 # Default dropout rate
n_layer = 6  # Number of layers in the transformer
n_embd = 512  # Embedding size
n_heads = 8  # Number of attention heads
seed = 1  # Random seed

# Logging
log_interval = 5  # Number of steps between log updates
eval_interval = 5  # Number of steps between evaluations
eval_iters = 200 