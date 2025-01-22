# config.py
from miditok import REMI, Structured

# Paths
data_dir = "/csghome/hpdc04/Transformer_Code/CUPY/models/datasets/tokenized/"  # Path to the dataset directory
checkpoint_dir = "/csghome/hpdc04/Transformer_Code/checkpoints/"  # Path to save model checkpoints

tokenizer_name = Structured
tokenizer_name_str = tokenizer_name.__name__

vocab_file = f"/csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/tokenizer_{tokenizer_name_str}_2048.json"  # Path to the tokenizer vocabulary file


# Special tokens for the tokenizer
manually_set_sos_eos_trunc = True

# Training parameters
batch_size = 16  # Default batch size
context_length = 512  # Sequence context length
epochs = 120  # Number of epochs to train
gradient_accumulation_steps = 32  # Steps for gradient accumulation
learning_rate = 0.08 # Initial learning rate
dropout_rate = 0.1  # Default dropout rate
n_layer = 8  # Number of layers in the transformer
n_embd = 512  # Embedding size
n_heads = 8  # Number of attention heads
seed = 1  # Random seed

# Logging
log_interval = 5  # Number of steps between log updates
eval_interval = 5  # Number of steps between evaluations
eval_iters = 200 