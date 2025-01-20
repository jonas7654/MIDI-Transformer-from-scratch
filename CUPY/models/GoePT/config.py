# config.py

# Paths
data_dir = "/csghome/hpdc04/Transformer_Code/CUPY/models/datasets/tokenized/"  # Path to the dataset directory
checkpoint_dir = "/csghome/hpdc04/Transformer_Code/checkpoints/"  # Path to save model checkpoints
vocab_file = "/csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/tokenizer_512.json"  # Path to the tokenizer vocabulary file

# Special tokens for the tokenizer
manually_set_sos_eos_trunc = True

# Training parameters
batch_size = 6  # Default batch size
context_length = 1024  # Sequence context length
epochs = 14  # Number of epochs to train
gradient_accumulation_steps = 32  # Steps for gradient accumulation
learning_rate = 0.1  # Initial learning rate
dropout_rate = 0.2  # Default dropout rate
n_layer = 6  # Number of layers in the transformer
n_embd = 256  # Embedding size
n_heads = 4  # Number of attention heads
seed = 1  # Random seed

# Logging
log_interval = 5  # Number of steps between log updates
eval_interval = 5  # Number of steps between evaluations
eval_iters = 200 