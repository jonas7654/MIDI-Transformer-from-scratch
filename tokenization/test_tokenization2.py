from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from torch.utils.data import DataLoader
from pathlib import Path

# Creating a multitrack tokenizer, read the doc to explore all the parameters
config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
tokenizer = REMI(config)

# Train the tokenizer with Byte Pair Encoding (BPE)
files_paths = list(Path("/home/jv/Desktop/MIDI Dataset EDM").glob("**/*.mid"))
tokenizer.train(vocab_size=8192, files_paths=files_paths)
tokenizer.save(Path("/home/jv/GitHub/Transformer_Code/tokenization/dump/tokenizer.json/", "tokenizer.json"))

# Split MIDIs into smaller chunks for training
dataset_chunks_dir = Path("/home/jv/GitHub/Transformer_Code/tokenization/dump", "midi_chunks")
split_files_for_training(
    files_paths=files_paths,
    tokenizer=tokenizer,
    save_dir=dataset_chunks_dir,
    max_seq_len=32,
)


# manual
import numpy as np
files_paths=list(dataset_chunks_dir.glob("**/*.mid"))
seq_len = 8
dataset_tokenized = np.zeros((len(files_paths), seq_len))

for i in range(len(files_paths)):
    t = tokenizer(files_paths[i]).ids
    k = np.zeros(seq_len)
    k[0:len(t)] = t
    dataset_tokenized[i, :] = k

import os
print(dataset_tokenized)
out_dir = "/home/jv/Desktop/NN/datasets/tokenized/"
np.save("/home/jv/Desktop/NN/datasets/tokenized/dataset_tokenized", dataset_tokenized)

os.path.join(out_dir, "dataset_tokenized".replace('.txt', '.bin'))