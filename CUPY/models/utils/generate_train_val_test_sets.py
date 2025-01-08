from random import shuffle
import os
import numpy as np
from icecream import ic
from pathlib import Path

from miditok import REMI, TokenizerConfig  # here we choose to use REMI
from miditok.data_augmentation import augment_dataset
from miditok.utils import split_files_for_training

current_dir = os.getcwd()
data_path = os.path.join(os.path.dirname(current_dir) , "datasets")
files_path = list(Path(data_path, "midi_1/").glob("**/*.mid")) # NOTE: Specify the right directory for the raw data
tokenizer_path = os.path.join(os.path.dirname(current_dir), "tokenizers/")

seq_length = 256

# Load the pre-trained tokenizer
tokenizer = REMI(params = Path(tokenizer_path, "tokenizer.json"))
ic(tokenizer.vocab_size)

# Split the dataset into train/valid/test subsets, with 15% of the data for each of the two latter
midi_paths = list(Path(data_path).glob("**/*.mid"))
total_num_files = len(midi_paths)
num_files_valid = round(total_num_files * 0.15)
num_files_test = round(total_num_files * 0.15)
shuffle(midi_paths)
midi_paths_valid = midi_paths[:num_files_valid]
midi_paths_test = midi_paths[num_files_valid:num_files_valid + num_files_test]
midi_paths_train = midi_paths[num_files_valid + num_files_test:]

# Chunk MIDIs and perform data augmentation on each subset independently
if(True == False): # THIS IS HORRIBLE BUT FOR NOW ITS OK :TODO
    for files_paths, subset_name in (
        (midi_paths_train, "train"), (midi_paths_valid, "valid"), (midi_paths_test, "test")
    ):

        # Split the MIDIs into chunks of sizes approximately about 1024 tokens
        subset_chunks_dir = Path(f"{data_path}/dataset_{subset_name}")
        split_files_for_training(
            files_paths=files_paths,
            tokenizer=tokenizer,
            save_dir=subset_chunks_dir,
            max_seq_len=seq_length,
            num_overlap_bars=2,
        )

        # Perform data augmentation
        augment_dataset(
            subset_chunks_dir,
            pitch_offsets=[-12, 12],
            velocity_offsets=[-4, 4],
            duration_offsets=[-0.5, 0.5],
        )    
        
    
    
    
    
"""
@Author: Jonas
Here I implement the data generating process where we save the integer ids for the tokens 
as a numpy matrix and then save it as binary in order to feed it to our GoePT model.

"""
output_path = os.path.join(data_path, "tokenized")

def collator(input, seq_length):
    if (len(input) < seq_length):
        result =  input + [0] * (seq_length - len(input)) 
        return np.array(result, dtype=np.uint16)
    # if the input length is greatet than seq_len, truncate!
    return np.array(input[:seq_length], dtype=np.uint16)
        
    
    
    
# Create train, val, test token datasets


for subset_name in ("train", "valid", "test"):
    print(f"{data_path}/dataset_{subset_name}")
    files_paths = list(Path(f"{data_path}/dataset_{subset_name}").glob("**/*.mid"))
    dataset_tokenized = np.zeros((len(files_paths), seq_length))
    
    # Iterate over all midi files and tokenize them
    for i, midi_file in enumerate(files_paths):
        midi_file_tokenized = tokenizer(Path(midi_file))[0].ids        
        midi_processed = collator(midi_file_tokenized, seq_length)
    
        dataset_tokenized[i, :] = midi_processed
    
    # Sanity check
    assert np.all(dataset_tokenized < tokenizer.vocab_size), "Found out-of-vocabulary tokens in dataset"
    print(f"{subset_name}:")
    ic(dataset_tokenized[:100])

    # SAVE
    save_dir = os.path.join(output_path, f"{subset_name}.bin")
    Path(output_path).mkdir(parents = True, exist_ok = True)
    dataset_tokenized.astype(np.uint16).tofile(save_dir)

ic(tokenizer.vocab_size)