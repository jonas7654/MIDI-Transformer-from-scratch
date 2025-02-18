from random import shuffle, seed
import os, shutil
import numpy as np
from icecream import ic
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tokenize_data_fast import tokenize_dataset_to_bin

import sys
sys.path.insert(0, '/csghome/hpdc04/Transformer_Code/CUPY/models/GoePT/')
import config

from miditok import Structured, REMI, TokenizerConfig  # here we choose to use REMI
from miditok.data_augmentation import augment_dataset
from miditok.utils import split_files_for_training
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader

seed(256)

"""
@Author: Jonas
Here we specify the folder structure.

1) data_path is the path to the datasets folder located in CUPY/models/
2) name_of_midi_data is the acutal name of the folder which contains the raw midi data. 
3) tokenizer_path corresponds to the pre trained tokenizer
4) :TODO how should we set seq_length?

---------------------------------------------------------------

"""

current_dir = os.getcwd()
data_path = os.path.join(os.path.dirname(current_dir) , "datasets")
name_of_midi_data = "small_midi"

midi_paths = list(Path(data_path, name_of_midi_data).glob("*.mid")) 
tokenizer_path = os.path.join(os.path.dirname(current_dir), "tokenizers/")

# :NOTE SET THIS!
seq_length = config.context_length

# Load the pre-trained tokenizer
tokenizer = config.tokenizer_name(params = Path(config.vocab_file))
ic(tokenizer.vocab_size)
# Split the dataset into train/valid/test subsets, with 15% of the data for each of the two latter

total_num_files = len(midi_paths)
print(f"Total number of midi files: {total_num_files}")

num_files_valid = round(total_num_files * 0.15)
num_files_test = round(total_num_files * 0.15)
shuffle(midi_paths)
midi_paths_valid = midi_paths[:num_files_valid]
midi_paths_test = midi_paths[num_files_valid:num_files_valid + num_files_test]
midi_paths_train = midi_paths[num_files_valid + num_files_test:]

train_val_test_path = {}
 # Chunk MIDIs and perform data augmentation on each subset independently
for files_paths, subset_name in (
     (midi_paths_train, "train"), (midi_paths_valid, "val"), (midi_paths_test, "test")
 ):
        
     # Split the MIDIs into chunks of sizes approximately about 'seq_length' tokens
    
     subset_chunks_dir = Path(f"{data_path}/dataset_{subset_name}")
     
     # Check if the folder exists and if it's a directory
     if subset_chunks_dir.exists() and subset_chunks_dir.is_dir():
        # Remove the folder and all its contents
        shutil.rmtree(subset_chunks_dir)
        print(f"Folder '{subset_chunks_dir}' and its contents have been removed.")
    
    # Recreate the directory after deletion
        os.makedirs(subset_chunks_dir)
        print(f"Directory '{subset_chunks_dir}' has been recreated.")
         
     # Save the subset directory for later    
     train_val_test_path[subset_name] = subset_chunks_dir
    
        # Split files into chunks for training
     split_files_for_training(
         files_paths=files_paths,
         tokenizer=tokenizer,
         save_dir=subset_chunks_dir,
         max_seq_len=seq_length,
         num_overlap_bars=2,  # Adjust as needed
     )
    
     # Perform data augmentation
     do_augment = False
     if do_augment:
         augment_dataset(
             subset_chunks_dir,
             pitch_offsets=[-12, 12],
             velocity_offsets=[0, 0],
             duration_offsets=[0, 0],
         )
     


    
"""
@Author: Jonas
Monkey Patch tokenizer.tokenize_data_fast
"""

collator = DataCollator(tokenizer.pad_token_id)

# Loop through subsets (train/val/test)
for subset in train_val_test_path:
    print(f"Processing {subset} subset...")

    # Initialize a list to collect all tokenized sequences for this subset
    all_tokens = []

    # Get the MIDI files in the directory for this subset
    files_path = train_val_test_path[subset]
    midi_files = list(Path(files_path).glob("**/*.mid"))  # Find all .mid files recursively

    # Ensure midi_files is not empty
    if not midi_files:
        raise ValueError(f"No MIDI files found in the directory: {files_path}")

    # Create the dataset
    subset_dataset = DatasetMIDI(
        files_paths=midi_files,  # Pass the list of MIDI files
        tokenizer=tokenizer,
        max_seq_len=seq_length,
        bos_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer["BOS_None"],
    )
    subset_loader = DataLoader(dataset=subset_dataset, collate_fn=collator)

    # Iterate through the DataLoader and save tokens
    for batch in subset_loader:
        input_ids = batch["input_ids"]  # Extract input_ids
        all_tokens.extend(input_ids.tolist())  # Convert tensor to list and collect sequences

    # Convert to NumPy array and save as a binary file
    tokenized_data = np.array(all_tokens, dtype=np.uint16)
    save_dir = os.path.join(
        output_path,
        f"{subset}_tokenized_seq_len_{seq_length}.bin"
    )
    tokenized_data.tofile(save_dir)  # Save as binary file

    # Update size_dict with the size of this subset
    size_dict[subset] = tokenized_data.shape[0]
    print(f"Saved {subset} subset to {save_dir}, size: {size_dict[subset]} sequences.")
