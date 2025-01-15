from random import shuffle, seed
import os, shutil
import numpy as np
from icecream import ic
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from json_to_bin import json_to_bin

from miditok import REMI, TokenizerConfig  # here we choose to use REMI
from miditok.data_augmentation import augment_dataset
from miditok.utils import split_files_for_training

seed(256)

"""
@Author: Jonas
Here we specify the folder structure.

1) data_path is the path to the datasets folder located in CUPY/models/
2) name_of_midi_data is the acutal name of the folder which contains the raw midi data. 
3) tokenizer_path corresponds to the pre trained tokenizer
4) :TODO how should we set seq_length?

---------------------------------------------------------------

In our current implementation we use the REMI tokenizer.
"""

current_dir = os.getcwd()
data_path = os.path.join(os.path.dirname(current_dir) , "datasets")
name_of_midi_data = "transposed_midi"

midi_paths = list(Path(data_path, name_of_midi_data).glob("*.mid")) 
tokenizer_path = os.path.join(os.path.dirname(current_dir), "tokenizers/")

seq_length = 128

# Load the pre-trained tokenizer
tokenizer = REMI(params = Path(tokenizer_path, "tokenizer.json"))
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
    
    # move the midi files into the corresponding subset folder
     for midi_file in files_paths:
        midi_file = Path(midi_file)
        dst = subset_chunks_dir / midi_file.name  
        try:
            #print(f"Moving {midi_file} to {dst}")
            shutil.copy(midi_file, dst)
        except Exception as e:
            print(f"Error moving {midi_file} to {dst}: {e}")
    
    
    
        """
    velocity_offsets – list of velocity offsets for augmentation. If you plan to tokenize these files,
    the velocity offsets should be chosen accordingly to the number of velocities in your tokenizer’s
    vocabulary (num_velocities). (default: None)
         """
     do_augment = False
     if (do_augment):
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

collator: PAD (PAD_None): a padding token to use when training a model with batches of sequences of unequal lengths.
          The padding token id is often set to 0.
"""


def collator(input, seq_length, PAD=tokenizer.special_tokens_ids[0]):
    if (len(input) < seq_length):
        result =  input + [PAD] * (seq_length - len(input)) 
        return np.array(result, dtype=np.uint16)
    # if the input length is greatet than seq_len, truncate!
    return np.array(input[:seq_length], dtype=np.uint16)
        


def process_midi_file(midi_file, seq_length, tokenizer, collator):
    midi_file_tokenized = tokenizer(Path(midi_file))[0].ids
    # print(f"processed {midi_file}")
    return collator(midi_file_tokenized, seq_length)
    
# Create train, val, test token datasets
size_dict = {}

for subset in train_val_test_path:
    # Get a list of all midi files (as paths)
    files_path = train_val_test_path[subset]
    
    tokenizer.tokenize_dataset(files_paths = files_path,
                               out_dir = Path(files_path, "json"),
                               overwrite_mode = True,
                               verbose = True)
    
    

    # Convert JSON to Binary
    save_dir = os.path.join(output_path, f"{subset}.bin")
    
    json_dir = Path(files_path, "json")
    Path(json_dir).mkdir(parents = True, exist_ok = True)

    json_to_bin(json_dir=json_dir,
                bin_dir=save_dir,
                seq_length=seq_length,
                tokenizer=tokenizer
                )
    

ic(tokenizer.vocab_size)

