from random import shuffle, seed
import os, shutil
import numpy as np
from icecream import ic
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from miditok import REMI, TokenizerConfig  # here we choose to use REMI
from miditok.data_augmentation import augment_dataset
from miditok.utils import split_files_for_training

seed(256)


current_dir = os.getcwd()
data_path = os.path.join(os.path.dirname(current_dir) , "datasets")
name_of_midi_data = "transposed_midi"

midi_paths = list(Path(data_path, name_of_midi_data).glob("*.mid")) 
tokenizer_path = Path("/csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/tokenizer.json")

seq_length = 128

tokenizer = REMI(params = tokenizer_path)
ic(tokenizer.vocab_size)


test = tokenizer(Path("/csghome/hpdc04/Transformer_Code/CUPY/models/datasets/transposed_midi/... Baby One More Time_track1.mid"))
test2= tokenizer.encode_token_ids(test[0])
print(test[0].ids)
print(test2)