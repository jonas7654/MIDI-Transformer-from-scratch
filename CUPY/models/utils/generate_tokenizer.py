from random import shuffle
import os
import numpy as np
from icecream import ic

from miditok import REMI, TokenizerConfig  # here we choose to use REMI
from miditok.data_augmentation import augment_dataset
from miditok.utils import split_files_for_training
from pathlib import Path

# Specify the directories and configs
current_dir = os.getcwd()

data_dir = os.path.join(os.path.dirname(current_dir) , "datasets")
tokenizer_dir = os.path.join(os.path.dirname(current_dir), "tokenizers/")
files_path = list(Path(data_dir, "transposed_midi/").glob("*.mid")) # NOTE: Specify the right directory for the raw data

# Specify the vocab length
"""
NOTE: Eventhough we specified vocab_size it does not mean that the final size is equal to our value.
      If the value is to low in order to create a "meaningful" vocabulary or when it is to large the finale choice will differ
      See tokenizer.vocab_size
"""
vocab_size = 4096


# Create the Tokenizer
# :TODO : This is just random right now

"""
pitch range: The General MIDI 2 (GM2) specifications indicate the recommended ranges of pitches per MIDI program (instrument)
beat_res: determines the level of timing detail

use_programs: will use Program tokens to specify the instrument/MIDI program of the notes

"""
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109), 
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_chords": False,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_programs": False,
    "num_tempos": 32,  # number of tempo bins
    "tempo_range": (40, 250),  # (min, max)
}
config = TokenizerConfig(**TOKENIZER_PARAMS)

# Creates the tokenizer
tokenizer = REMI(config)

# Train the tokenizer
tokenizer.train(vocab_size = vocab_size, files_paths = files_path)
ic(tokenizer.vocab_size)

# Save the tokenizer
print(tokenizer_dir)
tokenizer.save(Path(tokenizer_dir,"tokenizer.json"))

