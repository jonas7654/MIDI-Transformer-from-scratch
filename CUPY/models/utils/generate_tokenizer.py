from random import shuffle
import os
import numpy as np
from icecream import ic
import argparse

from miditok import Structured, REMI, TokenizerConfig  # here we choose to use REMI
from miditok.data_augmentation import augment_dataset
from miditok.utils import split_files_for_training
from pathlib import Path

import sys
sys.path.insert(0, '/csghome/hpdc04/Transformer_Code/CUPY/models/GoePT/')
import config


def train_tokenizer(vocab_size):
    # Specify the directories and configs
    current_dir = os.getcwd()
    data_dir = os.path.join(os.path.dirname(current_dir), "datasets")
    tokenizer_dir = os.path.join(os.path.dirname(current_dir), "tokenizers/")
    files_path = list(Path(data_dir, "transposed_midi/").glob("*.mid"))  # Raw MIDI files
    
    # Create the Tokenizer
    TOKENIZER_PARAMS = {
        "pitch_range": (40, 109),
        "beat_res": {(4, 4): 8},
        "num_velocities": 1,
        "special_tokens": ["PAD", "BOS", "EOS"],
        "use_note_duration_programs": [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127],
        "use_chords": False,
        "use_rests": True,
        "use_tempos": False,
        "use_time_signatures": False,
        "use_programs": False,
        "use_pitchdrum_tokens": False,
        "default_note_duration" : 0.5, 
        "num_tempos": 32,  # number of tempo bins
        "tempo_range": (40, 250),  # (min, max)
        #"max_bar_embedding" : 8
    }
    
    tok_config = TokenizerConfig(**TOKENIZER_PARAMS)
    
    # Creates the tokenizer
    tokenizer = config.tokenizer_name(tok_config)

    # Train the tokenizer
    tokenizer.train(vocab_size=vocab_size, files_paths=files_path)
    ic(tokenizer.vocab_size)
    ic(tokenizer.special_tokens)
    ic(tokenizer.special_tokens_ids)
    
    # Save the tokenizer
    print(f"saved to: {tokenizer_dir}")
    tokenizer.save(Path(tokenizer_dir, f"tokenizer_{config.tokenizer_name_str}_{tokenizer.vocab_size}.json"))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tokenizer with different vocab sizes")
    # nargs = '+' allows to pass multiple arguments at once
    parser.add_argument('--vocab_sizes', type=int, nargs='+', required=True, help="Vocabulary size for training")
    
    args = parser.parse_args()

    for vocab_size in args.vocab_sizes:
        print(f"Training tokenizer ({config.tokenizer_name_str}) with vocab size: {vocab_size}")
        train_tokenizer(vocab_size=vocab_size)
