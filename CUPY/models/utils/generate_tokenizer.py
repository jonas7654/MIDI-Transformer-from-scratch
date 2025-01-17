from random import shuffle
import os
import numpy as np
from icecream import ic
import argparse

from miditok import REMI, TokenizerConfig  # here we choose to use REMI
from miditok.data_augmentation import augment_dataset
from miditok.utils import split_files_for_training
from pathlib import Path


def train_tokenizer(vocab_size):
    # Specify the directories and configs
    current_dir = os.getcwd()
    data_dir = os.path.join(os.path.dirname(current_dir), "datasets")
    tokenizer_dir = os.path.join(os.path.dirname(current_dir), "tokenizers/")
    files_path = list(Path(data_dir, "transposed_midi/").glob("*.mid"))  # Raw MIDI files
    
    # Create the Tokenizer
    TOKENIZER_PARAMS = {
        "pitch_range": (21, 109),
        "beat_res": {(0, 4): 8, (4, 12): 4},
        "num_velocities": 32,
        "special_tokens": ["PAD", "BOS", "EOS"],
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
    tokenizer.train(vocab_size=vocab_size, files_paths=files_path)
    ic(tokenizer.vocab_size)
    ic(tokenizer.special_tokens)
    ic(tokenizer.special_tokens_ids)
    
    # Save the tokenizer
    print(f"saved to: {tokenizer_dir}")
    tokenizer.save(Path(tokenizer_dir, f"tokenizer_{tokenizer.vocab_size}.json"))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tokenizer with different vocab sizes")
    # nargs = '+' allows to pass multiple arguments at once
    parser.add_argument('--vocab_sizes', type=int, nargs='+', required=True, help="Vocabulary size for training")
    
    args = parser.parse_args()

    for vocab_size in args.vocab_sizes:
        print(f"Training tokenizer with vocab size: {vocab_size}")
        train_tokenizer(vocab_size=vocab_size)
