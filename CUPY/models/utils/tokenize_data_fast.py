from miditok import REMI
from pathlib import Path
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections import Counter
import numpy as np
from tqdm import tqdm
from symusic import Score

SCORE_LOADING_EXCEPTION = (
    RuntimeError,
    ValueError,
    OSError,
    FileNotFoundError,
    IOError,
    EOFError,
)
"""
Reference: https://github.com/Natooz/MidiTok/blob/515ed5078740ca7dcd643501adc0755b244c97ad/src/miditok/midi_tokenizer.py

"""


def tokenize_dataset_to_bin(self, files_paths: str | Path | Sequence[str | Path],
                            validation_fn=None,
                            save_programs=None,
                            verbose=True,
                            seq_length = None):
    """
    @Jonas
    Custom method to tokenize files and return a NumPy array.
    This only works with a tokenizer object
    """
    self._verbose = verbose

    # Resolve file paths
    if not isinstance(files_paths, Sequence):
        if isinstance(files_paths, str):
            files_paths = Path(files_paths)
        files_paths = [
            path
            for path in files_paths.glob("**/*")
            if path.suffix in ['.mid', '.midi']
        ]

    if save_programs is None:
        save_programs = not self.config.use_programs

    all_ids = []  # To store token IDs
    max_length = 0  # Track the longest sequence for padding

    desc = "Tokenizing music files (in-memory)"
    for file_path in tqdm(files_paths, desc=desc):
        file_path = Path(file_path)
        try:
            score = Score(file_path)
        except FileNotFoundError:
            if self._verbose:
                warnings.warn(f"File not found: {file_path}", stacklevel=2)
            continue
        except SCORE_LOADING_EXCEPTION:
            continue

        # Validate the score
        if validation_fn is not None and not validation_fn(score):
            continue

        # Tokenize the Score
        tokens = self.encode(score)
        
        # Collect token IDs
        token_ids = tokens[0].ids
        all_ids.append(token_ids)
        max_length = max(max_length, len(token_ids))
        
        if seq_length is None:
            seq_length = max_length

    # Convert collected token IDs to a padded NumPy array
    token_array = np.array(
        [
            ids[:seq_length] + [self.pad_token_id] * (seq_length - len(ids[:seq_length]))
            for ids in all_ids
        ],
        dtype=np.int32
    )
    
    if(verbose):
        analysis_results = analyze_tokenized_data(token_array, self.pad_token_id, self.sos_token_id, self.eos_token_id)
        
        print("\nSequence Length Stats:")
        for k, v in analysis_results["length_stats"].items():
            print(f"{k}: {v}")

        print("\nSpecial Token Stats:")
        for k, v in analysis_results["token_stats"].items():
            print(f"{k}: {v}")
        

    self._verbose = False
    return token_array



def analyze_tokenized_data(token_array, pad_token_id, sos_token_id, eos_token_id):
    # Analyze sequence lengths before padding
    seq_lengths = [np.count_nonzero(row != pad_token_id) for row in token_array]
    
    # Statistics dict
    token_stats = {
        "pad_token_count": 0,
        "sos_token_count": 0,
        "eos_token_count": 0,
        "pad_positions": [],
        "sos_positions": [],
        "eos_positions": []
    }

    for row in token_array:
        token_stats["pad_token_count"] += np.sum(row == pad_token_id)
        token_stats["sos_token_count"] += np.sum(row == sos_token_id)
        token_stats["eos_token_count"] += np.sum(row == eos_token_id)
        
        token_stats["pad_positions"].extend(np.where(row == pad_token_id)[0])
        token_stats["sos_positions"].extend(np.where(row == sos_token_id)[0])
        token_stats["eos_positions"].extend(np.where(row == eos_token_id)[0])

    # Compute statistics on sequence lengths
    length_stats = {
        "max_length": max(seq_lengths),
        "min_length": min(seq_lengths),
        "average_length": sum(seq_lengths) / len(seq_lengths),
        "median_length": np.median(seq_lengths)
    }

    return {
        "length_stats": length_stats,
        "token_stats": token_stats
    }


"""
@Author: Jonas
Monkey Patch tokenizer.tokenize_data_fast

"""
REMI.tokenize_dataset_to_bin = tokenize_dataset_to_bin