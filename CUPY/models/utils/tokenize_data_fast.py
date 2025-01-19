from miditok import REMI
from pathlib import Path
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections import Counter
import numpy as np
from tqdm import tqdm
from symusic import Score
from icecream import ic
from tabulate import tabulate

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
                            seq_length = None,
                            manually_add_sos_eos = False):
    """
    @Jonas
    Custom method to tokenize files and return a NumPy array.
    This only works with a tokenizer object

    PAD (PAD_None): a padding token to use when training a model with batches of sequences of unequal lengths. The padding token id is often set to 0. If you use Hugging Face models, be sure to pad inputs with this tokens, and pad labels with -100.

    BOS (SOS_None): “Start Of Sequence” token, indicating that a token sequence is beginning.

    EOS (EOS_None): “End Of Sequence” tokens, indicating that a token sequence is ending. For autoregressive generation, this token can be used to stop it.

    MASK (MASK_None): a masking token, to use when pre-training a (bidirectional) model with a self-supervised objective like BERT.

    Note: you can use the tokenizer.special_tokens property to get the list of the special tokens of a tokenizer, and tokenizer.special_tokens for their ids.
    
    
    
    NOTE: SPECIAL TRUNCATE TOKEN: -1 (self defined and not from the tokenizer)
    """
    self._verbose = verbose
    
    if (manually_add_sos_eos):
        print(f"Warning: manually_add_sos_eos is set to True")

    
    
    pad_token = self.pad_token_id
    sos_token = self.special_tokens_ids[1]
    eos_token = self.special_tokens_ids[2]
    trunc_token = -1

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



    """
    If the sequence length is longer than 'seq_len' we add the sos token to the beginning and add a custom truncate token to the end
    to specify that we have truncated the midi file
    
    """
    # Convert collected token IDs to a padded NumPy array
    if (manually_add_sos_eos):
        token_array = np.empty((len(all_ids), seq_length), dtype = np.int16)
        
        for idx, ids in enumerate(all_ids):
            # Start with a start token
            token_sequence = [sos_token]
            
            if (len(ids) > seq_length - 2): # Truncate if too long
                token_sequence += ids[:seq_length - 2] + [trunc_token]
            else:
                token_sequence += ids
                token_sequence += [eos_token]
            
            # Ensure the sequence is exactly seq_length
            token_sequence = token_sequence[:seq_length]
            token_sequence += [pad_token] * (seq_length - len(token_sequence))
        
            # Assign to token array
            token_array[idx, :] = token_sequence
    else:
        token_array = np.array(
        [
            ids[:seq_length] + [pad_token] * (seq_length - len(ids[:seq_length]))
            for ids in all_ids
        ],
        dtype=np.int16
                              )
    
    
    
    
    
    if(verbose):
        ic(self.special_tokens)
        analysis_results = analyze_tokenized_data(token_array, pad_token, sos_token, eos_token)
        
         # Format the results for better readability
        length_stats_table = [
            [key, value] for key, value in analysis_results["length_stats"].items()
        ]
        token_stats_table = [
            [key, value if not isinstance(value, list) else f"{len(value)} positions"]
            for key, value in analysis_results["token_stats"].items()
        ]

        print("\nSequence Length Stats:")
        print(tabulate(length_stats_table, headers=["Metric", "Value"], tablefmt="grid"))

        print("\nSpecial Token Stats:")
        print(tabulate(token_stats_table, headers=["Token Type", "Count/Details"], tablefmt="grid"))


    self._verbose = False
    return token_array



def analyze_tokenized_data(token_array, pad_token_id, sos_token_id, eos_token_id):
    # Analyze sequence lengths before padding
    seq_lengths = [np.count_nonzero(row != pad_token_id) for row in token_array]
    max_token_id = np.max(token_array)
    
    # Statistics dict
    token_stats = {
        "pad_token_count": 0,
        "sos_token_count": 0,
        "eos_token_count": 0,
        "trunc_token_count": 0
    }

    for row in token_array:
        token_stats["pad_token_count"] += np.sum(row == pad_token_id)
        token_stats["sos_token_count"] += np.sum(row == sos_token_id)
        token_stats["eos_token_count"] += np.sum(row == eos_token_id)
        token_stats["trunc_token_count"] += np.sum(row == -1)
        

    # Compute statistics on sequence lengths
    length_stats = {
        "max_length": max(seq_lengths),
        "min_length": min(seq_lengths),
        "average_length": sum(seq_lengths) / len(seq_lengths),
        "median_length": np.median(seq_lengths),
        "max_token_id" : max_token_id
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