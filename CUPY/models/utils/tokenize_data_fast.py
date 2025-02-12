from miditok import Structured, REMI
from pathlib import Path
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections import Counter
import numpy as np
from tqdm import tqdm
from symusic import Score
from icecream import ic
from tabulate import tabulate
import matplotlib.pyplot as plt
import os

import sys
sys.path.insert(0, '/csghome/hpdc04/Transformer_Code/CUPY/models/GoePT/')
import config




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
                            manually_add_sos_eos = False,
                            subset = None):
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

    
    total_number_of_tokens = 0
    for ids in all_ids:
        total_number_of_tokens += len(ids)
    print(f"Total Number of Tokens: {total_number_of_tokens}")
    """
    Create a flat array
    """
    if (manually_add_sos_eos):
        token_sequence = []
        
        for ids in all_ids:
            # Start with a start token
            token_sequence.extend([sos_token] + ids + [eos_token])
    # Convert to numpy 
    print(len(token_sequence))
    token_array = np.array(token_sequence, dtype = np.int16)
    
    
    
    
    
    if(verbose):
        ic(self.special_tokens)
        analysis_results = analyze_tokenized_data(token_array, pad_token, sos_token, eos_token, trunc_token)
        
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

        if not (subset == None):
            visualize_tokenized_data_combined(token_array, pad_token, sos_token, eos_token, trunc_token, subset = subset) 

    self._verbose = False
    return token_array



def analyze_tokenized_data(token_array, pad_token_id, sos_token_id, eos_token_id, trunc_token_id):
    # Analyze sequence lengths before padding
    seq_lengths = [np.count_nonzero(row != pad_token_id) for row in token_array]
    max_token_id = np.max(token_array)
    number_of_min_len = np.sum([s == 2 for s in seq_lengths])
    
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
        token_stats["trunc_token_count"] += np.sum(row == trunc_token_id)
        

    # Compute statistics on sequence lengths
    length_stats = {
        "max_length": max(seq_lengths),
        "min_length": min(seq_lengths),
        "average_length": sum(seq_lengths) / len(seq_lengths),
        "median_length": np.median(seq_lengths),
        "max_token_id" : max_token_id,
        "number_of_min_length_rows": number_of_min_len
    }

    return {
        "length_stats": length_stats,
        "token_stats": token_stats
    }


def visualize_tokenized_data(token_array, pad_token_id, sos_token_id, eos_token_id, trunc_token_id,
                             output_path="/csghome/hpdc04/Transformer_Code/tokenization_summary_plots/",
                             subset = None):
    os.makedirs(output_path, exist_ok=True)
    
    # Analyze sequence lengths
    seq_lengths = [np.count_nonzero(row != pad_token_id) for row in token_array]
    
    # Token frequencies
    token_flat = token_array.flatten()
    token_counts = Counter(token_flat)
    
    # Special token counts
    special_token_counts = {
        "PAD": token_counts[pad_token_id],
        "SOS": token_counts[sos_token_id],
        "EOS": token_counts[eos_token_id],
        "TRUNC": token_counts.get(trunc_token_id, 0)  # Custom truncate token
    }
    
    # Plot sequence length distribution
    plt.figure(figsize=(12, 6))
    plt.hist(seq_lengths, bins=50, color="skyblue", edgecolor="black")
    plt.title("Sequence Length Distribution")
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")
    
    # Plot token frequency distribution
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(token_counts)), token_counts.values(), color="lightcoral")
    plt.title("Token Frequency Distribution")
    plt.xlabel("Token ID")
    plt.ylabel("Count")
    plt.xticks(range(0, max(token_counts.keys()), max(1, len(token_counts) // 20)), rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")
    
    # Plot special token statistics
    plt.figure(figsize=(8, 6))
    plt.bar(special_token_counts.keys(), special_token_counts.values(), color="lightgreen")
    plt.title("Special Token Statistics")
    plt.xlabel("Special Token")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")
    
    print("Token Frequency Sample (Top 10):")
    print(dict(Counter(token_flat).most_common(10)))









def visualize_tokenized_data_combined(token_array, pad_token_id, sos_token_id, eos_token_id, trunc_token_id,
                                      output_path="/csghome/hpdc04/Transformer_Code/tokenization_summary_plots/",
                                      subset = None):

    os.makedirs(output_path, exist_ok=True)
    output_path = Path(output_path, f"combined_visualization_{config.context_length}_{config.vo_size}_{config.tokenizer_name_str}_{subset}_manual_tokens_{config.manually_set_sos_eos_trunc}.png")
    
    # Analyze sequence lengths
    seq_lengths = [np.count_nonzero(row != pad_token_id) for row in token_array]
    
    # Token frequencies
    token_flat = token_array.flatten()
    token_counts = Counter(token_flat)
    
    # Special token counts
    special_token_counts = {
        "PAD": token_counts.get(pad_token_id, 0),
        "SOS": token_counts.get(sos_token_id, 0),
        "EOS": token_counts.get(eos_token_id, 0),
        "TRUNC": token_counts.get(trunc_token_id, 0)  # Custom truncate token
    }

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Sequence Length Distribution
    axs[0, 0].hist(seq_lengths, bins=50, color="skyblue", edgecolor="black")
    axs[0, 0].set_title("Sequence Length Distribution")
    axs[0, 0].set_xlabel("Sequence Length")
    axs[0, 0].set_ylabel("Frequency")
    axs[0, 0].grid(axis="y", linestyle="--", alpha=0.7)

    # Plot 2: Token Frequency Distribution
    axs[0, 1].bar(range(len(token_counts)), token_counts.values(), color="lightcoral")
    axs[0, 1].set_title("Token Frequency Distribution")
    axs[0, 1].set_xlabel("Token ID")
    axs[0, 1].set_ylabel("Count")
    axs[0, 1].set_xticks(range(0, max(token_counts.keys()) + 1, max(1, len(token_counts) // 20)))
    axs[0, 1].tick_params(axis="x", rotation=45)
    axs[0, 1].grid(axis="y", linestyle="--", alpha=0.7)

    # Plot 3: Special Token Statistics
    axs[1, 0].bar(special_token_counts.keys(), special_token_counts.values(), color="lightgreen")
    axs[1, 0].set_title("Special Token Statistics")
    axs[1, 0].set_xlabel("Special Token")
    axs[1, 0].set_ylabel("Count")
    axs[1, 0].grid(axis="y", linestyle="--", alpha=0.7)

    # Plot 4: Sequence Length Boxplot
    axs[1, 1].boxplot(seq_lengths, vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    axs[1, 1].set_title("Sequence Length Boxplot")
    axs[1, 1].set_xlabel("Sequence Length")
    axs[1, 1].grid(axis="x", linestyle="--", alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Combined visualization saved at: {output_path}")
    
    print("Token Frequency Sample (Top 10):")
    print(dict(Counter(token_flat).most_common(10)))



"""
@Author: Jonas
Monkey Patch tokenizer.tokenize_data_fast
"""
REMI.tokenize_dataset_to_bin = tokenize_dataset_to_bin
Structured.tokenize_dataset_to_bin = tokenize_dataset_to_bin
config.tokenizer_name.tokenize_dataset_to_bin = tokenize_dataset_to_bin