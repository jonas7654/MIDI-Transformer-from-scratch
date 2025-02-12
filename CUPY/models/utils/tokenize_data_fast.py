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
        token_sequence = np.empty(total_number_of_tokens, dtype=np.int16)
        current_position = 0
        
        for ids in all_ids:
            seq_len = len(ids)
            
            token_sequence[current_position] = sos_token
            token_sequence[current_position + 1:current_position + 1 + seq_len] = ids 
            token_sequence[current_position + 1 + seq_len] = eos_token
            
            current_position += seq_len + 2
    # Convert to numpy 
    print(f"Final array length: {len(token_array)}")
    
    
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
                                      subset=None):
    
    os.makedirs(output_path, exist_ok=True)
    output_path = Path(output_path, f"combined_visualization_{config.context_length}_{config.vo_size}_{config.tokenizer_name_str}_{subset}_manual_tokens_{config.manually_set_sos_eos_trunc}.png")
    
    # Analyze sequence lengths
    seq_lengths = [len(token_array)] if token_array.ndim == 1 else [np.count_nonzero(row != pad_token_id) for row in token_array]
    
    # Token frequencies
    token_flat = token_array.flatten()
    token_counts = Counter(token_flat)
    
    # Special token counts with labels including IDs
    special_labels = {
        "PAD": pad_token_id,
        "SOS": sos_token_id,
        "EOS": eos_token_id,
        "TRUNC": trunc_token_id
    }
    special_token_counts = {f"{k} ({v})": token_counts.get(v, 0) for k, v in special_labels.items()}

    # Create figure with subplots
    plt.figure(figsize=(18, 12))
    plt.suptitle(f"Tokenization Analysis - {subset.capitalize()} Set", y=1.02, fontsize=14)

    # Plot 1: Sequence Length Distribution (Log Scale)
    plt.subplot(2, 2, 1)
    plt.hist(seq_lengths, bins=50, color="skyblue", edgecolor="black", log=True)
    plt.title("Sequence Length Distribution (Log Scale)")
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Log Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Plot 2: Top 20 Frequent Tokens
    plt.subplot(2, 2, 2)
    top_tokens = token_counts.most_common(20)
    plt.barh([str(t[0]) for t in top_tokens], [t[1] for t in top_tokens], color="salmon")
    plt.title("Top 20 Frequent Tokens")
    plt.xlabel("Count")
    plt.gca().invert_yaxis()  # Highest count at top
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Plot 3: Special Token Statistics
    plt.subplot(2, 2, 3)
    bars = plt.bar(special_token_counts.keys(), special_token_counts.values(), color="lightgreen")
    plt.title("Special Token Counts")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:,}',
                 ha='center', va='bottom')
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Plot 4: Boxplot of Sequence Lengths
    plt.subplot(2, 2, 4)
    plt.boxplot(seq_lengths, vert=True, patch_artist=True, 
               boxprops=dict(facecolor="lightblue"), showfliers=False)
    plt.title("Sequence Length Distribution")
    plt.ylabel("Tokens")
    plt.xticks([1], ["All Sequences"])
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
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