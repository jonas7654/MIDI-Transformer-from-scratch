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
import pandas as pd
import matplotlib.ticker as ticker  # Import ticker for AutoMinorLocator
plt.style.use('fivethirtyeight')

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

    
    total_number_of_tokens = sum(len(ids) + 2 for ids in all_ids)
    
    print(f"Total Number of Tokens: {total_number_of_tokens}")
    individual_sequence_lengths = [len(ids) + 2 for ids in all_ids] # +2 since we add sos and eos
    
    """
    Create a flat array
    """
    if (manually_add_sos_eos):                    
        sequences = [np.concatenate(([sos_token], ids, [eos_token])).astype(np.int16) for ids in all_ids]
        token_array = np.concatenate(sequences)    
        
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
            visualize_tokenized_data_combined(token_array,
                                              pad_token,
                                              sos_token,
                                              eos_token,
                                              trunc_token,
                                              subset = subset,
                                              lengths = individual_sequence_lengths,
                                              sequences = all_ids) 

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









def visualize_tokenized_data_combined(token_array, pad_token_id, sos_token_id, eos_token_id, trunc_token=None,
                                      output_path="/csghome/hpdc04/Transformer_Code/tokenization_summary_plots/",
                                      subset=None,
                                      lengths=None,
                                      sequences=None):
    output_path = Path(output_path)
    os.makedirs(str(output_path), exist_ok=True)
    output_path = output_path / f"combined_visualization_{config.vo_size}_{config.tokenizer_name_str}_{str(subset)}_manual_tokens_{config.manually_set_sos_eos_trunc}.png"
        
    # Token frequencies
    token_flat = token_array.flatten()
    token_counts = Counter(token_flat)
    
    # Create figure with subplots
    plt.figure(figsize=(20, 20))
    plt.suptitle(f"Tokenization Analysis - {subset.capitalize()} Set", y=1.02, fontsize=16)

    # Plot 1: Sequence Length Distribution (Log Scale)
    plt.subplot(3, 3, 1)
    plt.hist(lengths, bins=np.arange(0, max(lengths)+50, 50), color="skyblue", log=False)
    plt.title("Sequence Length Distribution")
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Log Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Plot 2: Full Token Distribution (excluding tokens 0 and 1)
    plt.subplot(3, 3, 2)
    all_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    # Filter out tokens 0 and 1
    all_tokens_filtered = [t for t in all_tokens if t[0] not in {0, 1, 2}]
    plt.barh([str(t[0]) for t in all_tokens_filtered], [t[1] for t in all_tokens_filtered], color="salmon")
    plt.title("Full Token Distribution (Excluding Tokens 0 and 1)")
    plt.xlabel("Count")
    plt.gca().invert_yaxis()  # Highest count at top
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Plot 3: Cumulative Distribution of Sequence Lengths
    plt.subplot(3, 3, 3)
    sorted_lengths = np.sort(lengths)
    yvals = np.arange(len(sorted_lengths)) / float(len(sorted_lengths) - 1)
    plt.plot(sorted_lengths, yvals, color="purple", linewidth=2)
    plt.title("Cumulative Distribution of Sequence Lengths")
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Cumulative Proportion")
    plt.grid(linestyle="--", alpha=0.7)
    # Add a vertical line at the median
    median_length = np.median(lengths)
    plt.axvline(median_length, color="red", linestyle="--", label=f"Median = {median_length}")
    plt.legend()

    # Plot 4: Boxplot of Sequence Lengths
    plt.subplot(3, 3, 4)
    plt.boxplot(lengths, vert=True, patch_artist=True, 
               boxprops=dict(facecolor="lightblue"), showfliers=False)
    plt.title("Sequence Length Distribution")
    plt.ylabel("Tokens")
    plt.xticks([1], ["All Sequences"])
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Plot 5: Token Frequency Distribution (Log-Log Scale)
    plt.subplot(3, 3, 5)
    token_frequencies = [count for token, count in token_counts.items() 
                         if token not in {pad_token_id, sos_token_id, eos_token_id}]

    # Generate log-spaced bins (adjust min/max as needed)
    bins = np.logspace(np.log10(min(token_frequencies)), 
                   np.log10(max(token_frequencies)), 
                   50)

    plt.hist(token_frequencies, bins=bins, color="orange", edgecolor="black", 
             alpha=0.7, log=True)
    plt.xscale("log")  # Log-scale x-axis
    plt.title("Token Frequency Distribution (Log-Log)")
    plt.xlabel("Token Frequency (log)")
    plt.ylabel("Log Count")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Plot 6: Top N Most Frequent Tokens (Excluding Special Tokens)
    plt.subplot(3, 3, 6)
    top_n = 20
    top_tokens = [t for t in all_tokens if t[0] not in {pad_token_id, sos_token_id, eos_token_id}][:top_n]
    plt.barh([str(t[0]) for t in top_tokens], [t[1] for t in top_tokens], color="green")
    plt.title(f"Top {top_n} Most Frequent Tokens")
    plt.xlabel("Count")
    plt.gca().invert_yaxis()  # Highest count at top
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Plot 7: Token Length vs. Frequency Scatter Plot
    #plt.subplot(3, 3, 7)
    #token_ids = [t[0] for t in all_tokens if t[0] not in {pad_token_id, sos_token_id, eos_token_id}]
    #frequencies = [t[1] for t in all_tokens if t[0] not in {pad_token_id, sos_token_id, eos_token_id}]
    #plt.scatter(token_ids, frequencies, color="blue", alpha=0.5)
    #plt.title("Token ID vs. Frequency")
    #plt.xlabel("Token ID")
    #plt.ylabel("Frequency")
    #plt.grid(linestyle="--", alpha=0.7)

    # Plot 7: Token Length vs. Frequency (Elegant Version)
    plt.subplot(3, 3, 7)

    # Filter data (as before)
    token_ids = [t[0] for t in all_tokens if t[0] not in {pad_token_id, sos_token_id, eos_token_id}]
    frequencies = [t[1] for t in all_tokens if t[0] not in {pad_token_id, sos_token_id, eos_token_id}]

    # Create dataframe for easy handling
    df = pd.DataFrame({'token_id': token_ids, 'frequency': frequencies}).sort_values('token_id')

    # Plot with a line (no markers)
    plt.plot(df['token_id'], df['frequency'], 
         color='royalblue', 
         linewidth=1.5,
         alpha=0.8)

    # Set axis limits to match vocabulary size
    #plt.xlim(0, 4095)  # Explicitly constrain to your vocab size
    #plt.ylim(0, max(df['frequency']) * 1.1)  # Add 10% padding to y-axis

    # Formatting
    plt.title("Token Frequency by ID", fontsize=14, pad=15)
    plt.xlabel("Token ID", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Optional: Add minor gridlines for better readability
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(500))  # Minor ticks every 500 tokens
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Improve clarity
    plt.gca().set_facecolor('#f7f7f7')  # Light background
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()


    

    # Plot 8: bar plot 
    
    # Use tiny bars (width=1 to prevent gaps)
    plt.bar(df['token_id'], df['frequency'], 
        width=1,  # No gaps between bars
        color='royalblue', 
        alpha=0.5, 
        edgecolor='none')

    plt.xlim(0, 4095)
    plt.title("Token Frequency by ID", fontsize=14, pad=15)
    plt.xlabel("Token ID", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')  # Only horizontal grid
    plt.tight_layout()

    # Plot 9: Token Diversity Analysis
    plt.subplot(3, 3, 9)
    unique_tokens_per_seq = [len(set(seq)) for seq in sequences]  # Use 'sequences' instead of 'token_array'
    plt.hist(unique_tokens_per_seq, bins=50, color="teal", edgecolor="black")
    plt.title("Unique Tokens per Sequence")
    plt.xlabel("Number of Unique Tokens")
    plt.ylabel("Frequency")
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