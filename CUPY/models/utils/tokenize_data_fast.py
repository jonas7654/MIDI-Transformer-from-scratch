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
    if (manually_add_sos_eos and not config.use_random_padding_token):                    
        sequences = [np.concatenate(([sos_token], ids, [eos_token])).astype(np.int16) for ids in all_ids]
    elif (manually_add_sos_eos and config.use_random_padding_token):
        sequences = []
        max_pad = config.max_pad
        for ids in all_ids:
            # Randomly select number of padding tokens (0 to max_pad)
            num_pads = np.random.randint(0, max_pad + 1)

            # Create sequence with padding + SOS + tokens + EOS
            padded_sequence = [pad_token] * num_pads + [sos_token] + ids + [eos_token]
            sequences.append(np.array(padded_sequence, dtype=np.int16))  
            
            total_number_of_tokens = sum(len(seq) for seq in sequences)
            individual_sequence_lengths = [len(seq) for seq in sequences]  
               
    token_array = np.concatenate(sequences)
 
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
            
            visualize_tokenized_data_pgf(token_array,
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


def visualize_tokenized_data_pgf(token_array, pad_token_id, sos_token_id, eos_token_id, trunc_token=None,
                                 output_path="tokenization_summary_plots/",
                                 subset=None,
                                 lengths=None,
                                 sequences=None):
    output_path = Path(output_path)
    os.makedirs(output_path, exist_ok=True)

    base_filename = f"visualization_{config.vo_size}_{config.tokenizer_name_str}_{str(subset)}_manual_tokens_{config.manually_set_sos_eos_trunc}_random_padding_{config.use_random_padding_token}"


    token_flat = token_array.flatten()
    token_counts = Counter(token_flat)

    pgf_filenames = []

    # Function to save plots as PGF
    def save_pgf(fig, name):
        filename = f"{base_filename}_{name}.pgf"
        filepath = output_path / filename
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        pgf_filenames.append(filename)

    # 1. Sequence Length Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lengths, bins=np.arange(0, max(lengths) + 50, 50), color="skyblue")
    ax.set_title("Sequence Length Distribution")
    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    save_pgf(fig, "seq_len_dist_pgf")

    # 2. Token Frequency Distribution (Log-Log Scale)
    fig, ax = plt.subplots(figsize=(6, 4))
    token_frequencies = [count for token, count in token_counts.items() if token not in {pad_token_id, sos_token_id, eos_token_id}]
    bins = np.logspace(np.log10(min(token_frequencies)), np.log10(max(token_frequencies)), 50)
    ax.hist(token_frequencies, bins=bins, color="orange", edgecolor="black", alpha=0.7, log=True)
    ax.set_xscale("log")
    ax.set_title("Token Frequency Distribution (Log-Log)")
    ax.set_xlabel("Token Frequency (log)")
    ax.set_ylabel("Log Count")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    save_pgf(fig, "freq_dist_log_log_pgf")

    # 3. Top 20 Most Frequent Tokens
    fig, ax = plt.subplots(figsize=(6, 4))
    top_n = 20
    top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    ax.barh([str(t[0]) for t in top_tokens], [t[1] for t in top_tokens], color="green")
    ax.set_title(f"Top {top_n} Most Frequent Tokens")
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    save_pgf(fig, "top_frequent_tokens_pgf")









def visualize_tokenized_data_combined(token_array, pad_token_id, sos_token_id, eos_token_id, trunc_token=None,
                                      output_path="/csghome/hpdc04/Transformer_Code/tokenization_summary_plots/",
                                      subset=None,
                                      lengths=None,
                                      sequences=None):
    output_path = Path(output_path)
    os.makedirs(str(output_path), exist_ok=True)

    base_filename = f"visualization_{config.vo_size}_{config.tokenizer_name_str}_{str(subset)}_manual_tokens_{config.manually_set_sos_eos_trunc}_random_padding_{config.use_random_padding_token}"

    token_flat = token_array.flatten()
    token_counts = Counter(token_flat)

    # 1. Sequence Length Distribution
    plt.figure(figsize=(8, 6))
    plt.hist(lengths, bins=np.arange(0, max(lengths)+50, 50), color="skyblue", log=False)
    plt.title("Sequence Length Distribution")
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Log Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_path / f"{base_filename}_seq_length_distribution.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 2. Full Token Distribution (excluding tokens 0 and 1)
    plt.figure(figsize=(8, 6))
    all_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    all_tokens_filtered = [t for t in all_tokens if t[0] not in {0, 1, 2}]
    plt.barh([str(t[0]) for t in all_tokens_filtered], [t[1] for t in all_tokens_filtered], color="salmon")
    plt.title("Full Token Distribution (Excluding Tokens 0, 1, 2)")
    plt.xlabel("Count")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.savefig(output_path / f"{base_filename}_full_token_distribution.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 3. Cumulative Distribution of Sequence Lengths
    plt.figure(figsize=(8, 6))
    sorted_lengths = np.sort(lengths)
    yvals = np.arange(len(sorted_lengths)) / float(len(sorted_lengths) - 1)
    plt.plot(sorted_lengths, yvals, color="purple", linewidth=2)
    plt.title("Cumulative Distribution of Sequence Lengths")
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Cumulative Proportion")
    plt.grid(linestyle="--", alpha=0.7)
    plt.axvline(np.median(lengths), color="red", linestyle="--", label=f"Median = {np.median(lengths)}")
    plt.legend()
    plt.savefig(output_path / f"{base_filename}_cumulative_length_distribution.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 4. Boxplot of Sequence Lengths
    plt.figure(figsize=(8, 6))
    plt.boxplot(lengths, vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"), showfliers=False)
    plt.title("Sequence Length Distribution")
    plt.ylabel("Tokens")
    plt.xticks([1], ["All Sequences"])
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_path / f"{base_filename}_boxplot_sequence_lengths.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 5. Token Frequency Distribution (Log-Log Scale)
    plt.figure(figsize=(8, 6))
    token_frequencies = [count for token, count in token_counts.items() if token not in {pad_token_id, sos_token_id, eos_token_id}]
    bins = np.logspace(np.log10(min(token_frequencies)), np.log10(max(token_frequencies)), 50)
    plt.hist(token_frequencies, bins=bins, color="orange", edgecolor="black", alpha=0.7, log=True)
    plt.xscale("log")
    plt.title("Token Frequency Distribution (Log-Log)")
    plt.xlabel("Token Frequency (log)")
    plt.ylabel("Log Count")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_path / f"{base_filename}_token_frequency_loglog.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 6. Top N Most Frequent Tokens
    plt.figure(figsize=(8, 6))
    top_n = 20
    top_tokens = [t for t in all_tokens if t[0] not in {pad_token_id, sos_token_id, eos_token_id}][:top_n]
    plt.barh([str(t[0]) for t in top_tokens], [t[1] for t in top_tokens], color="green")
    plt.title(f"Top {top_n} Most Frequent Tokens")
    plt.xlabel("Count")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.savefig(output_path / f"{base_filename}_top_frequent_tokens.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 7. Token Length vs. Frequency Line Plot
    plt.figure(figsize=(8, 6))
    df = pd.DataFrame({'token_id': [t[0] for t in all_tokens if t[0] not in {pad_token_id, sos_token_id, eos_token_id}], 
                       'frequency': [t[1] for t in all_tokens if t[0] not in {pad_token_id, sos_token_id, eos_token_id}]}).sort_values('token_id')
    plt.plot(df['token_id'], df['frequency'], color='royalblue', linewidth=1.5, alpha=0.8)
    plt.title("Token Frequency by ID")
    plt.xlabel("Token ID")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_path / f"{base_filename}_token_frequency_by_id.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 8. Token Frequency Bar Plot
    plt.figure(figsize=(8, 6))
    plt.bar(df['token_id'], df['frequency'], width=1, color='royalblue', alpha=0.5, edgecolor='none')
    plt.xlim(0, 4095)
    plt.title("Token Frequency by ID")
    plt.xlabel("Token ID")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.savefig(output_path / f"{base_filename}_token_frequency_bar.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 9. Token Diversity Analysis
    plt.figure(figsize=(8, 6))
    unique_tokens_per_seq = [len(set(seq)) for seq in sequences]
    plt.hist(unique_tokens_per_seq, bins=50, color="teal", edgecolor="black")
    plt.title("Unique Tokens per Sequence")
    plt.xlabel("Number of Unique Tokens")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_path / f"{base_filename}_unique_tokens_per_sequence.png", bbox_inches="tight", dpi=300)
    plt.close()





"""
@Author: Jonas  
Monkey Patch tokenizer.tokenize_data_fast
"""
REMI.tokenize_dataset_to_bin = tokenize_dataset_to_bin
Structured.tokenize_dataset_to_bin = tokenize_dataset_to_bin
config.tokenizer_name.tokenize_dataset_to_bin = tokenize_dataset_to_bin