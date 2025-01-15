import os
import numpy as np
from pathlib import Path
import json

def json_to_bin(json_dir, output_dir, seq_length, tokenizer, subsets):
    """
    Convert JSON files containing token IDs to a single binary file per subset.

    Parameters:
    - json_dir: Directory containing the JSON files.
    - output_dir: Directory to save the combined binary files.
    - seq_length: Sequence length for padding/truncation.
    - tokenizer: Tokenizer object for padding IDs.
    - subsets: Dictionary with subset names and their corresponding file lists (e.g., train/val/test).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for subset, file_list in subsets.items():
        all_sequences = []

        for json_file in file_list:
            with open(json_file, 'r') as f:
                data = json.load(f)
                token_ids = data['ids'][0]

                # Pad or truncate the sequence
                if len(token_ids) < seq_length:
                    token_ids.extend([tokenizer.pad_token_id] * (seq_length - len(token_ids)))
                else:
                    token_ids = token_ids[:seq_length]  # Truncation

                all_sequences.append(token_ids)

        # Combine all sequences into a single numpy array
        combined_data = np.array(all_sequences, dtype=np.uint16)

        # Save combined data to binary file
        bin_file = Path(output_dir) / f"{subset}.bin"
        combined_data.tofile(bin_file)

        print(f"Saved {subset} data to {bin_file} with shape {combined_data.shape}")
