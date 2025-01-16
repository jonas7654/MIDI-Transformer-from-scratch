import os
import numpy as np
from pathlib import Path
import json
from icecream import ic

def json_to_bin(json_dir, output_dir, seq_length, tokenizer, subset, verbose = False):

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    json_dir = list(json_dir.glob("*.json"))
    
    all_sequences = []

    for i , json_file in enumerate(json_dir):
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
        
        if (verbose):
            if (i % 1000 == 0):
                assert np.all(combined_data < tokenizer.vocab_size), "Found out-of-vocabulary tokens in dataset"
                ic(combined_data[:100])
                ic(combined_data.shape)
                print(f"Saved {subset} data to {bin_file} with shape {combined_data.shape}")
 
        
                       # Sanity check
        
    
