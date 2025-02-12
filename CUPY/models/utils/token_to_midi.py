from miditok import REMI
from pathlib import Path
import os
import argparse
import numpy as np

tokenizer_name = REMI
tokenizer_name_str = tokenizer_name.__name__
vo_size = 2048
vocab_file = f"/csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/tokenizer_{tokenizer_name_str}_{vo_size}.json"  # Path to the tokenizer vocabulary file
output_file = "/csghome/hpdc04/Transformer_Code/tokens_txt/vocab_midi_output.txt"  # Output text file


def main():
    # Initialize tokenizer
    tokenizer = REMI(params=vocab_file)
    
    # Open output file for writing
    with open(output_file, "w") as f:
        # Iterate through all tokens in the vocabulary
        for token_id in range(vo_size):
            # Convert token to numpy array with shape (1, 1)
            token_array = np.array([[token_id]], dtype=np.int32)
            
            # Decode token to MIDI
            try:
                midi_output = tokenizer.decode(token_array)
                f.write(f"Token {token_id}: {midi_output}\n")  # Write to file
            except Exception as e:
                f.write(f"Token {token_id}: Error - {str(e)}\n")  # Handle errors
            
            # Print progress
            if token_id % 100 == 0:
                print(f"Processed {token_id}/{vo_size} tokens...")
    
    print(f"MIDI outputs written to {output_file}")

    


if __name__ == "__main__":
    main()
