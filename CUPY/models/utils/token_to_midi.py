from miditok import REMI
from pathlib import Path
import os
import argparse
import numpy as np

tokenizer_name = REMI
tokenizer_name_str = tokenizer_name.__name__
vo_size = 1024
vocab_file = f"/csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/tokenizer_{tokenizer_name_str}_{vo_size}.json"  # Path to the tokenizer vocabulary file


def main():
    parser = argparse.ArgumentParser(description='Midi GPT inference')
    parser.add_argument('--token', type = int,
                        default = '')
    

    
    args = parser.parse_args()
    
    tokenizer = REMI(params=vocab_file)
    
    to_tokenize = np.array(args.token)
    to_tokenize.shape = (1,1)
    
    to_midi = tokenizer(to_tokenize)
    print(f"Generated midi: {to_midi}")
    
    
    to_midi.dump_midi(path = Path("/csghome/hpdc04/Transformer_Code/CUPY/models/utils/token_to_midi/", f"{tokenizer_name_str}_{vo_size}_{args.token}"))


if __name__ == "__main__":
    main()
