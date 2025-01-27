from model import GoePT, compute_gradient
from layers import Softmax
import json
from icecream import ic 
from miditok import REMI, Structured
from miditok.utils import merge_scores
import cupy as cp
import numpy as np
import argparse
from pathlib import Path
import sys


sys.path.append('/csghome/hpdc04/Transformer_Code/CUPY/models/utils')

from tokenize_data_fast import tokenize_dataset_to_bin
import config

def load_model(checkpoint_path, vocab_file, batch_size):
    with open(checkpoint_path, mode = 'r', encoding = 'utf-8') as weights:
        state_dict = json.load(weights)
    
    model = GoePT.from_state_dict(state_dict, batch_size=batch_size)
    
    tokenizer = config.tokenizer_name(params=vocab_file)
    ic(tokenizer)
    return model, tokenizer

def softmax_with_temperature(logits, temperature=1.0, axis = -1):
    exp_logits = np.exp(logits / temperature)
    return exp_logits / np.sum(exp_logits, axis = axis)

def tokenize_input(midi_input, tokenizer):
    # Tokenize and return integer representation
    tokenized_input = tokenizer(midi_input)[0].ids
    return tokenized_input
    

def decode_tokens(midi_input, tokenizer):
    decoded_midi = tokenizer(midi_input)
    return decoded_midi

def main():
    parser = argparse.ArgumentParser(description='Midi GPT inference')
    parser.add_argument('--weights', type = str,
                        default = '')
    parser.add_argument('--vocab-file', type = str)   
    parser.add_argument('--input', type = str,
                        help = "Path to the Input midi file") 
    parser.add_argument('--save-dir', type = str)
    parser.add_argument('--context-length', type = int)
    parser.add_argument('--b', type = int)
    parser.add_argument('--manually-set-sos-eos-trunc', type = bool)
    

    
    args = parser.parse_args()
    
    file_path = Path(args.input)
    number_of_files = len(list(file_path.glob("*.mid")))
    
    model, tokenizer = load_model(args.weights, args.vocab_file, number_of_files)
    seq_len = model.context_length

    
    # Tokenize the input
    tokenized_data = tokenizer.tokenize_dataset_to_bin(files_paths = file_path,
                                      verbose = True,
                                      seq_length = seq_len,
                                      manually_add_sos_eos = args.manually_set_sos_eos_trunc) # args.manually_set_sos_eos_trunc
    
    
    generated_sequence = cp.asanyarray(tokenized_data.copy())
    generated_sequence[:, seq_len-1] = 186
    
    # Remove the EOS token : TODO : dont duplicate tokens at the end
    print(f"context_size: {seq_len}")
    print(f"Input sequence: \n {generated_sequence}")
    
    
    
    for idx in range(args.b):
        input_context = generated_sequence[:, -seq_len:]
        logits, _ = model.forward(input_context, targets = None)
        predictions = softmax_with_temperature(logits, temperature = 1)
        next_tokens = cp.argmax(predictions, axis = -1)  # axis -1 uses the last axis which is the vocabulary
        # Append the predicted token to the sequence
        generated_sequence = cp.concatenate([generated_sequence, next_tokens], axis=1) # add new column
        
    # convert back to numpy
    generated_sequence = generated_sequence.get()
    
    print("---------------------")
    
    # Just decode the predicted sequence
    truncated_sequence = generated_sequence[:, seq_len:]

    print(truncated_sequence[:, 0:args.b])
    
    decoded_sequence = tokenizer.decode(generated_sequence)
    decoded_sequence.dump_midi(path = Path(args.save_dir, "decoded_midi.mid"))
    print(decoded_sequence)
    
if __name__ == "__main__":
    main()
