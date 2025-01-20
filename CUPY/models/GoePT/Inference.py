from model import GoePT, compute_gradient
from layers import Softmax
import json
from icecream import ic 
from miditok import REMI
from miditok.utils import merge_scores
import cupy as cp
import numpy as np
import argparse
from pathlib import Path
import sys


sys.path.append('/csghome/hpdc04/Transformer_Code/CUPY/models/utils')

from tokenize_data_fast import tokenize_dataset_to_bin
import config

def load_model(checkpoint_path, vocab_file):
    with open(checkpoint_path, mode = 'r', encoding = 'utf-8') as weights:
        state_dict = json.load(weights)
    
    model = GoePT.from_state_dict(state_dict)
    
    tokenizer = REMI(params=vocab_file) # NOTE: CHANGE THIS IF THE TOKENIZER CHANGES
    print(model)
    
    return model, tokenizer

def softmax_with_temperature(logits, temperature=1.0, Softmax=None):
    logits = logits / temperature
    return Softmax.forward(logits)

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
    parser.add_argument('--context-length', type = int)
    
    args = parser.parse_args()
    
    file_path = Path(args.input)
    
    model, tokenizer = load_model(args.weights, args.vocab_file)
    seq_len = model.context_length

    
    # Tokenize the input
    tokenized_data = tokenizer.tokenize_dataset_to_bin(files_paths = file_path,
                                      verbose = True,
                                      seq_length = seq_len,
                                      manually_add_sos_eos = config.manually_set_sos_eos_trunc)
    
    # :TODO adjust model.batch_size to fit the passed batch

    tokenized_data_minus_last = np.zeros((len(tokenized_data), seq_len), dtype=tokenized_data.dtype)
    tokenized_data_minus_last[:, 1:] = tokenized_data[:, :seq_len - 1]
    print(tokenized_data_minus_last)
    
    # forward the tok_input to the pre-trained model
    logits, _ = model.forward(tokenized_data_minus_last, targets = None)

    # Apply softmax :TODO : Add Temperature ?
    softmax = Softmax(axis = 0) # use rows
    predictions = softmax_with_temperature(logits, temperature=0.5, Softmax=softmax)
    predictions = cp.argmax(predictions, axis = -1) # axis -1 uses the last axis which is the vocabulary
    
    # convert back to numpy
    predictions = predictions.get()
    
    print("---------------------")
    print("\n", predictions, "\n", tokenized_data[:,(seq_len - 1)]) 
    print(predictions.shape)
    
    decoded_predictions = tokenizer.decode(predictions)
   

if __name__ == "__main__":
    main()
