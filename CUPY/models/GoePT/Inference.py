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

def softmax_with_temperature(logits, temperature=1.5, axis = -1):
    exp_logits = cp.exp(logits / temperature)
    return exp_logits / cp.sum(exp_logits, axis = axis)

def top_p_sampling(prob_matrix, p=0.4):
    
    batch_x, batch_y, vocab_size = prob_matrix.shape
    sampled_indices = cp.zeros((batch_x, batch_y), dtype=int)

    for i in range(batch_x):
        for j in range(batch_y):
            probs = prob_matrix[i, j]

            # Sort probabilities and get sorted indices
            sorted_indices = cp.argsort(probs)[::-1]  # Descending order
            sorted_probs = probs[sorted_indices]

            # Compute cumulative probabilities
            cumulative_probs = cp.cumsum(sorted_probs)

            # Find the cutoff where cumulative probability exceeds p
            cutoff = cp.argmax(cumulative_probs > p) + 1  # Keep at least one token

            # Get the subset of valid token indices and their probabilities
            top_indices = sorted_indices[:cutoff]
            top_probs = sorted_probs[:cutoff]
            # Normalize probabilities
            top_probs /= cp.sum(top_probs)

            # Sample from the filtered distribution
            sampled_indices[i, j] = cp.random.choice(top_indices, size=1, p=top_probs)[0]

        return cp.asarray(sampled_indices)

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
    
    
    

    
    # Remove the EOS token : TODO : dont duplicate tokens at the end
    generated_sequence = cp.asanyarray(tokenized_data.copy())
    print(f"context_size: {seq_len}")
    print(f"Input sequence shape: \n {generated_sequence.shape}")
    
    
    
    for idx in range(args.b):
        input_context = generated_sequence[:, -seq_len:]
        logits, _ = model.forward(input_context, targets = None)
        predictions = softmax_with_temperature(logits, temperature = 1)
        print(f"Predictions shape: {predictions.shape}")
        next_tokens = top_p_sampling(predictions)  # axis -1 uses the last axis which is the vocabulary
        # Append the predicted token to the sequence
        generated_sequence = cp.concatenate([generated_sequence, next_tokens], axis=1) # add new column
        
    # convert back to numpy
    generated_sequence = generated_sequence.get()
    
    print("---------------------")
    
    # Just decode the predicted sequence
    for idx, midifile in enumerate(list(file_path.glob("*mid"))):
        fileName = midifile.name
        predicted_sequence = generated_sequence[idx:idx+1, seq_len:] # Here we preserve the 2D shape

        decoded_sequence = tokenizer.decode(predicted_sequence)
        decoded_sequence.dump_midi(path = Path(args.save_dir, fileName))
        print(f"{fileName}: {predicted_sequence} \n \n")
    
    
if __name__ == "__main__":
    main()
