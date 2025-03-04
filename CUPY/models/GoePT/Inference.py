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
import io
import pretty_midi
import os


sys.path.append('/csghome/hpdc04/Transformer_Code/CUPY/models/utils')

from tokenize_data_fast import tokenize_dataset_to_bin
import config

def make_monophonic(pretty_midi_stream):
    for instrument in pretty_midi_stream.instruments:
        # Sort notes by start time
        instrument.notes.sort(key=lambda note: note.start)
        # Prepare a new list to store monophonic notes
        monophonic_notes = []
        # Initialize a list to track simultaneous notes
        chord_group = []
        # Tolerance for identifying simultaneous notes (in seconds)
        time_tolerance = 0.01

        # Process notes
        for note in instrument.notes:
            # If the chord group is empty, add the first note
            if not chord_group:
                chord_group.append(note)
            else:
                # If the current note starts within the time tolerance of the first note in the group, add to chord
                if abs(note.start - chord_group[0].start) <= time_tolerance:
                    chord_group.append(note)
                else:
                    # Process the chord group to keep only the highest note
                    top_note = max(chord_group, key=lambda n: n.pitch)
                    # If overlap occurs, cut the previous note's end time
                    if monophonic_notes and monophonic_notes[-1].end > top_note.start:
                        monophonic_notes[-1].end = top_note.start
                    # Add the highest note to the monophonic list
                    monophonic_notes.append(top_note)
                    # Start a new chord group with the current note
                    chord_group = [note]

        # Handle the last chord group
        if chord_group:
            top_note = max(chord_group, key=lambda n: n.pitch)
            # Adjust overlap for the last group
            if monophonic_notes and monophonic_notes[-1].end > top_note.start:
                monophonic_notes[-1].end = top_note.start
            monophonic_notes.append(top_note)

        # Replace the instrument's notes with the monophonic notes
        instrument.notes = monophonic_notes

def load_model(checkpoint_path, vocab_file, batch_size):
    with open(checkpoint_path, mode = 'r', encoding = 'utf-8') as weights:
        state_dict = json.load(weights)
    
    model = GoePT.from_state_dict(state_dict, batch_size=batch_size)
    
    tokenizer = config.tokenizer_name(params=vocab_file)
    print(tokenizer.vocab_size)
    return model, tokenizer

def softmax_with_temperature(logits, temperature=1, axis = -1):
    exp_logits = cp.exp(logits / temperature)
    return exp_logits / cp.sum(exp_logits, axis = axis, keepdims=True)


def top_p_sampling(prob_matrix, p=0.2):
    batch_size, vocab_size = prob_matrix.shape
    sampled_indices = cp.zeros((batch_size,1), dtype=int)
    
    for i in range(batch_size):
        probs = prob_matrix[i].copy()  # Copy to avoid modifying original data
        
        # Sort probabilities in descending order and get indices
        sorted_indices = cp.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        
        # Compute cumulative probabilities
        cumulative_probs = cp.cumsum(sorted_probs)
        
        # Find the cutoff where cumulative probability exceeds p
        # Use argmax to find the first index where condition is True; add 1 to include that index
        # If none exceed p, argmax returns 0 (False for all), cutoff becomes 1 (keep the first)
        cutoff = cp.argmax(cumulative_probs > p) + 1
        
        # Slice the top indices and probabilities up to the cutoff
        top_indices = sorted_indices[:cutoff]
        top_probs = sorted_probs[:cutoff]
        
        # Normalize the probabilities
        top_probs /= cp.sum(top_probs)
        
        # Sample from the top distribution
        sampled_index = cp.random.choice(top_indices, size=1, p=top_probs)
        sampled_indices[i] = sampled_index[0]
    
    return sampled_indices

def tokenize_input(midi_input, tokenizer):
    # Tokenize and return integer representation
    tokenized_input = tokenizer(midi_input)[0].ids
    return tokenized_input
    

def decode_tokens(midi_input, tokenizer):
    decoded_midi = tokenizer(midi_input)
    return decoded_midi


def generate_sequence(model_weights: str, input_sequences: Path, vocab_file, max_tokens : int, p : float, T : float) -> np.array: 
    
    model_name = os.path.splitext(os.path.basename(model_weights))[0]
    file_path = Path(input_sequences)
    number_of_files = len(list(file_path.glob("*.mid")))
    
    model, tokenizer = load_model(model_weights, vocab_file, batch_size = 1)
    seq_len = model.context_length
    print(tokenizer.vocab_size)
    pad_token = tokenizer.pad_token_id
    sos_token = tokenizer.special_tokens_ids[1]
    eos_token = tokenizer.special_tokens_ids[2]
    
    tokenized_input_sequences = []
    for midifile in list(file_path.glob("*mid")):
        t = tokenizer(midifile)[0].ids
        #sequence = [sos_token] + t + [eos_token]
        sequence = [sos_token] + t 
        
        tokenized_input_sequences.append(sequence)
        
    
    print(f"context_size: {seq_len}")

    
    
    generated_sequences = []
    for input_sequence in tokenized_input_sequences:
        length_input = len(input_sequence)
        if length_input > seq_len:
            print("Truncated")
            input_sequence = input_sequence[0:seq_len]
            input_sequence = cp.asanyarray(input_sequence)
            input_sequence.shape = (1, seq_len)
            
        elif length_input < seq_len:
            diff = seq_len - length_input
            input_sequence = diff * [pad_token] + input_sequence
            
            if seq_len == len(input_sequence):
                print("OK")
                input_sequence = cp.asanyarray(input_sequence)
                input_sequence.shape = (1, seq_len)
            print(input_sequence.shape)
            
        generated_sequence = cp.asanyarray(input_sequence)
        
        prediction_start_idx = seq_len
        print("\n --------------------------------------------------------------- \n")
        for idx in range(max_tokens):
            logits, _ = model.forward(input_sequence, targets = None)
            logits = cp.squeeze(logits, axis = 1) # Transform to 2D shape b, vocab
            predictions = softmax_with_temperature(logits, temperature = T)
            next_tokens = top_p_sampling(predictions, p = p) 
            if next_tokens == eos_token:
                print("Encountered an EOS token. Stopping prediction")
                break
            # Append the predicted token to the sequence
            generated_sequence = cp.concatenate([generated_sequence, next_tokens], axis=1) # add new column
            print(generated_sequence)
        # convert back to numpy
        generated_sequence = generated_sequence.get()
        generated_sequences.append((generated_sequence, prediction_start_idx))
        
    return generated_sequences, model_name, tokenizer

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
    parser.add_argument('--p', type = float)
    parser.add_argument('--temperature', type = float)

    args = parser.parse_args()
    
    generated_sequences, model_name, tokenizer = generate_sequence(args.weights,
                                                       args.input,
                                                       args.vocab_file,
                                                       args.b,
                                                       p = args.p,
                                                       T = args.temperature)
    
    
    print("---------------------")
    # Just decode the predicted sequence
    file_path = Path(args.input)

    for idx, midifile in enumerate(list(file_path.glob("*mid"))):
        fileName = f"{midifile.name}_{args.p}_{args.temperature}_{model_name}"
        generated_sequence, prediction_start_idx = generated_sequences[idx] # Here we preserve the 2D shape and only take predicted tokens
        print(generated_sequence[:, prediction_start_idx:])
        predicted_sequence = generated_sequence[:, prediction_start_idx:]
        
        
        decoded_sequence = tokenizer.decode(predicted_sequence)

    
        decoded_sequence.dump_midi(path = Path(args.save_dir, fileName))


        pretty_midi_stream = pretty_midi.PrettyMIDI(midi_file = str(Path(args.save_dir, fileName)))
        make_monophonic(pretty_midi_stream)
        pretty_midi_stream.write(str(Path(args.save_dir, fileName)))

        print(f"{fileName}: \n \n {predicted_sequence}")
    
    print(f"Generated tokens with p = {args.p} and Temperature = {args.temperature}")
    
if __name__ == "__main__":
    main()
