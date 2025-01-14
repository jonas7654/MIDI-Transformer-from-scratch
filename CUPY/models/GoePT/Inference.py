from model import GoePT, compute_gradient
import json
from icecream import ic 
from miditok import REMI
import cupy as cp
import argparse


def load_model(checkpoint_path, vocab_file):
    with open(checkpoint_path, mode = 'r', encoding = 'utf-8') as weights:
        state_dict = json.load(weights)
    
    model = GoePT.from_state_dict(state_dict)
    tokenizer = REMI(params=vocab_file) # NOTE: CHANGE THIS IF THE TOKENIZER CHANGES
    print(model)
    
    print("Do i get here?")
    return model, tokenizer

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
    parser.add_argument('--i', type = str,
                        help = "Path to the Input midi file") 
    parser.add_argument('--context-length', type = int)
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.weights, args.vocab_file)

    print("Hello?")

    ic(model)
    ic(tokenizer)
    
    # Tokenize the input
    tok_input = tokenize_input(args.i, tokenizer)
    
    # forward the tok_input to the pre-trained model
    logits, _ = model.forward(tok_input, targets = None)
    
    print(logits)
    
if __name__ == "__main__":
    main()
