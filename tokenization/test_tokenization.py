from miditok import REMI, TokenizerConfig  # here we choose to use REMI
from pathlib import Path

# Parameters for the tokenizer
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_chords": True,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": False,
    "use_programs": False,
    "num_tempos": 32,
    "tempo_range": (40, 250),
}

config = TokenizerConfig(**TOKENIZER_PARAMS)

# Creates the tokenizer
tokenizer = REMI(config)

# Tokenize a MIDI file
tokens = tokenizer(Path("/home/jv/Desktop/MIDI Dataset EDM", "Martin Garrix - Set Me Free.mid"))  # automatically detects Score objects, paths, tokens

# Convert to MIDI and save it
generated_midi = tokenizer(tokens)  # MidiTok can handle PyTorch/Numpy/Tensorflow tensors
generated_midi.dump_midi(Path("/home/jv/GitHub/Transformer_Code/tokenization/dump", "decoded_midi.mid"))

# ------------------ #
# Train the tokenizer