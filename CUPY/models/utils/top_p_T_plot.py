import cupy as cp
from pathlib import Path
import matplotlib
matplotlib.use("webAgg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import math  # to compute grid dimensions
import sys

sys.path.append('/csghome/hpdc04/Transformer_Code/CUPY/models/GoePT')
from Inference import * 


def create_midi_distribution(score):
    # Initialize event count dictionary
    event_counts = {
        "Tempos": len(score.tempos),
        "Time Signatures": len(score.time_signatures),
        "Key Signatures": len(score.key_signatures),
        "Notes": 0,
        "Controls": 0,
        "Pitch Bends": 0,
        "Pedals": 0
    }

    # Count per-track events
    for track in score.tracks:
        event_counts["Notes"] += len(track.notes)
        event_counts["Controls"] += len(track.controls)
        event_counts["Pitch Bends"] += len(track.pitch_bends)
        event_counts["Pedals"] += len(track.pedals)
        
    return event_counts

def main():
    # Generate parameter ranges for p and T
    p_values = cp.arange(0.1, 1.0, 0.1)    
    T_values = cp.array([0.6])
    file_path = Path("/csghome/hpdc04/Transformer_Code/Inference/input_files")
    
    # Initialize lists to store p_values and corresponding number of notes
    p_val_list = []
    num_notes_list = []
    
    # Loop over each combination of p and T
    for p_val in p_values:
        for t_val in T_values:
            # Generate sequences using your custom function
            generated_sequences, model_name, tokenizer = generate_sequence(
                "/csghome/hpdc04/Transformer_Code/checkpoints/loyal-chocolate-387_5225.json",
                "/csghome/hpdc04/Transformer_Code/Inference/input_files",
                "/csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/tokenizer_REMI_4096_FULL_False.json",
                256,
                p=p_val,
                T=t_val
            )
            
            # Iterate over the MIDI files in the input directory
            for idx, midifile in enumerate(list(file_path.glob("*mid"))):
                # For each file, select its corresponding generated sequence and prediction start index
                generated_sequence, prediction_start_idx = generated_sequences[idx]
                # Keep only the predicted tokens (2D shape preserved)
                predicted_sequence = generated_sequence[:, prediction_start_idx:]
                # Decode the token sequence back to a MIDI-like representation (string or list of events)
                decoded_sequence = tokenizer.decode(predicted_sequence)
                
                # Create a dictionary with frequency counts for each MIDI event type
                event_counts = create_midi_distribution(decoded_sequence)
                
                # Append p_val and number of notes to the lists
                p_val_list.append(p_val.get())
                num_notes_list.append(event_counts["Notes"])
    
    # Plot the scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(p_val_list, num_notes_list, color='blue')
    plt.xlabel("p_val")
    plt.ylabel("Number of Notes")
    plt.title("Number of Notes vs p_val")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("/csghome/hpdc04/Transformer_Code/tokenization_summary_plots/notes_vs_p_val_scatter.png")  # Save as PNG
    plt.close() 

if __name__ == "__main__":
    main()
