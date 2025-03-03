from miditok import REMI
import numpy as np

# Define the range of tokens you're interested in
start_index = 0
end_index = 4096

# Path to the tokenizer vocabulary file
vocab_file = "/csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/tokenizer_REMI_4096_FULL_False.json"
# Output text file
output_file = "/csghome/hpdc04/Transformer_Code/tokens_txt/vocab_midi_output.txt"

def main():
    # Initialize tokenizer
    tokenizer = REMI(params=vocab_file)
    
    # Initialize note counter
    total_notes = 0

    # Open output file for writing
    with open(output_file, "w") as f:
        # Iterate through the specified range of token IDs
        for token_id in range(start_index, end_index):
            token_array = np.array([[token_id]], dtype=np.int32)  # Convert token to numpy array
            
            try:
                score = tokenizer.decode(token_array)  # Decode into Score object
                
                # Extract events from the decoded score
                events = [str(event) for track in score.tracks for event in track.notes]  # Extract note events
                
                # Count the number of notes
                total_notes += len(events)
                
                if not events:
                    events = [str(event) for event in score.tempos]  # Try extracting tempo events
                
                event_description = ", ".join(events) if events else "Unknown Event"
                
                # Write token ID and event description to file
                f.write(f"Token {token_id}: {event_description}\n")
            
            except Exception as e:
                f.write(f"Token {token_id}: Error - {str(e)}\n")  # Handle errors
            
            # Print progress every 10 tokens
            if token_id % 10 == 0:
                print(f"Processed {token_id}/{end_index} tokens...")

    print(f"Token descriptions written to {output_file}")
    print(f"Total number of notes written: {total_notes}")

if __name__ == "__main__":
    main()