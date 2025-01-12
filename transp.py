from pathlib import Path
import io
from music21 import converter, key, interval, tempo
import pretty_midi

def check_time_sig(pretty_midi_stream): #checks whether the time signature is 4/4 and doesn't change
    # Get the list of time signature changes
    time_signature_changes = pretty_midi_stream.time_signature_changes
    # Check if there are no time signature changes or only one constant time signature
    if len(time_signature_changes) == 1:
        time_signature = time_signature_changes[0]
        # Check if the time signature is 4/4
        if time_signature.numerator == 4 and time_signature.denominator == 4:
            return True
    # If there are multiple time signatures or it's not 4/4, return False
    return False

def filter_out_drums(pretty_midi_stream):
    filtered_midi = pretty_midi.PrettyMIDI()
    drum_count = 0
    for instr in pretty_midi_stream.instruments:
        if instr.is_drum:
            drum_count += 1
            continue
        filtered_midi.instruments.append(instr)
    return filtered_midi, drum_count

def detect_transpose_interval(music21_stream):
    key_signature = music21_stream.analyze('key')
    if key_signature.mode == "major":
        target_key = key.Key("C")
    elif key_signature.mode == "minor":
        target_key = key.Key("A")
    else:
        return None  # Skip non-major/minor modes
    transpose_interval = interval.Interval(key_signature.tonic, target_key.tonic)
    if transpose_interval.semitones > 6: #the closer way to C/Am is by using the reversed complement of the interval
        transpose_interval = transpose_interval.complement.reverse()
    if transpose_interval.semitones < -6: #with descending intervals the complement is always automatically reversed
        transpose_interval = transpose_interval.complement
    return transpose_interval.semitones

def transpose_midi(pretty_midi_stream, interval):
    # Transpose the MIDI using pretty_midi
    transposed_midi = pretty_midi.PrettyMIDI()

    # Loop over all instruments and transpose their notes
    for instrument in pretty_midi_stream.instruments:
        transposed_instrument = pretty_midi.Instrument(instrument.program, is_drum=instrument.is_drum)
        for note in instrument.notes:
            transposed_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch + interval,
                start=note.start,
                end=note.end
            )
            transposed_instrument.notes.append(transposed_note)
        transposed_midi.instruments.append(transposed_instrument)

    return transposed_midi

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

import pretty_midi

def remove_bass_tracks(pretty_midi_stream):
    
    # Initialize the count of removed bass tracks
    rmv_bass_count = 0
    
    # Filter out the instruments containing bass notes (pitch <= 53)
    new_instruments = []
    
    for instrument in pretty_midi_stream.instruments:
        # Check if the instrument contains any bass notes (pitch <= 53)
        if any(note.pitch <= 40 for note in instrument.notes):
            rmv_bass_count += 1  # Count this as a bass track removed
        else:
            new_instruments.append(instrument)  # Keep non-bass tracks
    
    # Update the MIDI object's instruments list
    pretty_midi_stream.instruments = new_instruments
    
    return rmv_bass_count


def main():
    folder_path = Path("AP_Proj/test_data")
    output_folder = Path("D:/PythonWorkspace/AP_Proj/transpose_test")
    output_folder.mkdir(parents=True, exist_ok=True)

    counter = 1
    drum_total = 0
    bad_mode_total = 0
    bad_time_sig = 0
    rmv_bass_total = 0
    for file in folder_path.glob("*.mid"):
        try:
            pretty_midi_stream = pretty_midi.PrettyMIDI(str(file)) #open file as pretty midi stream

            if check_time_sig(pretty_midi_stream):

                filtered_midi, drum_count = filter_out_drums(pretty_midi_stream)
                drum_total += drum_count #add number of removed drums to total
                    
                # Convert mido to music21 Stream
                output_file = f"AP_Proj/non_drum_data/{file.stem}_non_drum_file{counter}.mid"
                filtered_midi.write(filename=output_file)

                # Load the saved MIDI file into a music21 Stream
                music21_stream = converter.parse(output_file)

                transpose_interval = detect_transpose_interval(music21_stream)
                if transpose_interval is None:
                    bad_mode_total += 1
                    continue
                
                print(f"Transposing by {transpose_interval} semitones")
                transposed_stream = transpose_midi(filtered_midi, transpose_interval)

                make_monophonic(transposed_stream)
                
                rmv_bass_count = remove_bass_tracks(transposed_stream)
                rmv_bass_total += rmv_bass_count

                # Save the transposed file
                output_path = output_folder / f"{file.stem}_transmon{counter}.midi"
                counter += 1
                transposed_stream.write(str(output_path))
                print(f"Transposed MIDI saved to: {output_path}")
            
            else:
                bad_time_sig += 1
                print(f"Time signature not accepted. Skipping file {counter}.")
                counter += 1
                continue
        except Exception as e:
            print(f"Skipping file, due to this exception: {e}")
    print(f"\n{counter - 1} files processed.")
    print(f"Skipped {bad_time_sig} files in total due to time signature.")
    print(f"Removed {drum_total} drum tracks in total.")
    print(f"Removed {bad_mode_total} midi files due to non-major/minor mode.")
    print(f"Removed {rmv_bass_total} bass tracks in total.")

if __name__ == "__main__":
    main()
