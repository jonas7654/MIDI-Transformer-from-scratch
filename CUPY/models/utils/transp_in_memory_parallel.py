from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# These two are used for storing temporary midi files in main memory
import io

import os
from music21 import converter, key, interval, tempo, midi, converter
import pretty_midi


def check_time_sig(
    pretty_midi_stream,
):  # checks whether the time signature is 4/4 and doesn't change
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
    key_signature = music21_stream.analyze("key")
    if key_signature.mode == "major":
        target_key = key.Key("C")
    elif key_signature.mode == "minor":
        target_key = key.Key("A")
    else:
        return None  # Skip non-major/minor modes
    transpose_interval = interval.Interval(key_signature.tonic, target_key.tonic)
    if (
        transpose_interval.semitones > 6
    ):  # the closer way to C/Am is by using the reversed complement of the interval
        transpose_interval = transpose_interval.complement.reverse()
    if (
        transpose_interval.semitones < -6
    ):  # with descending intervals the complement is always automatically reversed
        transpose_interval = transpose_interval.complement
    return transpose_interval.semitones


def transpose_midi(pretty_midi_stream, interval):
    # Transpose the MIDI using pretty_midi
    transposed_midi = pretty_midi.PrettyMIDI()

    # Loop over all instruments and transpose their notes
    for instrument in pretty_midi_stream.instruments:
        transposed_instrument = pretty_midi.Instrument(
            instrument.program, is_drum=instrument.is_drum
        )
        for note in instrument.notes:
            transposed_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch + interval,
                start=note.start,
                end=note.end,
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

    new_instruments = []

    for instrument in pretty_midi_stream.instruments:
        # Check if the instrument contains any bass notes (pitch <= 40)
        if any(note.pitch <= 40 for note in instrument.notes):
            rmv_bass_count += 1  # Count this as a bass track removed
        else:
            new_instruments.append(instrument)  # Keep non-bass tracks

    # Update the MIDI object's instruments list
    pretty_midi_stream.instruments = new_instruments

    return rmv_bass_count


def process_file(file, output_folder, counter):
    try:
        pretty_midi_stream = pretty_midi.PrettyMIDI(str(file))
        
        if not check_time_sig(pretty_midi_stream):
            return {"counter": counter, "bad_time_sig": 1, "bad_mode_total": 0, "drum_total": 0, "rmv_bass_total": 0}

        filtered_midi, drum_count = filter_out_drums(pretty_midi_stream)

        # Save filtered MIDI to memory buffer
        memory_buffer = io.BytesIO()
        filtered_midi.write(memory_buffer)
        memory_buffer.seek(0)

        music21_stream = converter.parse(memory_buffer.read())
        transpose_interval = detect_transpose_interval(music21_stream)

        if transpose_interval is None:
            return {"counter": counter, "bad_time_sig": 0, "bad_mode_total": 1, "drum_total": drum_count, "rmv_bass_total": 0}

        transposed_stream = transpose_midi(filtered_midi, transpose_interval)
        make_monophonic(transposed_stream)
        rmv_bass_count = remove_bass_tracks(transposed_stream)

        # Save the transposed file
        output_path = output_folder / f"{file.stem}_transmon{counter}.mid"
        transposed_stream.write(str(output_path))

        return {
            "counter": counter,
            "bad_time_sig": 0,
            "bad_mode_total": 0,
            "drum_total": drum_count,
            "rmv_bass_total": rmv_bass_count,
        }
    except Exception as e:
        print(f"Skipping file {file}, due to exception: {e}")
        return {"counter": counter, "bad_time_sig": 0, "bad_mode_total": 0, "drum_total": 0, "rmv_bass_total": 0}


def main():
    current_dir = os.getcwd()
    name_of_raw_midi_folder = "clean_midi/"
    folder_path = Path(
        os.path.dirname(current_dir), "datasets", name_of_raw_midi_folder
    )
    output_folder = Path(os.path.dirname(current_dir), "datasets/transposed_midi")
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Input Folder: {folder_path}")
    print(f"Output Folder: {output_folder}")

    counter = 0
    drum_total = 0
    bad_mode_total = 0
    bad_time_sig = 0
    rmv_bass_total = 0

    midi_files_list = list(folder_path.rglob("*.mid"))
    print(f"Found {len(midi_files_list)} MIDI files.")

    results = []
    with ProcessPoolExecutor(max_workers=96) as executor:
        futures = [
            executor.submit(process_file, file, output_folder, idx + 1)
            for idx, file in enumerate(midi_files_list)
        ]
        for future in futures:
            results.append(future.result())

    # Aggregate results


    total_files = len(results)
    total_bad_time_sig = sum(res["bad_time_sig"] for res in results)
    total_bad_mode = sum(res["bad_mode_total"] for res in results)
    total_drums_removed = sum(res["drum_total"] for res in results)
    total_bass_removed = sum(res["rmv_bass_total"] for res in results)

    # Print aggregated statistics
    print(f"\n{total_files} files processed.")
    print(f"Skipped {total_bad_time_sig} files due to time signature.")
    print(f"Skipped {total_bad_mode} files due to non-major/minor mode.")
    print(f"Removed {total_drums_removed} drum tracks.")
    print(f"Removed {total_bass_removed} bass tracks.")


if __name__ == "__main__":
    main()
