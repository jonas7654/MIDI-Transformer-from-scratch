from mido import MidiFile, MidiTrack, Message, MetaMessage
from pathlib import Path

# io for storing directly into main memory
# concurrent for parallel execution
import io
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

import os

from music21 import converter, key, interval, tempo, midi
import pretty_midi

def add_tempo_with_mido(output_path, tempo_bpm):
    # Convert BPM to microseconds per beat
    microseconds_per_beat = int(60 * 1_000_000 / tempo_bpm)

    # Create a new MIDI file with Mido
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Add a tempo change
    track.append(MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))

def get_tempo(pretty_midi_stream):
    tempo_times, tempos = pretty_midi_stream.get_tempo_changes()
    return tempos

def check_time_sig(pretty_midi_stream): #checks whether the time signature is 4/4 and doesn't change
    # Get the list of time signature changes
    time_signature_changes = pretty_midi_stream.time_signature_changes
    tempos = get_tempo(pretty_midi_stream)
    # Check if there are no time signature changes or only one constant time signature
    if len(time_signature_changes) == 1 and len(tempos) == 1:
        time_signature = time_signature_changes[0]
        # Check if the time signature is 4/4
        if time_signature.denominator == 4:
            if time_signature.numerator == 4:
                return True
            if time_signature.numerator == 2:
                pretty_midi_stream.time_signature_changes[0].numerator = 4
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


def remove_bass_tracks(pretty_midi_stream):
    
    # Initialize the count of removed bass tracks
    rmv_bass_count = 0
    
    # Filter out the instruments containing bass notes (pitch <= 40)
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

def greedy_collect(pretty_midi_stream, tempo, output_path_pref):
    tracks_coll = 0
    tracks_skipped = 0
    for i, instr in enumerate(pretty_midi_stream.instruments):
        if instr.notes: #this might be this case with removed bass tracks f.e.
            midi = pretty_midi.PrettyMIDI(initial_tempo=120) #create a new midi file for writing
            seconds_per_beat = 60.0 / tempo
            seconds_per_bar = 4 * seconds_per_beat #4 beats in one bar (4/4, we treat 2/4 just like 4/4 here)
            bar_duration = seconds_per_bar * 8  # Duration of 8 bars in seconds

            first_note_time = min(note.start for note in instr.notes) 

            cutoff_time = first_note_time + bar_duration #cutoff point 8 bars after first note

            new_instrument = pretty_midi.Instrument(program=instr.program, is_drum=instr.is_drum)
            new_instrument.notes = [note for note in instr.notes if first_note_time <= note.start < cutoff_time] #notes in 8 bars from first note

            bars = [[] for _ in range(8)]

            for note in new_instrument.notes:
                # Find the bar index (0-based index for 8 bars)
                bar_index = int((note.start - first_note_time) // seconds_per_bar)
                if 0 <= bar_index < 8:  # Ensure it's within the 8 bars
                    bars[bar_index].append(note)

            bars_with_notes = sum(1 for bar in bars if len(bar) > 0)

            if bars_with_notes >= 5: #something must be played in at least 5 out of 8 bars
                for note in new_instrument.notes:
                    note.start = max(0, note.start - first_note_time)
                    note.end = max(0, min(note.end - first_note_time, cutoff_time - first_note_time))
                    note.start *= tempo/120 #Stretch to 120 bpm from original tempo
                    note.end *= tempo/120
                if len(new_instrument.notes) >= 12: #there must be at least 12 notes in file, otherwise the hook is pretty boring
                    midi.instruments.append(new_instrument)
                    op_new = str(output_path_pref) + f"_track{i}.mid"
                    midi.write(str(op_new))
                    #print(f"8 bar excerpt saved to: {op_new}")
                    tracks_coll += 1
                else: tracks_skipped += 1
            else: tracks_skipped += 1
    return tracks_coll, tracks_skipped


def process_file(file, output_folder, counter):
    try:
        pretty_midi_stream = pretty_midi.PrettyMIDI(str(file))
        
        if not check_time_sig(pretty_midi_stream):
            return {"counter": counter,
                    "tracks_coll": 0,
                    "bad_time_sig": 1,
                    "bad_mode_total": 0,
                    "drum_total": 0,
                    "rmv_bass_total": 0,
                    "note_skipped" : 0
                    }

        filtered_midi, drum_count = filter_out_drums(pretty_midi_stream)

        """
        @Author: Jonas
                
        We want to save the temporary pretty midi output to the main memory instead of writing it to the Disk.
        We do that by creating a buffer in main memory and writing the filtered_midi directly to that buffer.
                
        Docs: https://docs.python.org/3/library/io.html
        Section: Binary I/O
        """
        
        # Save filtered MIDI to memory buffer
        memory_buffer = io.BytesIO()
        filtered_midi.write(memory_buffer)
        memory_buffer.seek(0)

        music21_stream = converter.parse(memory_buffer.read())
        transpose_interval = detect_transpose_interval(music21_stream)

        if transpose_interval is None:
            return {"counter": counter,
                    "bad_time_sig": 0,
                    "bad_mode_total": 1,
                    "drum_total": drum_count,
                    "rmv_bass_total": 0,
                    "tracks_coll" : 0,
                    "note_skipped": 0}

        transposed_stream = transpose_midi(filtered_midi, transpose_interval)
        make_monophonic(transposed_stream)
        rmv_bass_count = remove_bass_tracks(transposed_stream)
        
        curr_tempo = get_tempo(pretty_midi_stream)[0]
        output_path_prefix = output_folder / f"{file.stem}"
        tracks_coll, tracks_skipped = greedy_collect(transposed_stream, curr_tempo, output_path_prefix)

        # Save the transposed file
        output_path = output_folder / f"{file.stem}_transmon{counter}.mid"
        transposed_stream.write(str(output_path))
        

        return {
            "counter": counter,
            "bad_time_sig": 0,
            "bad_mode_total": 0,
            "drum_total": drum_count,
            "rmv_bass_total": rmv_bass_count,
            "tracks_coll" : tracks_coll,
            "note_skipped" : tracks_skipped
        }
    except Exception as e:
        print(f"Skipping file {file}, due to exception: {e}")
        return {
            "counter": counter,
            "tracks_coll": 0,
            "bad_time_sig": 0,
            "bad_mode_total": 0,
            "drum_total": 0,
            "rmv_bass_total": 0,
            "note_skipped": 0,
        }

        
"""
Author: Jonas
NOTES:
I removed the global counters since this could be a problem due to shared memory. 
I locally count within the process_file() function and then aggregate these at the end
"""
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

    midi_files_list = list(folder_path.rglob("*.mid"))
    print(f"Found {len(midi_files_list)} MIDI files.")
    
    results = []
    
    #with ProcessPoolExecutor(max_workers = multiprocessing.cpu_count()) as executor:
    #    futures = [
    #        executor.submit(process_file, file, output_folder, idx + 1)
    #        for idx, file in enumerate(midi_files_list)
    #    ]
    #    for future in futures:
    #        results.append(future.result())
    
    with ProcessPoolExecutor(max_workers=48) as executor:
        futures = [
            executor.submit(process_file, file, output_folder, idx + 1)
            for idx, file in enumerate(midi_files_list)
        ]
        for future in tqdm(futures, desc="Processing MIDI files", total=len(futures)):
            results.append(future.result())

    # Aggregate results:
    total_files = len(results)
    total_bad_time_sig = sum(res["bad_time_sig"] for res in results)
    total_bad_mode = sum(res["bad_mode_total"] for res in results)
    total_drums_removed = sum(res["drum_total"] for res in results)
    total_bass_removed = sum(res["rmv_bass_total"] for res in results)
    track_total = sum(res["tracks_coll"] for res in results)
    note_skipped_total = sum(res["note_skipped"] for res in results)
            
    print(f"\n {total_files} files processed.")
    print(f"Removed {total_bad_mode} midi files due to non-major/minor mode.")
    print(f"Skipped {total_bad_time_sig} files in total due to time signature.")
    print(f"Skipped {total_bad_mode + total_bad_time_sig} files in total.")

    print(f"\n{track_total} 8 bar excerpts collected.")
    print(f"Removed {total_drums_removed} drum tracks in total.")
    print(f"Removed {total_bass_removed} bass tracks in total.")
    print(f"Skipped {note_skipped_total} tracks in total due to note density.")
    print(f"Skipped {total_drums_removed + total_bass_removed + note_skipped_total} tracks in total.")
    print(f"CPU cores: {multiprocessing.cpu_count()}")
    
if __name__ == "__main__":
    main()
