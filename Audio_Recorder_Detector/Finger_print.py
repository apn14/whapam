import os
import sys
import libs
import libs.fingerprint as fingerprint
import argparse
from itertools import zip_longest
from termcolor import colored
from libs.reader_microphone import MicrophoneReader
from libs.visualiser_console import VisualiserConsole as visual_peak
from libs.visualiser_plot import VisualiserPlot as visual_plot

# Path to the folder containing reference audio files
REFERENCE_SONG_FOLDER = r"C:\Audio_Recorder_Detector\output_wav_folder"

# Function to compute fingerprints for all reference songs
def compute_reference_fingerprints(reference_folder):
    """
    Compute fingerprints for all .wav files in the reference folder.
    """
    fingerprints = {}
    for file in os.listdir(reference_folder):
        if file.endswith('.wav'):
            file_path = os.path.join(reference_folder, file)
            print(colored(f"Processing file: {file}", 'yellow'))
            try:
                samples, Fs = fingerprint.read_wav(file_path)
                hashes = list(fingerprint.fingerprint(samples, Fs=Fs))
                fingerprints[file] = hashes
            except Exception as e:
                print(colored(f"Error processing {file}: {e}", 'red'))
    return fingerprints

# Function to find matches between query audio and reference fingerprints
def find_matches(query_hashes, reference_fingerprints):
    """
    Compare the hashes of the query audio with the precomputed reference fingerprints.
    """
    matches = []
    for song_name, song_hashes in reference_fingerprints.items():
        match_offsets = [
            (song_name, offset - ref_offset)
            for (hash_val, ref_offset) in song_hashes
            for (q_hash, offset) in query_hashes
            if q_hash == hash_val
        ]
        matches.extend(match_offsets)
    return matches

# Function to align matches and find the best match
def align_matches(matches):
    """
    Align matches and find the best match by analyzing offset differences.
    """
    diff_counter = {}
    largest = 0
    largest_count = 0
    song_id = None

    for song_name, diff in matches:
        if diff not in diff_counter:
            diff_counter[diff] = {}

        if song_name not in diff_counter[diff]:
            diff_counter[diff][song_name] = 0

        diff_counter[diff][song_name] += 1

        if diff_counter[diff][song_name] > largest_count:
            largest = diff
            largest_count = diff_counter[diff][song_name]
            song_id = song_name

    return {
        "SONG_NAME": song_id,
        "CONFIDENCE": largest_count,
        "OFFSET": int(largest),
    }

# Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Audio fingerprinting from local directory.")
    parser.add_argument('-s', '--seconds', type=int, required=True, help="Duration of recording in seconds.")
    args = parser.parse_args()

    seconds = args.seconds
    chunksize = 2**12
    channels = 2

    # Precompute fingerprints for reference songs
    print(colored("Computing fingerprints for reference songs...", 'cyan'))
    reference_fingerprints = compute_reference_fingerprints(REFERENCE_SONG_FOLDER)
    print(colored(f"Computed fingerprints for {len(reference_fingerprints)} songs.", 'green'))

    # Record audio from microphone
    reader = MicrophoneReader(None)
    reader.start_recording(seconds=seconds, chunksize=chunksize, channels=channels)

    print(colored("Recording started...", 'cyan'))
    buffer_size = int(reader.rate / reader.chunksize * seconds)
    for i in range(buffer_size):
        reader.process_recording()

    reader.stop_recording()
    print(colored("Recording stopped. Processing recorded audio...", 'cyan'))

    # Process recorded audio
    recorded_data = reader.get_recorded_data()[0]
    query_hashes = list(fingerprint.fingerprint(recorded_data, Fs=fingerprint.DEFAULT_FS))

    # Match against reference fingerprints
    matches = find_matches(query_hashes, reference_fingerprints)
    if matches:
        print(colored(f"Found {len(matches)} total matches.", 'green'))
        song = align_matches(matches)
        print(colored(f"Best match: {song['SONG_NAME']}", 'green'))
        print(colored(f"Confidence: {song['CONFIDENCE']} matches", 'green'))
        print(colored(f"Offset: {song['OFFSET']} samples", 'green'))
    else:
        print(colored("No matches found.", 'red'))
