import os
from flask import Flask, render_template, request
import librosa
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
from scipy.signal import butter, lfilter

# Paths and Folders
REFERENCE_SONG_FOLDER = r"C:\Audio_Recorder_Detector\output_wav_folder"
RECORDINGS_FOLDER = "recordings"
UPLOAD_FOLDER = "uploads"
os.makedirs(RECORDINGS_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Recording parameters
SAMPLE_RATE = 16000  # Hz
DURATION = 20  # seconds to record or extract
LOW_CUTOFF = 1  # Hz (Band-pass filter lower cutoff frequency)
HIGH_CUTOFF = 3000  # Hz (Band-pass filter upper cutoff frequency)

# Create a mapping between song names and their file paths
song_file_mapping = {
    os.path.splitext(song_file)[0]: os.path.join(REFERENCE_SONG_FOLDER, song_file)
    for song_file in os.listdir(REFERENCE_SONG_FOLDER)
    if song_file.endswith('.wav')
}

# Band-pass filter implementation
def band_pass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a band-pass filter to the input data.
    - data: The input audio data.
    - lowcut: The lower cutoff frequency for the band.
    - highcut: The upper cutoff frequency for the band.
    - fs: Sampling rate of the audio.
    - order: The order of the filter (higher = sharper cutoff).
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data

# Function to record audio from the microphone and filter it
def record_audio(output_file):
    """
    Records audio from the microphone, applies a band-pass filter, and saves it.
    """
    print(f"Recording for {DURATION} seconds...")
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()  # Wait for the recording to finish
    print("Recording complete. Applying band-pass filter...")

    # Apply band-pass filter
    filtered_audio = band_pass_filter(audio_data.flatten(), LOW_CUTOFF, HIGH_CUTOFF, SAMPLE_RATE)
    filtered_audio = np.int16(filtered_audio / np.max(np.abs(filtered_audio)) * 32767)  # Normalize to int16

    # Save as a .wav file
    wav.write(output_file, SAMPLE_RATE, filtered_audio)
    print(f"Filtered audio saved to {output_file}")

# Function to extract audio features
def extract_features(audio, sr=16000, n_mfcc=20, is_path=True):
    """
    Extract MFCCs and Chroma features from an audio file or array.
    """
    if is_path:
        audio, _ = librosa.load(audio, sr=sr)  # Load from path
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    return mfcc, chroma

# Function to calculate similarity
def correlate_features(chunk_features, ref_features):
    """
    Compute the cross-correlation between chunk features and reference features.
    """
    chunk_flat = chunk_features.flatten()
    ref_flat = ref_features.flatten()
    return np.correlate(chunk_flat, ref_flat, mode='valid').max()

# Function to match recorded or uploaded audio to the first 10 seconds of full-length songs
def match_full_audio_to_songs(full_audio_path, song_file_mapping, sr=16000, duration=20):
    """
    Match a full-length audio file against the first 'duration' seconds of each reference song.
    """
    # Load the uploaded audio file
    uploaded_audio, _ = librosa.load(full_audio_path, sr=sr)
    uploaded_features, _ = extract_features(uploaded_audio[:duration * sr], sr=sr, is_path=False)

    song_scores = {}
    for song_name, song_path in song_file_mapping.items():
        # Load only the first 'duration' seconds of the reference song
        ref_audio, _ = librosa.load(song_path, sr=sr, duration=duration)
        ref_features, _ = extract_features(ref_audio, sr=sr, is_path=False)
        score = correlate_features(uploaded_features, ref_features)
        song_scores[song_name] = score

    best_match = max(song_scores, key=song_scores.get)
    return best_match, song_scores

# Flask application
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/record", methods=["POST"])
def record():
    output_file = os.path.join(RECORDINGS_FOLDER, "recorded_audio.wav")
    record_audio(output_file)
    return "Recording complete. File saved as 'recorded_audio.wav'. You can now process this file."

@app.route("/process", methods=["POST"])
def process():
    recorded_file = os.path.join(RECORDINGS_FOLDER, "recorded_audio.wav")
    if not os.path.exists(recorded_file):
        return "No recording found. Please record audio first.", 400

    best_match, song_scores = match_full_audio_to_songs(recorded_file, song_file_mapping)
    if best_match:
        match_results = {
            "predicted_song": best_match,
            "match_scores": song_scores
        }
        return render_template("result.html", results=match_results)
    else:
        return render_template("result.html", results={"predicted_song": "No matches found", "match_scores": {}})

@app.route("/match", methods=["POST"])
def match():
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    best_match, song_scores = match_full_audio_to_songs(file_path, song_file_mapping)
    if best_match:
        match_results = {
            "predicted_song": best_match,
            "match_scores": song_scores
        }
        return render_template("result.html", results=match_results)
    else:
        return render_template("result.html", results={"predicted_song": "No matches found", "match_scores": {}})

if __name__ == "__main__":
    app.run(debug=True)
