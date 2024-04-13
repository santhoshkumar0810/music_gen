import librosa
from dtw import *


def compare_music_similarity(generated_audio_path, original_audio_path):
    generated_y, generated_sr = librosa.load(generated_audio_path)
    original_y, original_sr = librosa.load(original_audio_path)

    # Extract MFCC features
    generated_mfcc = librosa.feature.mfcc(y=generated_y, sr=generated_sr)
    original_mfcc = librosa.feature.mfcc(y=original_y, sr=original_sr)

    # Calculate cosine similarity between MFCC features (a basic measure)
    distance = dtw(generated_mfcc, original_mfcc)

    return distance


# Example usage (replace paths with your actual files)
generated_audio_path = "audio_output/123.wav"
original_audio_path = "audio_output/123.wav"

similarity = compare_music_similarity(generated_audio_path, original_audio_path)
print(similarity)

# Remember, this is a basic approach with limitations. Consider human evaluation for more reliable assessment.
