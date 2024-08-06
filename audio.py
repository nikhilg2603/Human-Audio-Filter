from pydub import AudioSegment
import numpy as np
from scipy.io import wavfile
import noisereduce as nr

# Load the audio file
audio = AudioSegment.from_file("/home/nikhil/Documents/white_lines robocup/trial.wav")

# Export to temporary WAV file for processing with scipy and noisereduce
audio.export("temp.wav", format="wav")

# Read the WAV file
rate, data = wavfile.read("temp.wav")

# Convert stereo to mono if necessary
if len(data.shape) > 1:
    data = np.mean(data, axis=1)

# Apply noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate)

# Convert the filtered samples back to an AudioSegment
filtered_audio = AudioSegment(
    reduced_noise.tobytes(), 
    frame_rate=rate,
    sample_width=reduced_noise.dtype.itemsize, 
    channels=1
)

# Export the filtered audio to a new file
filtered_audio.export("filtered_voice_audio.wav", format="wav")

print("Filtered audio saved as 'filtered_voice_audio.wav'")