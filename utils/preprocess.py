import librosa
import numpy as np

def preprocess_audio(file_path, duration=3, sr=16000):
    audio, sr = librosa.load(file_path, sr=sr)
    max_len = sr * duration
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]

    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    mel_db = np.expand_dims(mel_db, axis=-1)
    mel_db = np.repeat(mel_db, 3, axis=-1)  # 3 channels
    mel_db = np.expand_dims(mel_db, axis=0)  # batch dimension
    return mel_db, mel_db[0]  # return one copy for plotting