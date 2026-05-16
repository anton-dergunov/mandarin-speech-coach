import librosa
import soundfile as sf
import tempfile


def trim_silence(audio_path: str) -> str:
    y, sr = librosa.load(audio_path, sr=16000)

    yt, _ = librosa.effects.trim(y, top_db=20)

    trimmed_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    sf.write(trimmed_path, yt, sr)

    return trimmed_path
