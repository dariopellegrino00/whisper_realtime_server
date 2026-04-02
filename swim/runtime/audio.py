from functools import lru_cache

import librosa
import numpy as np


@lru_cache(10**6)
def load_audio(fname):
    audio, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return audio


def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg * 16000)
    end_s = int(end * 16000)
    return audio[beg_s:end_s]
