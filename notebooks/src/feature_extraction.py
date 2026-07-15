import numpy as np
import librosa


def load_audio_fixed_length(path, sample_rate, target_length):
    y, sr = librosa.load(path, sr=sample_rate, mono=True)

    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    return y


def summarize_feature(feature_matrix, prefix):
    values = []
    values.extend(np.mean(feature_matrix, axis=1))
    values.extend(np.std(feature_matrix, axis=1))
    values.extend(np.min(feature_matrix, axis=1))
    values.extend(np.max(feature_matrix, axis=1))

    names = []
    n = feature_matrix.shape[0]
    for stat in ["mean", "std", "min", "max"]:
        for i in range(n):
            names.append(f"{prefix}_{i+1}_{stat}")

    return np.array(values), names


def extract_features(path, config):
    y = load_audio_fixed_length(
        path, sample_rate=config["sample_rate"], target_length=config["target_length"]
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=config["sample_rate"],
        n_mfcc=config["n_mfcc"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
    )

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=config["sample_rate"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
    )
    logmel = librosa.power_to_db(mel, ref=np.max)

    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=config["n_fft"], hop_length=config["hop_length"]
    )

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=config["sample_rate"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=config["sample_rate"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=config["sample_rate"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
    )

    features = []
    names = []

    feature_groups = [
        (mfcc, "mfcc"),
        (zcr, "zcr"),
        (centroid, "spectral_centroid"),
        (bandwidth, "spectral_bandwidth"),
        (rolloff, "spectral_rolloff"),
    ]

    for matrix, prefix in feature_groups:
        values, feature_names = summarize_feature(matrix, prefix)
        features.extend(values)
        names.extend(feature_names)

    return np.array(features), names, logmel
