import os
import re
import sys
import librosa
from random import shuffle
import numpy as np
from typing import Tuple, Union
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import utils

def features(X, sample_rate: float) -> np.ndarray:

    stft = np.abs(librosa.stft(X))

    pitches, magnitudes = librosa.piptrack(X, sample_rate, S=stft, fmin=70, fmax=400)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch.append(pitches[index, i])

    pitch_tuning_offset = librosa.pitch_tuning(pitches)
    pitchmean = np.mean(pitch)
    pitchstd = np.std(pitch)
    pitchmax = np.max(pitch)
    pitchmin = np.min(pitch)

    cent = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
    cent = cent / np.sum(cent)
    meancent = np.mean(cent)
    stdcent = np.std(cent)
    maxcent = np.max(cent)

    flatness = np.mean(librosa.feature.spectral_flatness(y=X))

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccsstd = np.std(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccmax = np.max(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    zerocr = np.mean(librosa.feature.zero_crossing_rate(X))

    S, phase = librosa.magphase(librosa.stft(X))
    meanMagnitude = np.mean(S)
    stdMagnitude = np.std(S)
    maxMagnitude = np.max(S)

    rmse = librosa.feature.rms(S=S)[0]
    meanrms = np.mean(rmse)
    stdrms = np.std(rmse)
    maxrms = np.max(rmse)

    ext_features = np.array([
        flatness, zerocr, meanMagnitude, maxMagnitude, meancent, stdcent,
        maxcent, stdMagnitude, pitchmean, pitchmax, pitchstd,
        pitch_tuning_offset, meanrms, maxrms, stdrms
    ])

    ext_features = np.concatenate((ext_features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast))

    return ext_features

def extract_features(file: str, pad: bool = False) -> np.ndarray:

    X, sample_rate = librosa.load(file, sr=None)
    max_ = X.shape[0] / sample_rate
    if pad:
        length = (max_ * sample_rate) - X.shape[0]
        X = np.pad(X, (0, int(length)), 'constant')

    return features(X, sample_rate)

def get_max_min(files: list) -> Tuple[float]:

    min_, max_ = 100, 0

    for file in files:
        sound_file, samplerate = librosa.load(file, sr=None)
        t = sound_file.shape[0] / samplerate
        if t < min_:
            min_ = t
        if t > max_:
            max_ = t

    return max_, min_

def get_data_path(data_path: str, class_labels: list) -> list:

    wav_file_path = []

    cur_dir = os.getcwd()
    sys.stderr.write('Curdir: %s\n' % cur_dir)
    os.chdir(data_path)

    for _, directory in enumerate(class_labels):
        os.chdir(directory)

        for filename in os.listdir('.'):
            if not filename.endswith('wav'):
                continue
            filepath = os.path.join(os.getcwd(), filename)
            wav_file_path.append(filepath)

        os.chdir('..')
    os.chdir(cur_dir)

    shuffle(wav_file_path)
    return wav_file_path

def load_feature(config, train: bool) -> Union[Tuple[np.ndarray], np.ndarray]:

    feature_path = os.path.join(config.feature_folder, "train.p" if train == True else "predict.p")

    features = pd.DataFrame(
        data = joblib.load(feature_path),
        columns = ['file_name', 'features', 'emotion']
    )

    X = list(features['features'])
    Y = list(features['emotion'])

    scaler_path = os.path.join(config.checkpoint_path, 'SCALER_LIBROSA.m')

    if train == True:
        scaler = StandardScaler().fit(X)
        
        utils.files.mkdirs(config.checkpoint_path)
        joblib.dump(scaler, scaler_path)
        X = scaler.transform(X)

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test

    else:
        
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
        return X

def get_data(config, data_path: str, train: bool) -> Union[Tuple[np.ndarray], np.ndarray]:
    
    if train == True:
        files = get_data_path(data_path, config.class_labels)
        max_, min_ = get_max_min(files)

        mfcc_data = []
        for file in files:
            label = re.findall(".*-(.*)-.*", file)[0]
            features = extract_features(file, max_)
            mfcc_data.append([file, features, config.class_labels.index(label)])

    else:
        features = extract_features(data_path)
        mfcc_data = [[data_path, features, -1]]

    utils.files.mkdirs(config.feature_folder)
    feature_path = os.path.join(config.feature_folder, "train.p" if train == True else "predict.p")
    pickle.dump(mfcc_data, open(feature_path, 'wb'))

    return load_feature(config, train=train)