import os
import csv
import sys
from typing import Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import utils

# Number of features for each feature set
FEATURE_NUM = {
    'IS09_emotion': 384,
    'IS10_paraling': 1582,
    'IS11_speaker_state': 4368,
    'IS12_speaker_trait': 6125,
    'IS13_ComParE': 6373,
    'ComParE_2016': 6373
}

def get_feature_opensmile(config, filepath: str) -> list:
    """
    Extract features from an audio file using Opensmile

    Args:
        config: Configuration
        filepath (str): Path to the audio file

    Returns:
        vector (list): Feature vector of the audio file
    """

    # Project path
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    # Path to single_feature.csv
    single_feat_path = os.path.join(BASE_DIR, config.feature_folder, 'single_feature.csv')
    # Path to Opensmile configuration file
    opensmile_config_path = os.path.join(config.opensmile_path, 'config', config.opensmile_config + '.conf')

    # Opensmile command
    cmd = 'cd ' + config.opensmile_path + ' && ./SMILExtract -C ' + opensmile_config_path + ' -I ' + filepath + ' -O ' + single_feat_path + ' -appendarff 0'
    print("Opensmile cmd: ", cmd)
    os.system(cmd)

    reader = csv.reader(open(single_feat_path, 'r'))
    rows = [row for row in reader]
    last_line = rows[-1]
    return last_line[1: FEATURE_NUM[config.opensmile_config] + 1]

def load_feature(config, train: bool) -> Union[Tuple[np.ndarray], np.ndarray]:
    """
    Load feature data from "{config.feature_folder}/*.csv" files

    Args:
        config: Configuration
        train (bool): Whether it is training data

    Returns:
        - X (Tuple[np.ndarray]): Training features, testing features, and corresponding labels
        - X (np.ndarray): Prediction features
    """

    feature_path = os.path.join(config.feature_folder, "train.csv" if train else "predict.csv")

    # Load feature data
    df = pd.read_csv(feature_path)
    features = [str(i) for i in range(1, FEATURE_NUM[config.opensmile_config] + 1)]

    X = df.loc[:, features].values
    Y = df.loc[:, 'label'].values

    # Path to standardization model
    scaler_path = os.path.join(config.checkpoint_path, 'SCALER_OPENSMILE.m')

    if train:
        # Standardize data
        scaler = StandardScaler().fit(X)
        # Save standardization model
        utils.mkdirs(config.checkpoint_path)
        joblib.dump(scaler, scaler_path)
        X = scaler.transform(X)
        # Split training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test
    else:
        # Standardize data
        # Load standardization model
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
        return X

def get_data(config, data_path: str, train: bool) -> Union[Tuple[np.ndarray], np.ndarray]:
    """
    Extract features from all audio files using Opensmile: Traverse all folders, read audio files in each folder, 
    extract features of each audio file, and save all features in "{config.feature_path}/*.csv" files.

    Args:
        config: Configuration
        data_path (str): Dataset folder / Test file path
        train (bool): Whether it is training data

    Returns:
        - train = True: Training features, testing features, and corresponding labels
        - train = False: Prediction features
    """
    # Create config.feature_folder if it doesn't exist
    utils.mkdirs(config.feature_folder)

    # Path to store features
    feature_path = os.path.join(config.feature_folder, "train.csv" if train else "predict.csv")

    # Write header
    writer = csv.writer(open(feature_path, 'w'))
    first_row = ['label']
    for i in range(1, FEATURE_NUM[config.opensmile_config] + 1):
        first_row.append(str(i))
    writer.writerow(first_row)

    writer = csv.writer(open(feature_path, 'a+'))
    print('Opensmile extracting...')

    if train:
        cur_dir = os.getcwd()
        sys.stderr.write('Curdir: %s\n' % cur_dir)
        os.chdir(data_path)

        # Traverse folders
        for i, directory in enumerate(config.class_labels):
            sys.stderr.write("Started reading folder %s\n" % directory)
            os.chdir(directory)

            # label_name = directory
            label = config.class_labels.index(directory)

            # Read audio files in the folder
            for filename in os.listdir('.'):
                if not filename.endswith('wav'):
                    continue
                filepath = os.path.join(os.getcwd(), filename)

                # Extract features of the audio file
                feature_vector = get_feature_opensmile(config, filepath)
                feature_vector.insert(0, label)
                # Organize features of each audio file into a csv file
                writer.writerow(feature_vector)

            sys.stderr.write("Ended reading folder %s\n" % directory)
            os.chdir('..')
        os.chdir(cur_dir)

    else:
        feature_vector = get_feature_opensmile(config, data_path)
        feature_vector.insert(0, '-1')
        writer.writerow(feature_vector)

    print('Opensmile extract done.')

    # Temporary workaround for a mysterious bug
    # Unable to directly load prediction data features of feature sets other than IS10_paraling, very strange
    if train:
        return load_feature(config, train=train)
