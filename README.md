### Overview

Speech-Emotion-Rec-PTBR is a Speech Emotion Recognition (SER) project focused on Brazilian Portuguese. This repository provides implementations for various machine learning models to classify emotions from speech data. The primary dataset used for training and testing is the CORAA SER PT dataset.

### Features

   - Dataset: CORAA SER PT
   - Models Implemented:
        - Multilayer Perceptron (MLP) - Achieved 80% accuracy
        - Support Vector Machine (SVM) - (Implementation in progress)
        - Convolutional Neural Network (CNN) - (Implementation in progress)
        - Long Short-Term Memory (LSTM) - (Implementation in progress)
    
### Getting Started

#### Prerequisites
  - Python 3.8 or higher
  - Install OpenSMILE 2.3.0:
    - Download and install OpenSMILE 2.3.0 from [this link](https://biicgitlab.ee.nthu.edu.tw/adam1214/ser_offline_pytest_v1/-/tree/c2650e6b520eba31e20a6f2cc9f5c35635d078d6/opensmile-2.3.0). 

#### Structure

```txt
├── models/                // Model implementations
│   ├── base.py            // Base class for all models
│   ├── dnn/               // Neural network models <TO BE ADDED> 
│   │   ├── dnn.py         // Base class for all neural network models <TO BE ADDED>
│   │   ├── cnn.py         // CNN <TO BE ADDED>
│   │   └── lstm.py        // LSTM <TO BE ADDED>
│   └── ml.py              // SVM & MLP
├── extract_feats/         // Feature extraction
│   ├── librosa.py         // Feature extraction with librosa
│   └── opensmile.py       // Feature extraction with OpenSMILE
├── utils/
│   ├── files.py           // For organizing datasets (classification, batch renaming)
│   ├── opts.py            // Read parameters from the command line using argparse
│   └── plot.py            // Plotting (radar charts, spectrograms, waveforms)
├── config/                // Configuration parameters (.yaml)
├── features/              // Store extracted features
├── checkpoints/           // Store trained model weights
├── datasets/              // Store datasets
├── train.py               // Train models
├── predict.py             // Predict emotions of specified audio using trained models
└── preprocess.py          // Data preprocessing (extract features from dataset audio and save)

```

#### Configuration

Configure parameters in the configuration files (YAML) located in the configs/ folder.

The currently supported OpenSMILE standard feature sets are:

  - IS09_emotion: The INTERSPEECH 2009 Emotion Challenge, 384 features.
  - IS10_paraling: The INTERSPEECH 2010 Paralinguistic Challenge, 1582 features.
  - IS11_speaker_state: The INTERSPEECH 2011 Speaker State Challenge, 4368 features.
  - IS12_speaker_trait: The INTERSPEECH 2012 Speaker Trait Challenge, 6125 features.
  - IS13_ComParE: The INTERSPEECH 2013 ComParE Challenge, 6373 features.
  - ComParE_2016: The INTERSPEECH 2016 Computational Paralinguistics Challenge, 6373 features.

If you need to use other feature sets, you can modify the FEATURE_NUM item in extract_feats/opensmile.py.
#### Preprocess

First, you need to extract features from the audio in the dataset and save them locally. Features extracted by OpenSMILE will be saved in .csv files, while features extracted by librosa will be saved in .p files.
```sh
python preprocess.py --config configs/example.yaml
```

#### Train

The dataset path can be configured in the configs/ folder. Place audio files with the same emotion in the same folder (you can refer to utils/files.py to organize the data), for example:

```txt
└── datasets
    ├── neutral
    ├── happy
    ├── sad
    ...
```

Then run:

```sh
python train.py --config configs/example.yaml
```

#### Predict

Use the trained model to predict the emotion of a specified audio file. There are some pre-trained models in the checkpoints/ directory.

```sh
python predict.py --config configs/example.yaml
```

### Acknowledgments

  - Based on [Renovamen/Speech-Emotion-Recognition](https://github.com/Renovamen/Speech-Emotion-Recognition/blob/master/utils/plot.py)
  - [CORAA SER PT Dataset](https://www.icmc.usp.br/eventos/5802-coraa-ser-v1-um-dataset-para-tarefas-de-reconhecimento-de-emocoes-a-partir-de-fala-espontanea-para-o-portugues-brasileiro)
