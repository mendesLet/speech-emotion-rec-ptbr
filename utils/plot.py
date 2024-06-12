import wave
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.io.wavfile as wav
import numpy as np

def play_audio(file_path: str) -> None:
    """
    Play audio

    Args:
        file_path (str): Path to the audio file to be played
    """
    import pyaudio
    p = pyaudio.PyAudio()
    f = wave.open(file_path, "rb")
    stream = p.open(
        format=p.get_format_from_width(f.getsampwidth()),
        channels=f.getnchannels(),
        rate=f.getframerate(),
        output=True
    )
    data = f.readframes(f.getparams()[3])
    stream.write(data)
    stream.stop_stream()
    stream.close()
    f.close()

def curve(train: list, val: list, title: str, y_label: str) -> None:
    """
    Plot loss and accuracy curves

    Args:
        train (list): Array of training loss or accuracy values
        val (list): Array of validation loss or accuracy values
        title (str): Title of the plot
        y_label (str): Label for the y-axis
    """
    plt.plot(train)
    plt.plot(val)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

def radar(data_prob: np.ndarray, class_labels: list) -> None:
    """
    Plot radar chart of confidence probabilities

    Args:
        data_prob (np.ndarray): Array of probabilities
        class_labels (list): Emotion labels
    """
    angles = np.linspace(0, 2 * np.pi, len(class_labels), endpoint=False)

    # Close the plot
    data = np.concatenate((data_prob, [data_prob[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    class_labels = class_labels + [class_labels[0]]

    fig = plt.figure()

    # polar parameter
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, data, "bo-", linewidth=2)
    ax.fill(angles, data, facecolor="r", alpha=0.25)
    ax.set_thetagrids(angles * 180 / np.pi, class_labels)
    ax.set_title("Emotion Recognition", va="bottom")

    # Set the max value for radar chart
    ax.set_rlim(0, 1)

    ax.grid(True)
    # plt.ion()
    plt.show()
    # plt.pause(4)
    # plt.close()

def waveform(file_path: str) -> None:
    """
    Plot audio waveform

    Args:
        file_path (str): Path to the audio file
    """
    data, sampling_rate = librosa.load(file_path)
    plt.figure(figsize=(15, 5))
    librosa.display.waveshow(y=data, sr=sampling_rate)
    plt.show()

def spectrogram(file_path: str) -> None:
    """
    Plot spectrogram

    Args:
        file_path (str): Path to the audio file
    """

    # sr: sampling rate
    # x: numpy array of audio data
    sr, x = wav.read(file_path)

    # step: 10ms, window: 30ms
    nstep = int(sr * 0.01)
    nwin = int(sr * 0.03)
    nfft = nwin
    window = np.hamming(nwin)

    nn = range(nwin, len(x), nstep)
    X = np.zeros((len(nn), nfft // 2))

    for i, n in enumerate(nn):
        xseg = x[n - nwin:n]
        z = np.fft.fft(window * xseg, nfft)
        X[i, :] = np.log(np.abs(z[:nfft // 2]))

    plt.imshow(X.T, interpolation="nearest", origin="lower", aspect="auto")
    plt.show()
