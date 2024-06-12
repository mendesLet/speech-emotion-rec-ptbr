from tensorflow.keras.layers import LSTM as KERAS_LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
from .dnn import DNN

class LSTM(DNN):
    def __init__(self, model: Sequential, trained: bool = False) -> None:
        super(LSTM, self).__init__(model, trained)

    @classmethod
    def make(
        cls,
        input_shape: int,
        rnn_size: int,
        hidden_size: int,
        dropout: float = 0.5,
        n_classes: int = 6,
        lr: float = 0.001
    ):
        """
        Build the model

        Args:
            input_shape (int): Feature dimension
            rnn_size (int): LSTM hidden layer size
            hidden_size (int): Fully connected layer size
            dropout (float, optional, default=0.5): Dropout rate
            n_classes (int, optional, default=6): Number of label categories
            lr (float, optional, default=0.001): Learning rate
        """
        model = Sequential()

        model.add(KERAS_LSTM(rnn_size, input_shape=(1, input_shape)))  # (time_steps = 1, n_feats)
        model.add(Dropout(dropout))
        model.add(Dense(hidden_size, activation='relu'))

        model.add(Dense(n_classes, activation='softmax'))  # Classification layer
        optimizer = Adam(lr=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return cls(model)

    def reshape_input(self, data: np.ndarray) -> np.ndarray:
        """Convert 2D array to 3D array"""
        # (n_samples, n_feats) -> (n_samples, time_steps = 1, input_size = n_feats)
        return np.reshape(data, (data.shape[0], 1, data.shape[1]))
