from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, Activation, BatchNormalization, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
from .dnn import DNN

class CNN1D(DNN):
    def __init__(self, model: Sequential, trained: bool = False) -> None:
        super(CNN1D, self).__init__(model, trained)

    @classmethod
    def make(
        cls,
        input_shape: int,
        n_kernels: int,
        kernel_sizes: list,
        hidden_size: int,
        dropout: float = 0.5,
        n_classes: int = 6,
        lr: float = 0.001
    ):
        """
        Build the model

        Args:
            input_shape (int): Feature dimension
            n_kernels (int): Number of convolutional kernels
            kernel_sizes (list): Sizes of kernels for each convolutional layer, list length equals number of convolutional layers
            hidden_size (int): Size of the fully connected layer
            dropout (float, optional, default=0.5): Dropout rate
            n_classes (int, optional, default=6): Number of label categories
            lr (float, optional, default=0.001): Learning rate
        """
        model = Sequential()

        for size in kernel_sizes:
            model.add(Conv1D(
                filters=n_kernels,
                kernel_size=size,
                padding='same',
                input_shape=(input_shape, 1)
            ))  # Convolutional layer
            model.add(BatchNormalization(axis=-1))
            model.add(Activation('relu'))
            model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(hidden_size))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

        model.add(Dense(n_classes, activation='softmax'))  # Classification layer
        optimizer = Adam(lr=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return cls(model)

    def reshape_input(self, data: np.ndarray) -> np.ndarray:
        """Convert 2D array to 3D array"""
        # (n_samples, n_feats) -> (n_samples, n_feats, 1)
        return np.reshape(data, (data.shape[0], data.shape[1], 1))
