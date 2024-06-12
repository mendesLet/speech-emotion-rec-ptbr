import os
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np
from tensorflow.keras.models import Sequential, model_from_json
from ..base import BaseModel
from utils import curve

class DNN(BaseModel, ABC):
    """
    Base class for all deep learning models based on Keras

    Args:
        n_classes (int): Number of label categories
        lr (float): Learning rate
    """
    def __init__(self, model: Sequential, trained: bool = False) -> None:
        super(DNN, self).__init__(model, trained)
        print(self.model.summary())

    def save(self, path: str, name: str) -> None:
        """
        Save the model

        Args:
            path (str): Model path
            name (str): Model file name
        """
        h5_save_path = os.path.join(path, name + ".h5")
        self.model.save_weights(h5_save_path)

        save_json_path = os.path.join(path, name + ".json")
        with open(save_json_path, "w") as json_file:
            json_file.write(self.model.to_json())

    @classmethod
    def load(cls, path: str, name: str):
        """
        Load the model

        Args:
            path (str): Model path
            name (str): Model file name
        """
        # Load json
        model_json_path = os.path.abspath(os.path.join(path, name + ".json"))
        json_file = open(model_json_path, "r")
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # Load weights
        model_path = os.path.abspath(os.path.join(path, name + ".h5"))
        model.load_weights(model_path)

        return cls(model, True)

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        batch_size: int = 32,
        n_epochs: int = 20
    ) -> None:
        """
        Train the model

        Args:
            x_train (np.ndarray): Training samples
            y_train (np.ndarray): Training labels
            x_val (np.ndarray, optional): Validation samples
            y_val (np.ndarray, optional): Validation labels
            batch_size (int): Batch size
            n_epochs (int): Number of epochs
        """
        if x_val is None or y_val is None:
            x_val, y_val = x_train, y_train

        x_train, x_val = self.reshape_input(x_train), self.reshape_input(x_val)

        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=n_epochs,
            shuffle=True,  # Shuffle training data before each epoch
            validation_data=(x_val, y_val)
        )

        # Loss and accuracy on the training set
        acc = history.history["accuracy"]
        loss = history.history["loss"]
        # Loss and accuracy on the validation set
        val_acc = history.history["val_accuracy"]
        val_loss = history.history["val_loss"]

        curve(acc, val_acc, "Accuracy", "acc")
        curve(loss, val_loss, "Loss", "loss")

        self.trained = True

    def predict(self, samples: np.ndarray) -> np.ndarray:
        """
        Predict the emotion of the audio

        Args:
            samples (np.ndarray): Audio features to be recognized

        Returns:
            results (np.ndarray): Recognition results
        """
        # Model has not been trained or loaded
        if not self.trained:
            raise RuntimeError("There is no trained model.")

        samples = self.reshape_input(samples)
        return np.argmax(self.model.predict(samples), axis=1)

    def predict_proba(self, samples: np.ndarray) -> np.ndarray:
        """
        Predict the probability of each emotion in the audio

        Args:
            samples (np.ndarray): Audio features to be recognized

        Returns:
            results (np.ndarray): Probability of each emotion
        """
        if not self.trained:
            raise RuntimeError('There is no trained model.')

        if hasattr(self, 'reshape_input'):
            samples = self.reshape_input(samples)
        return self.model.predict(samples)[0]

    @abstractmethod
    def reshape_input(self):
        pass
