from abc import ABC, abstractmethod
from tensorflow.keras.models import Sequential
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from typing import Union
import numpy as np

class BaseModel(ABC):
    
    def __init__(
            self, 
            model: Union[Sequential, BaseEstimator], 
            trained: bool = False) -> None:
        self.model = model
        self.trained = trained

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def predict(self, samples: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, samples: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: str, name: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(self, path: str, name: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def make(cls):
        pass

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
        
        predictions = self.predict(x_test)
        accuracy = accuracy_score(y_pred=predictions, y_true=y_test)
        print('Accuracy: %.3f\n' % accuracy)

        return accuracy