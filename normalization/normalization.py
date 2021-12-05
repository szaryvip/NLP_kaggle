import numpy as np
import csv
import typing
import os


def normalize(words: np.ndarray) -> np.ndarray:
    indexes = np.arange(words.shape[0])
    indexes = (indexes - np.min(indexes)) / (np.max(indexes) - np.min(indexes))
    words[:, 0] = indexes
    return words
