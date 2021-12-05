import bag_of_words
from normalization import normalization
import typing
import numpy as np
import nltk
from nltk.stem import PorterStemmer
import csv
import sys
sys.path.insert(0, '..')


def stem(word: str) -> str:
    porter = nltk.PorterStemmer()
    return porter.stem(word)


def stem_array(array: np.ndarray) -> np.ndarray:
    words = np.array([])
    for text in array:
        words = np.append(words, text.split())
    for i in range(words.size):
        words[i] = stem(words[i])
    return words


def stem_from_csv(csv_path: str, column: int) -> np.ndarray:
    tweets = np.array([])

    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        for line in csv_reader:
            tweets = np.append(tweets, line[column])
    tweets = np.delete(tweets, 0)

    words = stem_array(tweets)
    return words


def save_to_csv(csv_path: str, words: np.ndarray) -> None:
    with open(csv_path, 'w') as new_file:
        csv_writer = csv.writer(new_file)

        for i in words:
            csv_writer.writerow([i])


def stem_normalize_from_to_csv(from_path: str, to_path: str) -> None:
    words = stem_from_csv(from_path, 3)
    bag_of_words.create_bag_of_words(words)
    indexes = np.arange(words.size)
    stacked_words = np.stack((indexes, words), axis=1)
    normalization.normalize(stacked_words, to_path)
