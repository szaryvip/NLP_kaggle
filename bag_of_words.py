import numpy as np
from lemmatization import lemmatization
import csv
import normalization


def create_bag_of_words(words: np.array) -> np.ndarray:
    bag_of_words = [(0, words[0])]
    i = 1
    for word in words[1:]:
        if any(word in sublist for sublist in bag_of_words[:]):
            pass
        else:
            bag_of_words.append((i, word))
            i += 1
    return np.array(bag_of_words)


def sentence_as_number(text: str, all_words: np.ndarray) -> np.ndarray:
    words_in = lemmatization.lemmatization(text)
    containing = []
    for word in all_words:
        if word in words_in:
            containing.append(1)
        else:
            containing.append(0)
    return np.array(containing)


def all_as_numbers(path: str, words: np.ndarray, number:int) -> np.ndarray:
    containing = []
    tweets = lemmatization.read_from_csv(path)
    i = 0
    maxi = len(tweets)
    for text in tweets:
        i+=1
        if i == number:
            break
        print(i, "/", maxi)
        sentence = sentence_as_number(text, words)
        containing.append(sentence)
    return np.array(containing)


def is_bad(path: str) -> np.ndarray:
    opinion = np.array([])
    with open(path, 'r', -1, 'utf8') as file:
        data = csv.reader(file)
        for row in data:
            opinion = np.append(opinion, row[4])
        opinion = opinion[1:]
    return opinion


def divide_data(train: np.ndarray, opinions: np.ndarray,
                percent: float):
    border = int(len(train) * percent)
    return train[:border], train[border:], opinions[:border], opinions[border:]

    
if __name__ == "__main__":
    words = lemmatization.lemmatize_all_csv("data/train.csv")
    # print(create_bag_of_words(words))
    print(all_as_numbers('data/train.csv', words, 3500))
    print(is_bad('data/train.csv'))
