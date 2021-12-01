import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import csv
import numpy as np
import string


def read_from_csv(path: string) -> np.array:
    tweets = np.array([])
    with open(path, 'r', -1, 'utf8') as file:
        data = csv.reader(file)
        for row in data:
            tweets = np.append(tweets, row[3])
        tweets = tweets[1:]
    return tweets


def tagger(nltk_tag):
    if nltk_tag[0] == 'J':
        return wordnet.ADJ
    elif nltk_tag[0] == 'V':
        return wordnet.VERB
    elif nltk_tag[0] == 'N':
        return wordnet.NOUN
    elif nltk_tag[0] == 'R':
        return wordnet.ADV
    else:
        return None


def lemmatization(text: string) -> np.array:
    wnl = WordNetLemmatizer()
    tokens_tagged = nltk.pos_tag(nltk.word_tokenize(text))
    tokens_tagged = list(map(lambda x: (x[0], tagger(x[1])), tokens_tagged))
    lemmatized_text = np.array([])
    for word, tag in tokens_tagged:
        if tag is None:
            lemmatized_text = np.append(lemmatized_text, word)
        else:
            lemmatized_text = np.append(lemmatized_text, wnl.lemmatize(word, tag))
    return lemmatized_text


def lemmatize_all_csv(path: string) -> np.array:
    tweets = read_from_csv(path)
    words = np.array([])
    for tweet in tweets:
        lemmatized = lemmatization(tweet)
        words = np.concatenate((words, lemmatized))
    return words


if __name__ == "__main__":
    text = "the bats saw sawing the cats with best stripes hanging upside down by their feet"
    print(lemmatization(text))
    print(lemmatize_all_csv("data/train.csv"))
        
