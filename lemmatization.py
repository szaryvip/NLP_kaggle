import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import csv


def read_from_csv(path):
    tweets = []
    with open(path, 'r', -1, 'utf8') as file:
        data = csv.reader(file)
        for row in data:
            tweets.append(row[3])
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


def lemmatization(text):
    wnl = WordNetLemmatizer()
    tokens_tagged = nltk.pos_tag(nltk.word_tokenize(text))
    tokens_tagged = list(map(lambda x: (x[0], tagger(x[1])), tokens_tagged))
    lemmatized_text = []
    for word, tag in tokens_tagged:
        if tag is None:
            lemmatized_text.append((word, word))
        else:
            lemmatized_text.append((word, wnl.lemmatize(word, tag)))
    return lemmatized_text


if __name__ == "__main__":
    text = "the bats saw sawing the cats with best stripes hanging upside down by their feet"
    lemmatized_tokens = lemmatization(text)
    print(lemmatized_tokens)
    tweets = read_from_csv("data/train.csv")
    for tweet in tweets:
        print(lemmatization(tweet))
        
