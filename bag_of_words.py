import numpy as np
import lemmatization
import normalization


def create_bag_of_words(words: np.array) -> np.array:
    bag_of_words = [(0, words[0])]
    i = 1
    for word in words[1:]:
        if any(word in sublist for sublist in bag_of_words[:]):
            pass
        else:
            bag_of_words.append((i, word))
            i += 1
    return np.array(bag_of_words)

    
if __name__ == "__main__":
    print(create_bag_of_words(['dupa', 'twoja', 'dupa', 'twoja', 'stara']))    
