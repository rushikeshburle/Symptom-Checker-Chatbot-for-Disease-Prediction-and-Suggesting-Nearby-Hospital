import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag



# 1. tokenize(sentence)
# Purpose: Tokenizes the input sentence into individual words.
# How it works:
# Uses the word_tokenize function from the nltk library to break the sentence into a list of words (tokens).
# This function handles punctuation, splitting words correctly even when they include punctuation marks.

# 2. stem(word)
# Purpose: Stems a word to its root form using the Porter Stemmer.
# How it works:
# The function takes a word, converts it to lowercase, and then applies the PorterStemmer from NLTK to reduce the word to its root form (e.g., "running" becomes "run").

# 3. bag_of_words(tokenized_sentence, words)
# Purpose: Creates a bag-of-words (BoW) vector for a given tokenized sentence, indicating the presence or absence of words in a predefined list.
# How it works:
# The function takes a list of tokenized words from a sentence and compares it with a predefined list of words (words).
# It first stems each word in the tokenized sentence using the stem() function to match the words in their root forms.
# Then, it initializes a vector bag with zeros, where each index corresponds to a word in the words list.
# If a word from the words list is present in the tokenized sentence, its corresponding index in the bag vector is set to 1.
  
# This code is used to process text data by converting sentences into a numerical format (bag-of-words) that machine learning algorithms can use. The stemming process 
# reduces the complexity of the words by mapping similar words (like "running" and "run") to the same root form, ensuring the algorithm treats them as the same word.

   





