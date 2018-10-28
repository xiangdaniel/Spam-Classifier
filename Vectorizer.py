import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Build the class of vectorizer
class Vectorizer(object):
    def __init__(self):
        pass
        
    def fit_transform(self, data):
        vocab_set = set([])
        for line in data:
            words = line.split()
            for word in words:
                vocab_set.add(word.lower())
        self.vocablary = list(vocab_set)
        
        word_matrix = []
        for line in data:
            words = line.split()
            word_array = np.zeros(len(self.vocablary))
            for word in words:
                if word.lower() in self.vocablary:
                    word_array[self.vocablary.index(word.lower())] += 1
            word_matrix.append(word_array)
        return word_matrix
    
    def transform(self, data, vocablary):
        word_matrix = []
        for line in data:
            words = line.split()
            word_array = np.zeros(len(vocablary))
            for word in words:
                if word.lower() in vocablary:
                    word_array[vocablary.index(word.lower())] += 1
            word_matrix.append(word_array)
        return word_matrix
