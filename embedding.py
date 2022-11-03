import os
import pickle

import fasttext
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

dir_path = os.path.dirname(os.path.realpath(__file__))

try:
    # load lemmatized nested lists
    with open(os.path.join(dir_path, 'lemmaEN.pickle'), 'rb') as f:
        lemmaEN = pickle.load(f)
    with open(os.path.join(dir_path, 'lemmaFR.pickle'), 'rb') as f:
        lemmaFR = pickle.load(f)
except:
    print('No pickle file(s) found')
    exit()


def generateVector(ft, sentence):
    # gets the average embedding of the words constituting the sentence
    return ft.get_sentence_vector(sentence)


def get_vectors(ft, lemmalist):
    vectors = []
    for act in lemmalist:
        sentence_list = []  # temp list for sentences in an act
        for sentence in act:
            # make a string out of words in a list with join()
            sentence_list.append(generateVector(ft, ' '.join(sentence)))
        vectors.append(sentence_list)
    return vectors


ft_en = fasttext.load_model(os.path.join(
    dir_path, os.path.join('data', 'cc.en.300.bin')))
en_vectors = get_vectors(ft_en, lemmaEN)
# delete model to clear memory
del (ft_en)

ft_fr = fasttext.load_model(os.path.join(
    dir_path, os.path.join('data', 'cc.fr.300.bin')))
fr_vectors = get_vectors(ft_fr, lemmaFR)
# delete model to clear memory
del (ft_fr)


def similarity(v1, v2):
    return np.dot(v1, v2)/(norm(v1)*norm(v2))


similarities = []
all_similarities = []
for act_index, act in enumerate(en_vectors):
    act_similarities = []
    for line_index, line in enumerate(act):
        # avoid IndexError, else continue
        if line_index < len(en_vectors[act_index]) and line_index < len(fr_vectors[act_index]):
            act_similarities.append(similarity(
                en_vectors[act_index][line_index], fr_vectors[act_index][line_index]))
            all_similarities.append(similarity(
                en_vectors[act_index][line_index], fr_vectors[act_index][line_index]))
    print(act_similarities)
    similarities.append(act_similarities)

try:
    with open('similarities.pickle', 'wb') as f:
        pickle.dump(all_similarities, f)
except:
    print("Error writing to similarities.pickle")
