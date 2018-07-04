# bridging Python versions
from __future__ import absolute_import, division, print_function

# for openning and processing the data files
import codecs # encoding the corpus into utf-8
import glob  # Unix style pathname pattern
import logging
import multiprocessing
from joblib import Parallel, delayed  # for parallel computation
import os
import re
import time

import nltk  # for text cleaning
from gensim.models.phrases import Phraser, Phrases
from gensim.models import Word2Vec, FastText  # the models used for word embeddings

# for plotting and analyzing the results

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.manifold import TSNE # for dimensionality reduction to plot similar words

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

nltk.download("punkt")

book_names = sorted(glob.glob("data/*.txt"))

raw_corpus = u""  # will use utf-8
for filename in book_names:
    print("Reading {}...".format(filename.split("/")[1]))
    with codecs.open(filename,"r","utf-8") as book:
        raw_corpus += book.read()
    print("Corpus now is {} characters and {} words long".format(len(raw_corpus), len(raw_corpus.split())))
    print("~"*30)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

raw_sentences = tokenizer.tokenize(raw_corpus.lower())

def sentence_to_wordlist(raw:str):
    return re.sub("[^a-zA-Z]"," ", raw).split()

# converting sentences to wordlists, utilizing all the cpu cores
tokenized_sentences = Parallel(n_jobs=-1)(
                delayed(sentence_to_wordlist)(
                    raw_sentence) for raw_sentence in raw_sentences)

phrases = Phrases(tokenized_sentences)

bigram = Phraser(phrases)

sentences = list(bigram[tokenized_sentences])

tokens_count = sum([len(sen) for sen in sentences])
no_bigram_count = sum([len(s) for s in tokenized_sentences])
print("""This corpus has:\n
                {0:,} tokens with bigrams\n
                {1:,} tokens without bigrams\n
                {2:} of the tokens have been included in phrases""".format(
    tokens_count,no_bigram_count, 1-tokens_count/no_bigram_count))


workers = multiprocessing.cpu_count()
min_word_count = 3
features_count = 300
window_size = 5
subsampling = 0.001
seed = 48 # the answer to the universe is ... 48 :D


if not os.path.exists("trained"):
    os.makedirs("trained")


start = time.time()
w2v_cbow = Word2Vec(sentences=sentences, size=300, window=7, min_count=3, workers=workers, sg=0, seed=seed)
print("Training Word2Vec CBOW took {} seconds".format(time.time()-start))

w2v_cbow.save(os.path.join("trained", "w2v_cbow.bin"))

start = time.time()
w2v_sg = Word2Vec(sentences=sentences, size=300, window=7, min_count=3, workers=workers, sg=1, seed=seed)
print("Training Word2Vec Skip-Gram took {} seconds".format(time.time()-start))

w2v_sg.save(os.path.join("trained", "w2v_sg.bin"))

start = time.time()
fasttext_cbow = FastText(sentences, size=300, window=7, min_count=3, workers=workers,sg=0, seed=seed)
print("Training FastText CBOW took {} seconds".format(time.time()-start))

fasttext_cbow.save(os.path.join("trained", "fasttext_cbow.bin"))

start = time.time()
fasttext_sg = FastText(sentences, size=300, window=7, min_count=3, workers=workers,sg=1, seed=seed)
print("Training FastText CBOW took {} seconds".format(time.time()-start))

fasttext_sg.save(os.path.join("trained", "fasttext_sg.bin"))