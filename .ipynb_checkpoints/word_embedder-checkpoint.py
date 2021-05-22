from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim


class word_embedder():
    def word_embedder(df,filename):
        sentences = list()

        lines = df['original_text'].values.tolist()

        for line in tqdm(lines):
            tokens = word_tokenize(line)
            tokens = [word.lower() for word in tokens]
            table = str.maketrans('','',string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            words = [word for word in stripped if word.isalpha()]
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if not w in stop_words]
            sentences.append(words)
        print('Generating word embeddings') 
        EMBEDDING_DIM = 100

        model = gensim.models.Word2Vec(
            sentences = sentences,
            size = EMBEDDING_DIM,
            window = 5,
            min_count = 1)

        words = list(model.wv.vocab)
            
        model.save("{}.model".format(filename))
        model.save("{}.bin".format(filename))
        
        return print('New word embeddings file {} has been generated'.format(filename))