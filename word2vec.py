import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load data
df = pd.read_csv('SQLi.csv')
data = df.values

# split data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# get sentences and labels
train_sentences = train_data[:, 0]
train_labels = train_data[:, 1].astype(np.float64)

test_sentences = test_data[:, 0]
test_labels = test_data[:, 1].astype(np.float64)

#hyperparameters
vocab_size = 10000
embedding_dim = 16
max_length = 120

# tokenize
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(train_sentences)
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# train word2vec model
model = Word2Vec(train_sentences, vector_size=150, window=5, min_count=1, workers=4)
model.train(train_sentences, total_examples=len(train_sentences), epochs=10)

# similarity
print(model.wv.most_similar('SELECT'))