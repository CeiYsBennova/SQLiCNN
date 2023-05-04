import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
# read csv
df = pd.read_csv('SQLi.csv')

# split data
data = df.values
train_data, test_data = train_test_split(data, test_size=0.4, random_state=42)

# get sentences and labels
train_sentences = train_data[:, 0]
train_labels = train_data[:, 1].astype(np.float64)

test_sentences = test_data[:, 0]
test_labels = test_data[:, 1].astype(np.float64)

# load tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# tokenize
train_sequences = tokenizer.texts_to_sequences(train_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)

# pad sequences
max_length = 120
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# expand dimension
train_padded = np.expand_dims(train_padded, axis=2)
test_padded = np.expand_dims(test_padded, axis=2)

# test_padded shape
print(test_padded.shape)
