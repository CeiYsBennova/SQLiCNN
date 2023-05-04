import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer

# read data
df = pd.read_csv('SQLi.csv')

# split data
data = df.values
train_data, test_data = train_test_split(data, test_size=0.4, random_state=42)

# get sentences and labels
train_sentences = train_data[:, 0]
train_labels = train_data[:, 1].astype(np.float64)

test_sentences = test_data[:, 0]
test_labels = test_data[:, 1].astype(np.float64)

# vectorize
vectorizer = CountVectorizer()
vectorizer.fit(train_sentences)

train_vectors = vectorizer.transform(train_sentences)
test_vectors = vectorizer.transform(test_sentences)

# train model
model = GaussianNB()

model.fit(train_vectors.toarray(), train_labels)

# predict
predictions = model.predict(test_vectors.toarray())

# evaluate
print('accuracy: ', accuracy_score(test_labels, predictions))
print('precision: ', precision_score(test_labels, predictions))
print('recall: ', recall_score(test_labels, predictions))
print('f1: ', f1_score(test_labels, predictions))

# save model
import pickle

with open('naivebayes.pkl', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

# save vectorizer
with open('vectorizerNB.pkl', 'wb') as handle:
    pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    