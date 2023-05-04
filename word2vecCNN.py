import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
#import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phrases, Phraser

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Embedding, Conv1D, MaxPooling1D

# read data
df = pd.read_csv('SQLi.csv')
#df = pd.read_csv('../WafaMole_dataset/wafamole_dataset.csv')

# split data
data = df.values
train_data, test_data = train_test_split(data, test_size=0.4, random_state=42)

# get sentences and labels
train_sentences = train_data[:, 0]
train_labels = train_data[:, 1].astype(np.float64)

test_sentences = test_data[:, 0]
test_labels = test_data[:, 1].astype(np.float64)

#model word2vec
word2vec_model = Word2Vec(vector_size=300, window=3, min_count=20,negative=20,sample=6e-5, workers=4)

# Build a vocabulary
word2vec_train = [sentence.split() for sentence in train_sentences]
phraser = Phrases(word2vec_train, min_count=20, progress_per=10000)
biagram = Phraser(phraser)
word2vec_train = biagram[word2vec_train]
word2vec_model.build_vocab(word2vec_train, progress_per=10000)

# train word2vec model
word2vec_model.train(train_sentences, total_examples=word2vec_model.corpus_count, epochs=10, report_delay=1)



#save model
#word2vec_model.save('word2vec.model')
#word2vec_model.save('word2vec1.model')

# load pretrained model
#word2vec_model = Word2Vec.load('word2vec.model')
#word2vec_model = Word2Vec.load('word2vec1.model')

#  prepare embedding matrix
vocab_size = 10000
embedding_dim = 16
max_length = 120

# tokenize
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_sentences)
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

#add dimension in train_padded
train_padded = np.expand_dims(train_padded, axis=2)
test_padded = np.expand_dims(test_padded, axis=2)
# create embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i >= vocab_size:
        continue
    try:
        embedding_vector = word2vec_model.wv[word]
        embedding_matrix[i] = embedding_vector
    except:
        embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), embedding_dim)

# create embedding layer
embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False)

# create model
model = Sequential()
model.add(embedding_layer)
'''model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(32, (4, 4), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))'''
model.add(Conv1D(16, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(32, 4, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
model.fit(train_padded, train_labels, epochs=10, validation_data=(test_padded, test_labels))

# evaluate model
test_loss, test_acc = model.evaluate(test_padded, test_labels)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# save model
#model.save('word2vecCNN.h5')
#model.save('word2vecCNN1.h5')

# save tokenizer
import pickle
#with open('tokenizer.pkl', 'wb') as f:
#with open('tokenizer1.pkl', 'wb') as f:
 #   pickle.dump(tokenizer, f, pickle.HIGHEST_PROTOCOL)

#predict the model
def predict(sentence):
    sequences = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    if model.predict(padded)[0][0] > 0.5:
        print('SQLi')
    else:
        print('Normal')

# predict the model
predict('SELECT Employees.LastName, COUNT ( Orders.OrderID )  AS NumberOfOrders FROM  ( Orders INNER JOIN Employees ON Orders.EmployeeID  =  Employees.EmployeeID )  GROUP BY LastName HAVING COUNT ( Orders.OrderID )  > 10;')
predict('SELECT * FROM users WHERE username = "admin" AND password = "password" OR 1=1')
predict('Hello World!')
predict('union select version(),user(),3,4,--+-')
predict('from users where id  =  1<@<@ union select 1,version()-- 1')
predict('SELECT min (failed) FROM nation SELECT SUM(economy)')
predict('UnIOn sElecT 1,2,3,id(),--+-')

# check weights and accuracy
print(model.evaluate(test_padded, test_labels))

# apply metrics on test data
y_pred = model.predict(test_padded)

# print accuracy
print("Accuracy score: ",accuracy_score(test_labels, y_pred.round()))

# print precision
print("Precision score: ",precision_score(test_labels, y_pred.round()))

# print recall
print("Recall score: ",recall_score(test_labels, y_pred.round()))

# print f1-score
print("F1-score: ",f1_score(test_labels, y_pred.round()))

