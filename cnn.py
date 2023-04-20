import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Data():
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(self.path)
        self.data = self.df.values
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.2, random_state=42)
        self.train_sentences = self.train_data[:, 0]
        self.train_labels = self.train_data[:, 1].astype(np.float64)
        self.test_sentences = self.test_data[:, 0]
        self.test_labels = self.test_data[:, 1].astype(np.float64)
        self.vocab_size = 10000
        self.embedding_dim = 16
        self.max_length = 120

    def get_data(self):
        return self.train_labels, self.test_labels
    
    def tokenize(self):
        tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        tokenizer.fit_on_texts(self.train_sentences)
        train_sequences = tokenizer.texts_to_sequences(self.train_sentences)
        train_padded = pad_sequences(train_sequences, maxlen=self.max_length, padding='post', truncating='post')
        test_sequences = tokenizer.texts_to_sequences(self.test_sentences)
        test_padded = pad_sequences(test_sequences, maxlen=self.max_length, padding='post', truncating='post')
        return train_padded,  test_padded
    
    
class ConvBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(ConvBlock, self).__init__()
        
        self.conv = tf.keras.layers.Conv2D(filters, (kernel_size,kernel_size), padding='same')
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.bn = tf.keras.layers.BatchNormalization()
        #self.relu = tf.keras.activations.relu()

    def call(self, inputs, training=False, mask=None):
        x = self.conv(inputs)
        x = self.bn(x)
        x = tf.nn.relu(x)
        x = self.pool(x)
        return x
    
class CNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        
        self.conv1 = ConvBlock(16, 3)
        self.conv2 = ConvBlock(32, 4)
        self.conv3 = ConvBlock(64, 5)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs, training=False, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
    
# Start program
if __name__ == '__main__':
    data = Data('SQLi.csv')
    train_labels, test_labels = data.get_data()
    train_padded, test_padded = data.tokenize()

    # model
    model = CNN(2)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train
    epochs = 20
    history = model.fit(train_padded, train_labels, epochs=epochs, validation_data=(test_padded, test_labels), verbose=2)

    # save model
    model.save('cnn.h5')

    # test model
    test_loss, test_acc = model.evaluate(test_padded, test_labels)
    print('Test Loss: {}'.format(test_loss))

    # predict
    test = test_padded[40]
    predictions = model.predict(np.array([test]))
    if predictions[0] > predictions[1]:
        print('SQLi')
    else:
        print('Normal')
