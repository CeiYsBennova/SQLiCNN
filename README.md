# SQLiCNN
Dựa trên bài báo: [tại đây](https://www.researchgate.net/publication/349022673_SQL_Injection_Attack_Detection_and_Prevention_Techniques_Using_Deep_Learning)

## Pretrained model
Sử dụng `word2vec` gồm 2 kỹ thuật chính:
- Skip-gram: input 1 -> output nhiều
- CBOW: input nhiều -> output 1

Trong bài, sử dụng thư viện `gensim` để thực hiện `word2vec`:
```
from gensim.models.word2vec import Word2Vec
```

Build vocabulary:
```
#model word2vec
word2vec_model = Word2Vec(vector_size=300, window=3, min_count=20,negative=20,sample=6e-5, workers=4)

# Build a vocabulary
word2vec_train = [sentence.split() for sentence in train_sentences]
word2vec_model.build_vocab(word2vec_train, progress_per=10000)
```

Load model pretrained vào `embedding matrix` ( do `keras` không còn hỗ trợ trực tiếp load model từ gensim):
```
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i >= vocab_size:
        continue
    try:
        embedding_vector = word2vec_model.wv[word]
        embedding_matrix[i] = embedding_vector
    except:
        embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), embedding_dim)
```       

## CNN
Mạng CNN được sử dụng là `Conv1D`, với `activation='relu'`, hàm phân loại sử dụng `sigmoid`:
```
# create embedding layer
embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False)

model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(16, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(32, 4, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
```