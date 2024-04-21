
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import add
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import Input
from tensorflow.keras import optimizers
import pandas as pd
import numpy as np
from PIL import Image
from time import time
import re


def caption_preprocessing(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    # tokenize
    text=text.split()
    # convert to lower case
    text = [word.lower() for word in text]
    # remove hanging 's' and 'a'
    # text = [word for word in text if len(word)>1]

    # remove tokens with numbers in them
    text = [word for word in text if word.isalpha()]
    # store as string
    text =  ' '.join(text)

    # insert 'startseq', 'endseq' cho chuỗi
    text = 'startseq ' + text + ' endseq'
    return text


image_path = 'train/Images/'

# embedding_matrix.shape

df = pd.read_csv("train/captions.txt")
train, val = np.split(df.sample( frac=1,random_state=42), [int(.8*len(df)),])
# print(len(df), train.shape, val.shape)
df['caption'] = df['caption'].apply(caption_preprocessing)

word_counts = {}
max_length = 0
for text in df['caption']:
    words = text.split()
    max_length = len(words) if (max_length < len(words)) else max_length
    for w in words:
        try:
            word_counts[w] +=1
        except:
            word_counts[w] = 1
print(len(word_counts))
print(max_length)
# Chỉ lấy các từ xuất hiện trên 10 lần
word_count_threshold = 10
vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

i2w = {}
w2i = {}

id = 1
for w in vocab:
    w2i[w] = id
    i2w[id] = w
    id += 1

print(len(i2w), len(w2i))
print(i2w[300])

embedding_dim = 200
vocab_size = len(vocab) + 1 # thêm 1 padding


from pickle import dump, load
embedding_matrix = load(open("embedding_matrix.pkl", "rb"))


images = {}
captions = {}

start = time()
for i in range(len(df)):
    images[df['image'][i]] = np.array(Image.open(image_path + df['image'][i]))
    try:
        captions[df['image'][i]].append(df['caption'][i])
    except:
        captions[df['image'][i]] = [df['caption'][i]]

print(len(images), len(captions))
print('Time: ',time() - start)

captions = load(open("encoded_captions.pkl", "rb"))
# len(captions)

from pickle import load
train_features = load(open("encoded_train_images.pkl", "rb"))
# len(train_features)

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def data_generator(captions, images, w2i, max_length, batch_size):

  X_image, X_cap, y = [], [], []
  n = 0
  while 1:
    for id, caps in captions.items():
      n += 1
      image = images[id]
      for cap in caps:
        # encode the sequence
        seq = [w2i[word] for word in cap.split(' ') if word in w2i]

        for i in range(1, len(seq)):
          # split into input and output pair
          in_seq, out_seq = seq[:i], seq[i]

          # pad input sequence
          in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
          # encode output sequence
          out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

          # store
          X_image.append(image)
          X_cap.append(in_seq)
          y.append(out_seq)
      if n == batch_size:
        yield ([np.array(X_image), np.array(X_cap)], np.array(y))
        X_image, X_cap, y = [], [], []
        n = 0


tmp = np.array([[0,2,1],[3,5,6]])
tmp = np.expand_dims(tmp, axis=0)

# Tạo model
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)


# max_length = 35, vocab_size = 2005, embedding_dim = 200
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

from keras.models import load_model
model = load_model('model4.h5')

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.optimizer.lr = 0.0001
epochs = 30
batch_size = 16
steps = len(train_features)

checkpoint_path = "./checkpoint/cp.ckpt"
from tensorflow.keras.callbacks import ModelCheckpoint
cp_callback = ModelCheckpoint(filepath=checkpoint_path,save_best_only=False, save_weights_only=True, verbose=1)

generator = data_generator(captions=captions, images=train_features, w2i=w2i, max_length=max_length, batch_size=batch_size)
model.fit(generator, epochs=epochs, steps_per_epoch=steps, verbose=1, callbacks=[cp_callback])

model.save('model5.h5')