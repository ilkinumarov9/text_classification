import tensorflow as tf
import keras
import numpy as np
data = keras.datasets.imdb

(train_data,train_labels), (test_data,test_labels) = data.load_data(num_words=80000)


#print(train_data[0])

word_ind = data.get_word_index()
word_ind = {k: (v+3) for k, v in word_ind.items()}
word_ind["<PAD>"] = 0
word_ind["<START>"] = 1
word_ind["<UNK>"] = 2
word_ind["<UNUSED>"] = 3

reverse_word_index = dict([(value,key) for (key,value) in word_ind.items()])

train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,value=word_ind["<PAD>"], padding="post", maxlen=250
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,value=word_ind["<PAD>"], padding="post", maxlen=250
)


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_ind:
            encoded.append(word_ind[word.lower()])
        else:
            encoded.append(2)
    return encoded
model = keras.models.load_model("tc_mode.h5")

with open('irishman.txt',encoding="utf-8") as ir:
    for line in ir.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode],value=word_ind["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        #print(line)
        #print(encode)
        print(predict[0])