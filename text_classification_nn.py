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


"""
#print(decode_review(test_data[0]))

model = keras.Sequential()
model.add = keras.layers.Embedding(88000,16)
model.add = keras.layers.GlobalAveragePooling1D()
model.add = keras.layers.Dense(16,activation="relu")
model.add = keras.layers.Dense(1,activation="sigmoid")



model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]


fit_model = model.fit(x_train,y_train,epochs=60, batch_size=512,validation_data=(x_val,y_val),verbose=1)


results = model.evaluate(test_data,test_labels)
print(results)

model.save("tc_mode.h5")



#I will split the code from know on to another python file

model = keras.models.load_model("tc_mode.h5")




i = int(input("Please insert a number to choose a review from our database: "))

review_types = ['Bad','Good']
predict = model.predict(test_data)
print("Review:")
print(decode_review(test_data[i]))

x = int(predict[i])

print("Prediction: " + review_types[x])
print("Actual: " + review_types[test_labels[i]])

if x == test_labels[i]:
    print("The prediction is correct!")
else:
    print("Sorry, not this time...")

