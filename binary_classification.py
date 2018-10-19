# BASED ON: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import pandas as pd
import math
import pprint as pp

#load and split data
data = pd.read_csv("./zoo-animal-classification/zoo.csv")
data_total_len = data[data.columns[0]].size
data_train_frac = 0.6
split_index = math.floor(data_total_len*data_train_frac)
training_data = data.iloc[:split_index]
evaluation_data = data.iloc[split_index:];

def preprocess(data):
    features = data.iloc[:, 1:17] #all rows, all the features and no labels
    labels= data.iloc[:, 17] #all rows, label only
    labels = labels-1 # shift value range from 1-7 to be 0-6
    return features, labels

(train_data, train_labels) = preprocess(training_data);
(eval_data, eval_labels) = preprocess(evaluation_data);

print("Training Data: ", train_data)
print("Training Labeks: ", train_labels)

# OR
# data = numpy.loadtxt("./zoo-animal-classification/zoo.csv", delimiter=",")
# X = dataset[:,0:8]
# Y = dataset[:,8]


# build the model
model = keras.Sequential()
model.add(keras.layers.Dense(12, input_shape=(16,), activation='relu'))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=150, batch_size=12)
scores = model.evaluate(eval_data, eval_labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(eval_data)
print(predictions)
