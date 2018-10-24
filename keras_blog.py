# TensorFlow and tf.keras
import tensorflow as tf
# Only necessary if you're using Keras (obviously)
from tensorflow import keras

# Helper libraries
import numpy as np
import pandas as pd
import math
import pprint as pp

data = pd.read_csv("./zoo-animal-classification/zoo.csv")

# Shuffle
data = data.sample(frac=1).reset_index(drop=True)
# Split
data_total_len = data[data.columns[0]].size
data_train_frac = 0.6
split_index = math.floor(data_total_len*data_train_frac)
training_data = data.iloc[:split_index]
evaluation_data = data.iloc[split_index:];

column_count = 18;
label_column_index = 17 # Zero based index (so this is the 18th column)

def preprocess(data):
    X = data.iloc[:, 1:column_count-1] #all rows, all the features and no labels
    y = data.iloc[:, label_column_index] #all rows, label only
    y = y-1 # shift label value range from 1-7 to 0-6
    return X, y

(train_data, train_labels) = preprocess(training_data);
(eval_data, eval_labels) = preprocess(evaluation_data);

model = keras.Sequential()
model.add(keras.layers.Dense(30, input_shape=(16,)))
model.add(keras.layers.Dense(20, activation=tf.nn.relu))
model.add(keras.layers.Dense(7, activation=tf.nn.softmax))

epochs = 150
batch_size = 12

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_data, train_labels, epochs = epochs, batch_size = batch_size)

model.evaluate(eval_data, eval_labels)

animal_type=['Mammal', 'Bird', 'Reptile', 'Fish', 'Amphibian', 'Bug', 'Invertebrate']
prediction_data = eval_data

predictions = model.predict(prediction_data)

for i, prediction in enumerate(predictions):
  predicted_animal = animal_type[prediction.argmax(axis=-1)]
  correct_animal = animal_type[eval_labels.iloc[i]]
  print("Predicted:   {}\nActual answer:   {}\nProbabilities: {}\n".format(predicted_animal, correct_animal, prediction))
