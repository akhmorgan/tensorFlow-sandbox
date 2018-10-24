# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import pandas as pd
import math
import pprint as pp

print(tf.__version__)

# load data
data = pd.read_csv("./zoo-animal-classification/zoo.csv")

# shuffle and split the dataset
data = data.sample(frac=1).reset_index(drop=True)
data_total_len = data[data.columns[0]].size
data_train_frac = 0.6
split_index = math.floor(data_total_len*data_train_frac)
training_data = data.iloc[:split_index]
evaluation_data = data.iloc[split_index:];

column_count = 18;
label_column_index = 17 # Zero based index (so this is the 18th column)

# preprocess data
def preprocess(data):
    features = data.iloc[:, 1:column_count-1] #all rows, all the features and no labels
    pp.pprint(features.describe());
    labels= data.iloc[:, label_column_index] #all rows, label only
    labels = labels-1 # shift value range from 1-7 to be 0-6
    return features, labels

animal_type=['Mammal', 'Bird', 'Reptile', 'Fish', 'Amphibian', 'Bug', 'Invertebrate']

(train_data, train_labels) = preprocess(training_data);
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print("Training labels: ", train_labels.unique())
(eval_data, eval_labels) = preprocess(evaluation_data);
print("Evaluation entries: {}, labels: {}".format(len(eval_data), len(eval_labels)))
print("Eval labels: ", eval_labels.unique())
#pp.pprint(evaluation_data)

# build the model
model = keras.Sequential()
model.add(keras.layers.Dense(30,input_shape=(16,)))
model.add(keras.layers.Dense(20, activation=tf.nn.relu))
model.add(keras.layers.Dense(7, activation=tf.nn.softmax))

#compile the model
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train the model
history = model.fit(train_data, train_labels, epochs=150, batch_size=12)
scores = model.evaluate(eval_data, eval_labels)

print(history)

# evaluate the model
print("RESULTS:")
test_loss, test_acc = model.evaluate(eval_data, eval_labels)

print('Test accuracy:', test_acc)

# make predictions
predictions = model.predict(eval_data)
print(predictions[0])
