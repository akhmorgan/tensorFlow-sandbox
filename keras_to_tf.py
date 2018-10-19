# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import pandas as pd
import math
import pprint as pp
#import matplotlib.pyplot as plt

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

# preprocess data
def preprocess(data):
    features = data.iloc[:, 1:17] #all rows, all the features and no labels

    labels= data.iloc[:, 17] #all rows, label only
    labels = labels-1 # shift value range from 1-7 to be 0-6
    return features, labels

X, y = preprocess(data);

animal_type=['Mammal', 'Bird', 'Reptile', 'Fish', 'Amphibian', 'Bug', 'Invertebrate']

(train_data, train_labels) = preprocess(training_data);
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

(eval_data, eval_labels) = preprocess(evaluation_data);
print("Evaluation entries: {}, labels: {}".format(len(eval_data), len(eval_labels)))

# build the model
model = keras.Sequential()
model.add(keras.layers.Dense(30,input_shape=(16,)))
model.add(keras.layers.Dense(20))
model.add(keras.layers.Dense(10))

#compile the model
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train the model
def generate_input_fn(data, batch_size=32, epochs=1, shuffle=True):
    features, labels = preprocess(data);

    def _input_fn():
        # convert inputs into a TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # shuffle, repeat, and batch the examples
        if shuffle:
            dataset = dataset.shuffle(1000)
        dataset = dataset.repeat(epochs).batch(batch_size)

        # return the Dataset
        return dataset
    return _input_fn

def get_feature_columns(data):
    feature_columns = [
        tf.feature_column.categorical_column_with_vocabulary_list(
            key=col_name,
            vocabulary_list=data[col_name].unique()
        ) for col_name in data.columns
    ]
    return feature_columns

estimator = tf.keras.estimator.model_to_estimator(keras_model=model)

def train_and_eval(model, train_data=training_data, eval_data=evaluation_data, epochs=1):
    model.train(input_fn=generate_input_fn(train_data, epochs=epochs))
    return model.evaluate(input_fn=generate_input_fn(eval_data, shuffle=False))

print(estimator)
results =  train_and_eval(estimator, epochs=5)
print('Estimator results:', results)

# evaluate the model
print("RESULTS:")
test_loss, test_acc = model.evaluate(eval_data, eval_labels)

print('Test accuracy:', test_acc)
