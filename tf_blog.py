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

X, y = preprocess(data);

(train_data, train_labels) = preprocess(training_data);
(eval_data, eval_labels) = preprocess(evaluation_data);


feature_columns = [
    tf.feature_column.categorical_column_with_vocabulary_list(
        key = col_name,
        vocabulary_list = data[col_name].unique()
    ) for col_name in X.columns
]

deep_features = [tf.feature_column.indicator_column(col) for col in feature_columns]

model = tf.estimator.DNNClassifier(
    feature_columns = deep_features,
    hidden_units = [30,20,10],
    n_classes = 7
)

epochs = 150
batch_size = 12

def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((dict(train_data), train_labels))
    # shuffle, repeat, and batch the examples
    dataset = dataset.shuffle(1000)
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset

model.train(input_fn = train_input_fn)

def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((dict(eval_data), eval_labels))
    # repeat, and batch the examples
    # NOTE: do not Shuffle
    dataset = dataset.repeat(1).batch(batch_size)
    return dataset

model.evaluate(input_fn=eval_input_fn)

animal_type=['Mammal', 'Bird', 'Reptile', 'Fish', 'Amphibian', 'Bug', 'Invertebrate']
prediction_data = evaluation_data

def predict_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((dict(prediction_data), eval_labels))
    # repeat, and batch the examples
    # NOTE: do not Shuffle
    dataset = dataset.repeat(1).batch(batch_size)
    return dataset

predictions = model.predict(input_fn = predict_input_fn)

predictions = [prediction["probabilities"] for prediction in predictions]

for i, prediction in enumerate(predictions):
  predicted_animal = animal_type[prediction.argmax(axis=-1)]
  correct_animal = animal_type[eval_labels.iloc[i]]
  print("Predicted:   {}\nActual answer:   {}\nProbabilities: {}\n".format(predicted_animal, correct_animal, prediction))
