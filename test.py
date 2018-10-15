# TensorFlow and tf.keras
import tensorflow as tf
#from tensorflow import keras

# Helper libraries
import numpy as np
import pandas as pd
import math
import pprint as pp

print(tf.__version__)


#trainingData = pd.read_csv("training-data.csv", header=None).as_matrix()
#testData = pd.read_csv("test-data.csv", header=None).as_matrix()
#train_input_fn
#eval_input_fn

# parse data
data = pd.read_csv("./zoo-animal-classification/zoo.csv")
#print(data.head(6))
#print(data.describe())

# shuffle and split the dataset
data = data.sample(frac=1).reset_index(drop=True)
data_total_len = data[data.columns[0]].size

# use 60% of the data for training and 40% for evaluating
data_train_frac = 0.6
split_index = math.floor(data_total_len*data_train_frac)

print(split_index)

train_data = data.iloc[:split_index]
eval_data = data.iloc[split_index:];

# preprocess data
def preprocess(data):
    X = data.iloc[:, 1:17] #all rows, all the features and no labels
    y= data.iloc[:, 17] #all rows, label only
    y = y-1 # shift value range from 1-7 to be 0-6
    return X, y

# This is used later in feature feature_columns

X, y = preprocess(eval_data);

#print(X.head());
#print(y.head());

# define input functions
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

#print(generate_input_fn(train_data)())

# list unique values in each column
#i=0
#for col_name in data.columns:
#    print("{}. {}: {}".format(i, col_name, data[col_name].unique()))
#    i+=1

# feature columns
def get_feature_columns(data):
    feature_columns = [
        tf.feature_column.categorical_column_with_vocabulary_list(
            key=col_name,
            vocabulary_list=data[col_name].unique()
        ) for col_name in data.columns
    ]
    return feature_columns

#feature_columns = get_feature_columns(X)
#pp.pprint(feature_columns)

#define linear model
linear_model = tf.estimator.LinearClassifier(
    feature_columns= get_feature_columns(X),
    n_classes=7
)

def train_and_eval(model, train_data=train_data, eval_data= eval_data, epochs=1):
    model.train(input_fn=generate_input_fn(train_data, epochs=epochs))
    return model.evaluate(input_fn=generate_input_fn(eval_data, shuffle=False))

linear_results =  train_and_eval(linear_model, epochs=1)
print(linear_results)

#define DNN model
# wrap feature columns in indicator columns so dnn can handle them
deep_features = [tf.feature_column.indicator_column(col) for col in get_feature_columns(X)]
deep_model = tf.estimator.DNNClassifier(
    feature_columns= deep_features,
    hidden_units=[30,20,10],
    n_classes=7
)

deep_results =  train_and_eval(deep_model, epochs=5)
print(deep_results)

#make predictions
animal_type=['Mammal', 'Bird', 'Reptile', 'Fish', 'Amphibian', 'Bug', 'Invertebrate']

predict_data = eval_data[10:15]
predictions = deep_model.predict(input_fn=generate_input_fn(predict_data, shuffle=False))

#for i, prediction in enumerate(predictions):
#    predicted_animal = animal_type[int(prediction["classes"][0].decode("utf8"))]
#    correct_animal = animal_type[predict_data["class_type"].iloc[i]-1]
#    print("Predicted: {} \n Actual: {}\n".format(predicted_animal, correct_animal))

for i, prediction in enumerate(predictions):
    predicted_animal = animal_type[int(prediction["classes"][0].decode("utf8"))]
    correct_animal = animal_type[predict_data["class_type"].iloc[i]-1]
    if int(prediction["classes"][0].decode("utf8")) != predict_data["class_type"].iloc[i]-1:
        print("[WRONG] Predictions: {} with probabilities {}\n Actual answer: {}"
        .format(
            predicted_animal,
            prediction["probabilities"],
            correct_animal)
        )

####

#classifier = tf.estimator.DNNClassifier(
#    feature_columns=feature_columns,
#    hidden_units=[30, 20, 10],
#    n_classes=7,
#    model_dir="models/"
#)

#classifier.train(input_fn=train_input_fn);
#classifier.evaluate(input_fn=eval_input_fn);
