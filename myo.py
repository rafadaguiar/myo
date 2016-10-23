from os import walk
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.layers.core import Dropout
from keras.wrappers.scikit_learn import KerasClassifier

import pandas as pd
import numpy as np
import re

NUMBER_OF_FEATURES = 8
NUMBER_OF_TIME_STEPS = 50
NUMBER_OF_CLASSES = 6
TEMP_DIR = "temp/"
SEED = 7

np.random.seed(SEED)

def preprocessing(input_dir):
    def to_categorical(y):
        nb_classes = np.max(y)
        Y = np.zeros((len(y), nb_classes))
        for i in range(0, len(y)):
            Y[i, y[i]-1] = 1.
            return Y
    print "Preprocessing files..."
    file_name_pattern = re.compile("Gesture(?P<gesture_type>\d+)_Example(?P<example_number>\d+).txt")
    X = []
    Y = []
    for dir_path, dir_names, file_names in walk(input_dir):
            for file_name in file_names:
                example = pd.read_csv(dir_path+file_name, header=None).as_matrix()
                example_nb_time_steps, example_nb_features = example.shape
                if example_nb_time_steps != NUMBER_OF_TIME_STEPS:
                    missing_values = np.zeros(
                        (NUMBER_OF_TIME_STEPS - example_nb_time_steps, NUMBER_OF_FEATURES))
                    example = np.vstack((example, missing_values))
                X.append(example)
                gesture_type, example_number = file_name_pattern.match(file_name).groups()
                Y.append(int(gesture_type))

    X = np.stack(X) # shape(number of samples, number of time steps, number of features)
    Y = to_categorical(np.array(Y))
    np.save(TEMP_DIR+"X.npy", X)
    np.save(TEMP_DIR+"Y.npy", Y)
    return X, Y

def load_data(input_dir = "gesture_data/"):
    for dir_path, dir_names, file_names in walk(TEMP_DIR):
        if "X.npy" not in file_names or "Y.npy" not in file_names:
            return preprocessing(input_dir)
    print "Loading files..."
    return np.load(TEMP_DIR+"X.npy"), np.load(TEMP_DIR+"Y.npy")

def build_model(first_layer_neurons, second_layer_neurons):
    model = Sequential()
    model.add(LSTM(first_layer_neurons, input_dim=NUMBER_OF_FEATURES, dropout_U=0.3))
    model.add(Dense(second_layer_neurons))
    model.add(Dropout(0.2))
    model.add(Dense(NUMBER_OF_CLASSES, activation="softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model

def cross_validation(build_model, X_train, y_train):
    cv_model = KerasClassifier(
        build_fn=build_model,
        nb_epoch=50,
        batch_size=100,
        verbose=2
    )
    param_grid = {
        "first_layer_neurons":[10, 50, 100, 150],
        "second_layer_neurons":[10, 50, 100, 150]
    }
    grid = GridSearchCV(estimator=cv_model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for params, mean_score, scores in grid_result.grid_scores_:
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
    return grid_result

def predict(model, X_test, y_test = None):
    predictions = model.predict(X_test)
    get_class = lambda classes_probabilities: np.argmax(classes_probabilities) + 1
    y_pred = np.array(map(get_class, predictions))
    if y_test is not None:
        y_true = np.array(map(get_class, y_test))
        print accuracy_score(y_true, y_pred)
    return y_pred

def main():
    X, Y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

    model = build_model(150, 100)
    model.fit(X_train, y_train, nb_epoch=200, batch_size=100, verbose=2)

    scores = model.evaluate(X_train, y_train)
    print("%s: %.2f" % (model.metrics_names[1], scores[1]))

    predict(model, X_test, y_test)

if __name__ == '__main__':
    main()
