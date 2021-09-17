import csv
import math
import pandas as pd
from sklearn.neural_network import MLPClassifier

training_set = pd.read_csv('digit-recognizer-data/train.csv')
x_test = pd.read_csv('digit-recognizer-data/test.csv')
x_train = training_set.iloc[:,training_set.columns != 'label'].values
y_train = training_set.iloc[:,training_set.columns == 'label'].values.ravel()

print(x_test)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#clf.fit(X, y)