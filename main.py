import csv
import math
import pandas as pd
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier

training_set = pd.read_csv('digit-recognizer-data/train.csv')
test = pd.read_csv('digit-recognizer-data/test.csv')
X = training_set.iloc[:,training_set.columns != 'label'].values
y = training_set.iloc[:,training_set.columns == 'label'].values.ravel()

clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(16, 16), random_state=1, activation='logistic', learning_rate_init=0.1)


scores = model_selection.cross_val_score(clf, X, y, scoring="accuracy", cv=10)
print(scores)
print(scores.mean())

clf.fit(X, y)
pred = clf.predict(test)
output = pd.DataFrame({'ImageId': [(x+1) for x in range(len(pred))], 'Label': pred})

output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")