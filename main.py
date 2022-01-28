import csv
import math
import pandas as pd
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

training_set = pd.read_csv('digit-recognizer-data/train.csv')
test = pd.read_csv('digit-recognizer-data/test.csv')
X = training_set.iloc[:,training_set.columns != 'label'].values
y = training_set.iloc[:,training_set.columns == 'label'].values.ravel()

print("Data processed!")

#clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(8, 8), random_state=1, activation='logistic', learning_rate_init=0.2)

# A simple k-nearest classifer works significantly better than a Multi-layer perteptron for this problem, i.e. 93% accuracy
# Intuitively if two numbers are the same it's very likely that their pixel representation is also close
clf = KNeighborsClassifier(n_neighbors=80)

print("Classifier created!")

scores = model_selection.cross_val_score(clf, X, y, scoring="accuracy", cv=5)
print(scores)
print(scores.mean())

clf.fit(X, y)
pred = clf.predict(test)
output = pd.DataFrame({'ImageId': [(x+1) for x in range(len(pred))], 'Label': pred})

output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")