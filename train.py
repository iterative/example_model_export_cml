from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
import os
import joblib
import numpy as np

# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")


# Fit a model
depth = 5
clf = RandomForestClassifier(max_depth=depth)
clf.fit(X_train,y_train)

# Calculate accuracy
acc = clf.score(X_test, y_test)
print(acc)

# Create model folder if it does not yet exist
if not os.path.exists('model'):
    os.makedirs('model')

# Write metrics to file
with open("model/metrics.txt", 'w+') as outfile:
        outfile.write("Accuracy: " + str(acc) + "\n")

# Plot confusion matrix
disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, cmap=plt.cm.Blues)
plt.savefig('model/confusion_matrix.png')

# Save the model
joblib.dump(clf, "model/random_forest.joblib")
