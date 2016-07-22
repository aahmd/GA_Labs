import pandas as pd

import pylab as pl

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.cross_validation import cross_val_score

X = pd.read_csv("https://s3.amazonaws.com/demo-datasets/wine.csv")
y = X.pop('high_quality')

le = LabelEncoder()
X.color = le.fit_transform(X.color)

results = []

for n in range(1, 50, 2):
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=n))
    c_val = cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()
    results.append([n, c_val])


print results

results = pd.DataFrame(results, columns=["n", "accuracy"])

pl.plot(results.n, results.accuracy)
pl.title("Accuracy with Increasing K")
pl.show()
