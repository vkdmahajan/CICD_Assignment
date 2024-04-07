import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import pickle
import numpy as np

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

# Trying to Use only the first 10 samples for training
model = LogisticRegression(solver='lbfgs').fit(X[:10], y[:10])

with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
