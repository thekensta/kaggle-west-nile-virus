"""scratchpad.py

stuff to test out in sklearn

"""
__author__ = 'chrisk'


import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

samples = ["A", "B", "C"]
data = samples * 2

# LabelEncoder turns strings (names) into numeric representation
# Throws ValueError: y contains new labels: ['D']

print("LabelEncoder turns strings into a numeric representation")
labeller = LabelEncoder()
labeller.fit(samples)

print("from ...")
print(data)
x = labeller.transform(data)
print("to...")
print(type(x))
# <class 'numpy.ndarray'>
print(x)
# [0 1 2 0 1 2]
print("...and back again...")
print(labeller.inverse_transform(x))

print()
print("OneHot Encoder creates a sparse array")
onehot = OneHotEncoder()
y = onehot.fit_transform(x)
print(type(y))
print(y.toarray())

print("OneHotEncoder requires a 2-d array")
x.shape = (6, 1)
y = onehot.fit_transform(x)
print(y.toarray())

print()

print("Pandas get_dummies() does one hot encoding direct from strings")
x = ['a', 'b', 'c', 'a', 'b', 'e']
print(pd.get_dummies(x))


