import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

print(digits.data)
print(digits.target)
print(digits.images[0])

clf = svm.SVC(gamma=0.001, C=100)
x,y = digits.data[:-1], digits.target[:-1]

print('-------------')
print(len(x))
print(y[301])
print('-------------')

clf.fit(x,y)

print('Prediction:', clf.predict(digits.data[-3]))
print(clf.predict(digits.data[-3]))

plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()