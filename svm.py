import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
X = [[30],[40],[50],[60],[20],[10],[70]]
y = [0,1,1,1,0,0,1]
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X,y)
X_marks=[[55]]
print(classifier.predict(X_marks))
