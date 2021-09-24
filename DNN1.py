from google.colab import files
uploaded = files.upload()
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('Linear Regression - Sheet1.csv')
print(data)
x = data["X"]
y = data["Y"]
model = Sequential()
model.add(Dense(150, input_dim=1, activation="relu"))
model.add(Dense(150, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
model.fit(x, y, epochs=200)
x = [[400]]
print(model.predict(x))


#link for the dataset : https://www.kaggle.com/tanuprabhu/linear-regression-dataset
