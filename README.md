import random

import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn import linear_model

x=list(range(0,100)) #C
#y=[1.8*F+32 for F in x] #F
y=[1.8*F+32 + random.randint(-3,3) for F in x]
print(f'x:{x}')
print(f'y:{y}')

plt.plot(x,y,'-*g')


x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)

xTrain, xTest , yTrain, yTest = model_selection.train_test_split(x,y, test_size=0.2)
#print(xTrain.shape)
model = linear_model.LinearRegression()
model.fit(xTrain,yTrain)
print(f'coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')


accuracy = model.score(xTest,yTest)
print(f'Accuracy: {round(accuracy*100,2)}')

x = x.reshape(1,-1)[0]
m = model.coef_[0][0]
c = model.intercept_[0]

y=[m*F + c for F in x]
plt.plot(x,y,'-*b')
plt.show()


