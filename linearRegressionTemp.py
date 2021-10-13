import sklearn
import matplotlib.pyplot as plt
import numpy as np

#y = mx + c
#F= 1.8*C + 32

x = list(range(0,40)) #C
y = [1.8*F+32 for F in x]    #F
print(f'x:{x}')
print(f'y:{y}')

plt.plot(x,y,'-*g')
plt.show()