import numpy as np
import random

def getPositon():
    #a = np.mat([[2, 5, 7, 8, 9, 9], [6, 7, 5, 4, 6, 4], [6, 7, 5, 4, 6, 4]])
    a = np.zeros((5,5,5))
    a[3,2,1] = 5
    height, raw, column = a.shape  # get the matrix of a raw and column

    _positon = np.argmax(a)  # get the index of max in the a
    print(_positon)
    h = int(_positon / raw/column)
    m, n = divmod(_positon-(raw*column)*h, column)
    print("The height is", h)
    print("The raw is ", m)
    print("The column is ", n)
    print("The max of the a is ", a[h, int(m), int(n)])
    print(a)

def Multi_3d_test():
    pass
I = set([1,2,3,4])
J = set([4,5,6,'A'])
K = [10,11]
kk = [[1,2,3],[4,5,6,'A'],[10,11]]
print(I&J)
from itertools import combinations
import itertools

print(list(itertools.product(kk[0],kk[1])))
a = random.randint(0,1)
print(a)


a = [[1,2,3,4,5,89,997] ,[1,2,3,4,5,79,1997]]

b=[1,2,3,4,5]
f1max = max(a,key=lambda x:x[-2])
print(f1max)
print(b[0:-1])
for i in range(1,5)[::-1]:
    print(i)
I = [1,2,3,4]
print(random.sample(I,4))