import numpy as np


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
