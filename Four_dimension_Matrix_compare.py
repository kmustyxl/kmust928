import matplotlib.pyplot as plt
from Bayes_Multi_object import *
from Bayes_without_Matrix import *
import time
time1 = time.clock()
Bayes = B_All_factory_dominated(pop_gen, ls_frequency,update_popsize, num_factory)
time2 = time.clock()
print('贝叶斯带四维矩阵消耗时间：%s'%(time2-time1))
time1 = time.clock()
Bayes_WM = B_All_factory_dominated_without_Matrix(pop_gen, ls_frequency,update_popsize, num_factory)
time2 = time.clock()
print('贝叶斯不带四维矩阵消耗时间：%s'%(time2-time1))

Bayes_x = []
Bayes_y = []
Bayes_no_x=[]
Bayes_no_y=[]
for individual in Bayes:
    Bayes_x.append(individual[-2])
    Bayes_y.append(individual[-1])
for individual in Bayes_WM:
    Bayes_no_x.append(individual[-2])
    Bayes_no_y.append(individual[-1])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Fitness')
ax.set_ylabel('Carbon')
ax.plot(Bayes_no_x, Bayes_no_y, 'rD',label='Bayes_WM')
ax.plot(Bayes_x, Bayes_y, 'gD',label='Bayes')
plt.grid()
plt.legend()
plt.show()