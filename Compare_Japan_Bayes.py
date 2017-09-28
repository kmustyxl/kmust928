import matplotlib.pyplot as plt
from Bayes_Multi_object import *
from Japan_Multi_object import *
time1 = time.clock()
Japan = Japan_Multi_object(pop_gen)
time2 = time.clock()
print('日本人程序共运行：%s'%(time2-time1))
time1 = time.clock()
Bayes = Green_Bayes_net(pop_gen, ls_frequency,update_popsize)
time2 = time.clock()
print('贝叶斯程序共运行：%s'%(time2-time1))
job_len = [len(factory_job_set[i]) for i in range(num_factory)]
Japan_x = []
Japan_y = []
Bayes_x = []
Bayes_y = []
for individual in Japan[0]:
    Japan_x.append(individual[job_len[0]])
    Japan_y.append(individual[job_len[0] + 1])
for individual in Bayes[0]:
    Bayes_x.append(individual[job_len[0]])
    Bayes_y.append(individual[job_len[0] + 1])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Fitness')
ax.set_ylabel('Carbon')
ax.plot(Japan_x, Japan_y, 'rD',label='Japan')
ax.plot(Bayes_x, Bayes_y, 'gD',label='Bayes')
plt.grid()
plt.legend()
plt.show()
