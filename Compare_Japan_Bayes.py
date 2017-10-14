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

for i in range(num_factory):






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

def R_NDS(num_factory, Japan_x, Japan_y, Bayes_x, Bayes_y):
    num_Japan_sol = len(Japan_x)
    num_Bayes_sol = len(Bayes_x)
    num_Bayes_pareto = 0
    num_Japan_pareto = 0
    for i in range(num_Bayes_sol):
        fitness1 = Bayes_x[i]
        fitness2 = Bayes_y[i]
        for j in range(num_Japan_sol):
            if Japan_x[j] <= fitness1 and Japan_y[j] <= fitness2:
                if Japan_x[j] < fitness1 or Japan_y[j] <= fitness2:
                    num_Bayes_pareto += 1
                    break
    for i in range(num_Japan_sol):
        fitness1 = Japan_x[i]
        fitness2 = Japan_y[i]
        for j in range(num_Bayes_sol):
            if Bayes_x[j] <= fitness1 and Bayes_y[j] <= fitness2:
                if Bayes_x[j] < fitness1 or Bayes_y[j] <= fitness2:
                    num_Japan_pareto += 1
                    break
    print('Bayes评价指标1： %.2f'%((num_Bayes_sol-num_Bayes_pareto)/num_Bayes_sol))
    print('Bayes评价指标2： %s'%(num_Bayes_sol-num_Bayes_pareto))
    print('Japan评价指标1： %.2f'%((num_Japan_sol-num_Japan_pareto)/num_Japan_sol))
    print('Japan评价指标1： %s'%(num_Japan_sol-num_Japan_pareto))

R_NDS(num_factory, Japan_x, Japan_y, Bayes_x, Bayes_y)