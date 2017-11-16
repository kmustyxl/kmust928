import time
import random
import numpy as np


def Green_Calcfitness(n, m, sort, test_data, v):
    c_time1 = np.zeros([n, m])
    c_time1[0][0] = test_data[sort[0]][0] / v[0][0]
    for i in range(1, n):
        c_time1[i][0] = c_time1[i - 1][0] + test_data[sort[i]][0] / v[i][0]
    for i in range(1, m):
        c_time1[0][i] = c_time1[0][i - 1] + test_data[0][i] / v[0][i]
    for i in range(1, n):
        for k in range(1, m):
            c_time1[i][k] = test_data[sort[i]][k] / v[i][k] + max(c_time1[i - 1][k], c_time1[i][k - 1])
    return c_time1[n - 1][m - 1]

def ECF(num_job, num_machine, test_data, num_factory, v, sort):
    first_job = []
    factory_job_set = [[] for i in range(num_factory)]
    first_job_index = []
    factory_data = [[] for i in range(num_factory)]
    factory_fit = [[] for i in range(num_factory)]
    c_time = np.zeros([num_job, num_machine])
    #确定每一个工厂的第一个排序工件号
    temp = []
    k = 0
    for i in range(num_factory):
        factory_job_set[i].append(sort[i])
        factory_data[i].append(test_data[sort[i]])
    while True:
        if k == num_job - num_factory :
            break
        for i in range(num_factory):
            factory_job_set[i].append(sort[num_factory + k])
            factory_data[i].append(test_data[sort[num_factory + k]])
            factory_fit[i]= Green_Calcfitness(len(factory_data[i]),num_machine,factory_job_set[i],test_data,v)
        index = np.argsort(factory_fit)[0]
        for i in range(num_factory):
            if i == index:
                continue
            else:
                factory_job_set[i].pop()
                factory_data[i].pop()
        k += 1
    return  factory_job_set





def TCE(num_job, num_machine, test_data,v, num_factory, sort):
    factory_job_set = ECF(num_job, num_machine, test_data, num_factory, v, sort)
    per_consumption_Standy = 1
    standy_time = 0
    all_f_f1 = 0
    all_f_f2 = 0
    fitness1_2 = [[0,0] for i in range(num_factory)]
    for f in range(num_factory):
        Energy_consumption = 0
        for k in range(num_machine):
            for i in range(len(factory_job_set[f])):
                per_consumption_V = 4 * v[i][k] * v[i][k] # 机器加速单位时间能源消耗
                Energy_consumption += test_data[factory_job_set[f][i]][k] / v[i][k] * per_consumption_V
        C_time = Green_Calcfitness(len(factory_job_set[f]),num_machine,factory_job_set[f], test_data, v)
        for k in range(num_machine):
            Key_time = C_time
            for i in range(len(factory_job_set[f])):
                Key_time -= test_data[factory_job_set[f][i]][k] / v[i][k]
            standy_time += Key_time
        Energy_consumption += standy_time * per_consumption_Standy
        fitness1_2[f][0] = int(C_time)
        fitness1_2[f][1] = int(Energy_consumption)
        all_f_f2 += fitness1_2[f][1]
    fitness1_2 = sorted(fitness1_2, key=lambda x:x[0])
    all_f_f1 = fitness1_2[-1][0]
    return all_f_f1, all_f_f2


def Multi_initial_Japan(num_machine, num_factory, num_job, test_data,v):
    #每个工厂的工件数
    job_set = [i for i in range(num_job)]
    #J_len_job = [len(J_factory_job_set[i]) for i in range(num_factory)]
    #每个工厂的初始数据
    J_Mat_pop = [[0 for i in range(num_job + 2)] for j in range(200)] #最后两个元素分别是经济指标和绿色指标
    J_non_dominated_pop = [] 
    #for i in range(num_factory):
    for j in range(200):
        sort = random.sample(job_set, num_job)
        for k in range(num_job + 1):
            if k == num_job:
                J_Mat_pop[j][k], J_Mat_pop[j][k+1]= TCE(num_job, num_machine, test_data,v, num_factory, sort)
            else:
                J_Mat_pop[j][k] = sort[k]
    J_Mat_pop = sorted(J_Mat_pop, key= lambda x:x[-2])
    for j in range(200):
        compare_fitness1 = J_Mat_pop[j][num_job]
        compare_fitness2 = J_Mat_pop[j][num_job+1]
        b_non_dominated = True
        for k in range(200):
            if j != k:
                if J_Mat_pop[k][num_job] <= compare_fitness1 and J_Mat_pop[k][num_job+1] <= compare_fitness2:
                    if J_Mat_pop[k][num_job] < compare_fitness1 or J_Mat_pop[k][num_job+1] < compare_fitness2:
                        b_non_dominated = False
                        break
        if b_non_dominated == True:
            if J_Mat_pop[j][0:num_job+2] not in J_non_dominated_pop:
                J_non_dominated_pop.append(J_Mat_pop[j][0:num_job+2])

    return J_Mat_pop, J_non_dominated_pop

def selection_prob(J_Mat_pop, num_factory, num_job):
    weights = [0.5, 0.5]
    weights[0] = np.random.random()
    weights[1] = 1 - weights[0]
    J_prob_Matrix =[-1 for i in range(200)]
   # J_len_job = [len(J_factory_job_set[i]) for i in range(num_factory)]
    #for i in range(num_factory):
    for individual in J_Mat_pop:
        scalar_fitness = weights[0] * individual[num_job] + weights[1] * individual[num_job + 1]
        individual.append(scalar_fitness)
    J_Mat_pop = sorted(J_Mat_pop, key=lambda x: x[num_job + 2])
    #for i in range(num_factory):
    f_max = J_Mat_pop[-1][num_job + 2]
    Ps_x_Denominator = 0
    for individual in J_Mat_pop:               #求出Step2中概率方程的分母
        Ps_x_Denominator += float(f_max - individual[num_job + 2])
    for j in range(200):
        J_prob_Matrix[j] = float(f_max - J_Mat_pop[j][num_job + 2]) / Ps_x_Denominator
    return J_prob_Matrix

def select_parents(J_prob_Matrix, J_Mat_pop, num_job):
    #采用轮盘赌选取一对父代个体
    temp = 0.0
    rand_p = random.random()
    if rand_p < J_prob_Matrix[0]:
        J_select_1 = J_Mat_pop[0][0:num_job]
    else:
        j = 0
        temp += J_prob_Matrix[0]
        while temp < rand_p:
            j += 1
            temp += J_prob_Matrix[j]
        J_select_1 = J_Mat_pop[j][0:num_job]
    temp = 0.0
    rand_p = random.random()
    if rand_p < J_prob_Matrix[0]:
        J_select_2 = J_Mat_pop[0][0:num_job]
    else:
        j = 0
        temp += J_prob_Matrix[0]
        while temp < rand_p:
            j += 1
            temp += J_prob_Matrix[j]
        J_select_2 = J_Mat_pop[j][0:num_job]
    return J_select_1, J_select_2

def crossover(J_select_1, J_select_2, num_job):
    J_child_1 = [lambda x:0 for x in range(num_job)]
    temp1 = random.randint(1,num_job - 3)
    while True:
        temp2 = random.randint(1,num_job - 3)
        if temp1 != temp2:
            break
    rand_pos1 = max(temp1, temp2)
    rand_pos2 = min(temp1, temp2)
    for i in range(rand_pos2):
        J_child_1[i] = J_select_1[i]
    for i in range(num_job - 1, rand_pos1 , -1):
        J_child_1[i] = J_select_1[i]
    for i in range(num_job):
        if J_select_2[i] not in J_child_1 and rand_pos2 <= num_job - 1:
            J_child_1[rand_pos2] = J_select_2[i]
            rand_pos2  += 1
    return J_child_1

def mutation(J_child_1,num_job):
    #num_job = len(J_child_1)
    J_temp_individual = [-1 for i in range(num_job)]
    temp1 = random.randint(1,num_job - 2)
    while True:
        temp2 = random.randint(1,num_job - 2)
        if abs(temp1 - temp2) >= 2:
            break
    J_temp_individual = J_child_1[:]
    rand_pos1 = max(temp1, temp2)
    rand_pos2 = min(temp1, temp2)
    J_child_1[rand_pos2] = J_child_1[rand_pos1]
    for i in range(rand_pos2 + 1, rand_pos1+1):
        J_child_1[i] = J_temp_individual[i - 1]
    return J_child_1

def generate_newpop(num_factory, num_job, J_prob_Matrix, J_Matpop):
    #J_len_job = [len(J_factory_job_set[i]) for i in range(num_factory)]
    cross_prob = 0.9
    mutation_prob = 0.6
    J_newpop = []
    #for i in range(num_factory):
    for j in range(190):
        #根据crossover概率进行交叉操作
        J_select_1, J_select_2 = select_parents(J_prob_Matrix, J_Matpop,num_job)
        temp1 = random.random()
        if temp1 <= cross_prob:
            child = crossover(J_select_1, J_select_2, num_job)
        else:
            temp2 = random.random()
            if temp2 <= 0.5:
                child = J_select_1
            else:
                child = J_select_2
        #根据mutation概率进行变异操作
        temp3 = random.random()
        if temp3 <= mutation_prob:
            child = mutation(child,num_job)
        J_newpop.append(child)
    for j in range(10):
        index = random.randint(0,199)
        J_newpop.append(J_Matpop[index][0:num_job])
    return J_newpop

def local_search(num_factory, J_newpop, num_job, num_machine,v,test_data):
    #J_len_job = [len(J_factory_job_set[i]) for i in range(num_factory)]
    local_search_prob = 0.8
    #for i in range(num_factory):
    for j in range(200):
        weights = [0.5, 0.5]
        weights[0] = np.random.random()
        weights[1] = 1 - weights[0]
        sort = list(J_newpop[j][0:num_job])
        C_time, Energy_consumption = TCE(num_job, num_machine, test_data,v, num_factory, sort)
        scalar_fitness = weights[0] * C_time + weights[1] * Energy_consumption
        J_newpop[j].append(C_time)
        J_newpop[j].append(Energy_consumption)
        J_newpop[j].append(scalar_fitness)
    J_newpop = sorted(J_newpop, key= lambda x:x[num_job + 2])
    #for i in range(num_factory):
    for j in range(5):
        r = random.random()
        if r <= local_search_prob:
            k_neighbor = []
            for k in range(2):
                weights = [0.5, 0.5]
                weights[0] = np.random.random()
                weights[1] = 1 - weights[0]
                child = mutation(J_newpop[j][0:num_job],num_job)
                C_time, Energy_consumption = TCE(num_job, num_machine, test_data,v, num_factory, child)
                scalar_fitness = weights[0] * C_time + weights[1] * Energy_consumption
                child.append(C_time)
                child.append(Energy_consumption)
                child.append(scalar_fitness)
                k_neighbor.append(child)
            k_neighbor = sorted(k_neighbor, key=lambda x:x[num_job + 2])
            if k_neighbor[0][num_job + 2] < J_newpop[j][num_job + 2]:
                J_newpop[j][:] = k_neighbor[0][:]
            else:
                continue
    return J_newpop

def J_select_non_dominated_pop(num_factory, num_job, J_Mat_pop):
    J_temp_non_dominated_pop = []
    #for i in range(num_factory):
    for j in range(len(J_Mat_pop)):
        compare_fitness1 = J_Mat_pop[j][num_job]
        compare_fitness2 = J_Mat_pop[j][num_job+1]
        b_non_dominated = True
        for k in range(len(J_Mat_pop)):
            if j != k:
                if J_Mat_pop[k][num_job] <= compare_fitness1 and J_Mat_pop[k][num_job+1] <= compare_fitness2:
                    if J_Mat_pop[k][num_job] < compare_fitness1 or J_Mat_pop[k][num_job+1] < compare_fitness2:
                        b_non_dominated = False
                        break
        if b_non_dominated == True:
            if J_Mat_pop[j][0:num_job+2] not in J_temp_non_dominated_pop:
                J_temp_non_dominated_pop.append(J_Mat_pop[j][0:num_job+2])
    return J_temp_non_dominated_pop

def update_non_dominated(J_non_dominated_pop, J_temp_non_dominated,num_job,num_factory):
    #J_len_job = [len(J_factory_job_set[i]) for i in range(num_factory)]
    #for i in range(num_factory):
    for j in range(len(J_temp_non_dominated)):
        if J_temp_non_dominated[j][0:num_job+2] not in J_non_dominated_pop:
            J_non_dominated_pop.append(J_temp_non_dominated[j][0:num_job+2])
    J_non_dominated_pop = J_select_non_dominated_pop(num_factory, num_job, J_non_dominated_pop)
    return J_non_dominated_pop

def Japan_Multi_object(pop_gen,test_time,v,num_job, num_machine, test_data, num_factory):
    test_timeup = time.clock()
    #J_factory_job_set = NEH2(num_job, num_machine, test_data, num_factory,v)
    #global J_len_job
    #J_len_job = [len(J_factory_job_set[i]) for i in range(num_factory)]
    J_Mat_pop, J_non_dominated_pop = Multi_initial_Japan(num_machine, num_factory, num_job, test_data,v)
    J_prob_Matrix = selection_prob(J_Mat_pop, num_factory, num_job)
    for gen in range(pop_gen):
        J_Mat_pop =  generate_newpop(num_factory, num_job, J_prob_Matrix, J_Mat_pop)
        J_Mat_pop = local_search(num_factory, J_Mat_pop, num_job, num_machine,v,test_data)
        J_temp_non_dominated = J_select_non_dominated_pop(num_factory, num_job, J_Mat_pop)
        J_non_dominated_pop = update_non_dominated(J_non_dominated_pop, J_temp_non_dominated,num_job,num_factory)
        test_timedown = time.clock()
        if float(test_timedown-test_timeup)>=float(test_time):
            break
    return J_non_dominated_pop#, float(test_timedown-test_timeup)