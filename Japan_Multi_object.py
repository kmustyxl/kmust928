from AssignRule import *
#from New_Bayes import *
#from Compare_result import *
import time
from itertools import combinations
import itertools


def Multi_initial_Japan(num_machine, num_factory, J_factory_job_set, test_data,v):
    #每个工厂的工件数
    J_len_job = [len(J_factory_job_set[i]) for i in range(num_factory)]
    #每个工厂的初始数据
    J_Mat_pop =[[[0 for i in range(J_len_job[k] + 2) ]for j in range(200)] for k in range(num_factory)] #最后两个元素分别是经济指标和绿色指标
    J_non_dominated_pop = [[] for k in range(num_factory)]
    for i in range(num_factory):
        for j in range(200):
            sort = random.sample(J_factory_job_set[i], J_len_job[i])
            for k in range(J_len_job[i] + 1):
                if k == J_len_job[i]:
                    J_Mat_pop[i][j][k], J_Mat_pop[i][j][k+1]= TCE(J_len_job[i], num_machine, sort, test_data,v)
                else:
                    J_Mat_pop[i][j][k] = sort[k]
        J_Mat_pop[i] = sorted(J_Mat_pop[i], key= lambda x:x[-2])
        for j in range(200):
            compare_fitness1 = J_Mat_pop[i][j][J_len_job[i]]
            compare_fitness2 = J_Mat_pop[i][j][J_len_job[i]+1]
            b_non_dominated = True
            for k in range(200):
                if j != k:
                    if J_Mat_pop[i][k][J_len_job[i]] <= compare_fitness1 and J_Mat_pop[i][k][J_len_job[i]+1] <= compare_fitness2:
                        if J_Mat_pop[i][k][J_len_job[i]] < compare_fitness1 or J_Mat_pop[i][k][J_len_job[i]+1] < compare_fitness2:
                            b_non_dominated = False
                            break
            if b_non_dominated == True:
                if J_Mat_pop[i][j][0:J_len_job[i]+2] not in J_non_dominated_pop[i]:
                    J_non_dominated_pop[i].append(J_Mat_pop[i][j][0:J_len_job[i]+2])

    return J_Mat_pop, J_non_dominated_pop

def selection_prob(J_Mat_pop, num_factory, J_factory_job_set):
    weights = [0.5, 0.5]
    weights[0] = np.random.random()
    weights[1] = 1 - weights[0]
    J_prob_Matrix = [[-1 for i in range(200)] for j in range(num_factory)]
    J_len_job = [len(J_factory_job_set[i]) for i in range(num_factory)]
    for i in range(num_factory):
        for individual in J_Mat_pop[i]:
            scalar_fitness = weights[0] * individual[J_len_job[i]] + weights[1] * individual[J_len_job[i] + 1]
            individual.append(scalar_fitness)
        J_Mat_pop[i] = sorted(J_Mat_pop[i], key=lambda x: x[J_len_job[i] + 2])
    for i in range(num_factory):
        f_max = J_Mat_pop[i][-1][J_len_job[i] + 2]
        Ps_x_Denominator = 0
        for individual in J_Mat_pop[i]:               #求出Step2中概率方程的分母
            Ps_x_Denominator += float(f_max - individual[J_len_job[i] + 2])
        for j in range(200):
            J_prob_Matrix[i][j] = float(f_max - J_Mat_pop[i][j][J_len_job[i] + 2]) / Ps_x_Denominator
    return J_prob_Matrix

def select_parents(J_prob_Matrix, J_Mat_pop, J_job_len):
    #采用轮盘赌选取一对父代个体
    temp = 0.0
    rand_p = random.random()
    if rand_p < J_prob_Matrix[0]:
        J_select_1 = J_Mat_pop[0][0:J_job_len]
    else:
        j = 0
        temp += J_prob_Matrix[0]
        while temp < rand_p:
            j += 1
            temp += J_prob_Matrix[j]
        J_select_1 = J_Mat_pop[j][0:J_job_len]
    temp = 0.0
    rand_p = random.random()
    if rand_p < J_prob_Matrix[0]:
        J_select_2 = J_Mat_pop[0][0:J_job_len]
    else:
        j = 0
        temp += J_prob_Matrix[0]
        while temp < rand_p:
            j += 1
            temp += J_prob_Matrix[j]
        J_select_2 = J_Mat_pop[j][0:J_job_len]
    return J_select_1, J_select_2

def crossover(J_select_1, J_select_2, J_job_len):
    J_child_1 = [lambda x:0 for x in range(J_job_len)]
    temp1 = random.randint(1,J_job_len - 3)
    while True:
        temp2 = random.randint(1,J_job_len - 3)
        if temp1 != temp2:
            break
    rand_pos1 = max(temp1, temp2)
    rand_pos2 = min(temp1, temp2)
    for i in range(rand_pos2):
        J_child_1[i] = J_select_1[i]
    for i in range(J_job_len - 1, rand_pos1 , -1):
        J_child_1[i] = J_select_1[i]
    for i in range(J_job_len):
        if J_select_2[i] not in J_child_1 and rand_pos2 <= J_job_len - 1:
            J_child_1[rand_pos2] = J_select_2[i]
            rand_pos2  += 1
    return J_child_1

def mutation(J_child_1):
    J_job_len = len(J_child_1)
    J_temp_individual = [-1 for i in range(J_job_len)]
    temp1 = random.randint(1,J_job_len - 2)
    while True:
        temp2 = random.randint(1,J_job_len - 2)
        if abs(temp1 - temp2) >= 2:
            break
    J_temp_individual = J_child_1[:]
    rand_pos1 = max(temp1, temp2)
    rand_pos2 = min(temp1, temp2)
    J_child_1[rand_pos2] = J_child_1[rand_pos1]
    for i in range(rand_pos2 + 1, rand_pos1+1):
        J_child_1[i] = J_temp_individual[i - 1]
    return J_child_1

def generate_newpop(num_factory, J_factory_job_set, J_prob_Matrix, J_Matpop):
    J_len_job = [len(J_factory_job_set[i]) for i in range(num_factory)]
    cross_prob = 0.9
    mutation_prob = 0.6
    J_newpop = [[] for i in range(num_factory)]
    for i in range(num_factory):
        for j in range(190):
            #根据crossover概率进行交叉操作
            J_select_1, J_select_2 = select_parents(J_prob_Matrix[i], J_Matpop[i],J_len_job[i])
            temp1 = random.random()
            if temp1 <= cross_prob:
                child = crossover(J_select_1, J_select_2, J_len_job[i])
            else:
                temp2 = random.random()
                if temp2 <= 0.5:
                    child = J_select_1
                else:
                    child = J_select_2
            #根据mutation概率进行变异操作
            temp3 = random.random()
            if temp3 <= mutation_prob:
                child = mutation(child)
            J_newpop[i].append(child)
        for j in range(10):
            index = random.randint(0,199)
            J_newpop[i].append(J_Matpop[i][index][0:J_len_job[i]])
    return J_newpop



def local_search(num_factory, J_newpop, J_factory_job_set, num_machine,v,test_data):
    J_len_job = [len(J_factory_job_set[i]) for i in range(num_factory)]
    local_search_prob = 0.8
    for i in range(num_factory):
        for j in range(200):
            weights = [0.5, 0.5]
            weights[0] = np.random.random()
            weights[1] = 1 - weights[0]
            sort = list(J_newpop[i][j][0:J_len_job[i]])
            C_time, Energy_consumption = TCE(J_len_job[i], num_machine, sort, test_data, v)
            scalar_fitness = weights[0] * C_time + weights[1] * Energy_consumption
            J_newpop[i][j].append(C_time)
            J_newpop[i][j].append(Energy_consumption)
            J_newpop[i][j].append(scalar_fitness)
        J_newpop[i] = sorted(J_newpop[i], key= lambda x:x[J_len_job[i] + 2])
    for i in range(num_factory):
        for j in range(5):
            r = random.random()
            if r <= local_search_prob:
                k_neighbor = []
                for k in range(2):
                    weights = [0.5, 0.5]
                    weights[0] = np.random.random()
                    weights[1] = 1 - weights[0]
                    child = mutation(J_newpop[i][j][0:J_len_job[i]])
                    C_time, Energy_consumption = TCE(J_len_job[i], num_machine, child, test_data, v)
                    scalar_fitness = weights[0] * C_time + weights[1] * Energy_consumption
                    child.append(C_time)
                    child.append(Energy_consumption)
                    child.append(scalar_fitness)
                    k_neighbor.append(child)
                k_neighbor = sorted(k_neighbor, key=lambda x:x[J_len_job[i] + 2])
                if k_neighbor[0][J_len_job[i] + 2] < J_newpop[i][j][J_len_job[i] + 2]:
                    J_newpop[i][j][:] = k_neighbor[0][:]
                else:
                    continue
    return J_newpop

def J_select_non_dominated_pop(num_factory, J_len_job, J_Mat_pop):
    J_temp_non_dominated_pop = [[] for k in range(num_factory)]
    for i in range(num_factory):
        for j in range(len(J_Mat_pop[i])):
            compare_fitness1 = J_Mat_pop[i][j][J_len_job[i]]
            compare_fitness2 = J_Mat_pop[i][j][J_len_job[i]+1]
            b_non_dominated = True
            for k in range(len(J_Mat_pop[i])):
                if j != k:
                    if J_Mat_pop[i][k][J_len_job[i]] <= compare_fitness1 and J_Mat_pop[i][k][J_len_job[i]+1] <= compare_fitness2:
                        if J_Mat_pop[i][k][J_len_job[i]] < compare_fitness1 or J_Mat_pop[i][k][J_len_job[i]+1] < compare_fitness2:
                            b_non_dominated = False
                            break
            if b_non_dominated == True:
                if J_Mat_pop[i][j][0:J_len_job[i]+2] not in J_temp_non_dominated_pop[i]:
                    J_temp_non_dominated_pop[i].append(J_Mat_pop[i][j][0:J_len_job[i]+2])
    return J_temp_non_dominated_pop



def update_non_dominated(J_non_dominated_pop, J_temp_non_dominated,J_factory_job_set,num_factory):
    J_len_job = [len(J_factory_job_set[i]) for i in range(num_factory)]
    for i in range(num_factory):
        for j in range(len(J_temp_non_dominated[i])):
            if J_temp_non_dominated[i][j][0:J_len_job[i]+2] not in J_non_dominated_pop[i]:
                J_non_dominated_pop[i].append(J_temp_non_dominated[i][j][0:J_len_job[i]+2])
    J_non_dominated_pop = J_select_non_dominated_pop(num_factory, J_len_job, J_non_dominated_pop)
    return J_non_dominated_pop

def J_select_all_f_non_dominated_pop(J_temp_all_f_dominated):
    #在所有工厂的帕累托解的组合中找总工厂的帕累托解
    J_temp_non_dominated_pop = []
    len_sol = len(J_temp_all_f_dominated)
    for i in range(len_sol):
        compare_fitness1 = J_temp_all_f_dominated[i][-2]
        compare_fitness2 = J_temp_all_f_dominated[i][-1]
        b_non_dominated = True
        for j in range(len_sol):
            if i != j:
                if J_temp_all_f_dominated[j][-2] <= compare_fitness1 and J_temp_all_f_dominated[j][-1] <= compare_fitness2:
                    if J_temp_all_f_dominated[j][-2] < compare_fitness1 or J_temp_all_f_dominated[j][-1] < compare_fitness2:
                        b_non_dominated = False
                        break
        if b_non_dominated == True:
            if J_temp_all_f_dominated[i] not in J_temp_non_dominated_pop:
                J_temp_non_dominated_pop.append(J_temp_all_f_dominated[i])
    return J_temp_non_dominated_pop



def Japan_Multi_object(pop_gen,test_time,v,num_job, num_machine, test_data, num_factory):
    test_timeup = time.clock()
    J_factory_job_set = NEH2(num_job, num_machine, test_data, num_factory,v)
    global J_len_job
    J_len_job = [len(J_factory_job_set[i]) for i in range(num_factory)]
    J_Mat_pop, J_non_dominated_pop = Multi_initial_Japan(num_machine, num_factory, J_factory_job_set, test_data,v)
    J_prob_Matrix = selection_prob(J_Mat_pop, num_factory, J_factory_job_set)
    for gen in range(pop_gen):
        J_Mat_pop =  generate_newpop(num_factory, J_factory_job_set, J_prob_Matrix, J_Mat_pop)
        J_Mat_pop = local_search(num_factory, J_Mat_pop, J_factory_job_set, num_machine,v,test_data)
        J_temp_non_dominated = J_select_non_dominated_pop(num_factory, J_len_job, J_Mat_pop)
        J_non_dominated_pop = update_non_dominated(J_non_dominated_pop, J_temp_non_dominated,J_factory_job_set,num_factory)
        test_timedown = time.clock()
        if float(test_timedown-test_timeup)>=float(test_time):
            break
    return J_non_dominated_pop, float(test_timedown-test_timeup)

def J_All_factory_dominated(num_factory,pop_gen,test_time,v,num_job, num_machine, test_data):
    #根据每个工厂的帕累托解确定总工厂的解
    J_non_dominated_pop, run_time = Japan_Multi_object(pop_gen, test_time,v,num_job, num_machine, test_data, num_factory)
    print('日本人程序共运行：%s' % (run_time))
    J_temp_all_f_dominated = []
    result = []
    temp_set = list(itertools.product(J_non_dominated_pop[0],J_non_dominated_pop[1]))#lambda x: list(x) for x in J_non_dominated_pop[0])
    for individual in temp_set:
        individual = sorted(individual,key=lambda x:x[-2])
        parteo_solution = [0 for i in range(len(individual[-1]))]
        sum_green_fitness = 0
        for indi_green in individual:
            sum_green_fitness += indi_green[-1]
        for i in range(len(individual[-1])):
            parteo_solution[i] = individual[-1][i]
        parteo_solution[-1] = sum_green_fitness
        J_temp_all_f_dominated.append(parteo_solution)
    result = J_select_all_f_non_dominated_pop(J_temp_all_f_dominated)
    return result

#J_All_factory_dominated(num_factory,pop_gen)




#J_non_dominated_pop = Japan_Multi_object(pop_gen)
#print(J_non_dominated_pop[0])








