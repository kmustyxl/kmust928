from AssignRule import *
from Bayes_Multi_object import *

factory_job_set =  NEH2(num_job, num_machine, test_data, num_factory)

def Multi_initial_Japan(num_machine, num_factory, factory_job_set, test_data):
    #每个工厂的工件数
    len_job = [len(factory_job_set[i]) for i in range(num_factory)]
    #每个工厂的初始数据
    Mat_pop =[[[0 for i in range(len_job[k] + 2) ]for j in range(200)] for k in range(num_factory)] #最后两个元素分别是经济指标和绿色指标
    non_dominated_pop = [[] for k in range(num_factory)]
    for i in range(num_factory):
        for j in range(200):
            sort = random.sample(factory_job_set[i], len_job[i])
            for k in range(len_job[i] + 1):
                if k == len_job[i]:
                    Mat_pop[i][j][k], Mat_pop[i][j][k+1]= TCE(len_job[i], num_machine, sort, test_data,v)
                else:
                    Mat_pop[i][j][k] = sort[k]
        Mat_pop[i] = sorted(Mat_pop[i], key= lambda x:x[-2])
        for j in range(200):
            compare_fitness1 = Mat_pop[i][j][len_job[i]]
            compare_fitness2 = Mat_pop[i][j][len_job[i]+1]
            b_non_dominated = True
            for k in range(200):
                if j != k:
                    if Mat_pop[i][k][len_job[i]] <= compare_fitness1 and Mat_pop[i][k][len_job[i]+1] <= compare_fitness2:
                        if Mat_pop[i][k][len_job[i]] < compare_fitness1 or Mat_pop[i][k][len_job[i]+1] < compare_fitness2:
                            b_non_dominated = False
                            break
            if b_non_dominated == True:
                if Mat_pop[i][j] not in non_dominated_pop[i]:
                    non_dominated_pop[i].append(Mat_pop[i][j])

    return Mat_pop, non_dominated_pop

def selection_prob(Mat_pop, num_factory, factory_job_set):
    weights = [0.5, 0.5]
    weights[0] = np.random.random()
    weights[1] = 1 - weights[0]
    prob_Matrix = [[-1 for i in range(200)] for j in range(num_factory)]
    len_job = [len(factory_job_set[i]) for i in range(num_factory)]
    for i in range(num_factory):
        for individual in Mat_pop[i]:
            scalar_fitness = weights[0] * individual[len_job[i]] + weights[1] * individual[len_job[i] + 1]
            individual.append(scalar_fitness)
        Mat_pop[i] = sorted(Mat_pop[i], key=lambda x: x[len_job[i] + 2])
    for i in range(num_factory):
        f_max = Mat_pop[i][-1][len_job[i] + 2]
        Ps_x_Denominator = 0
        for individual in Mat_pop[i]:               #求出Step2中概率方程的分母
            Ps_x_Denominator += float(f_max - individual[len_job[i] + 2])
        for j in range(200):
            prob_Matrix[i][j] = float(f_max - Mat_pop[i][j][len_job[i] + 2]) / Ps_x_Denominator
    return prob_Matrix

def select_parents(prob_Matrix, Mat_pop, job_len):
    #采用轮盘赌选取一对父代个体
    temp = 0.0
    rand_p = random.random()
    if rand_p < prob_Matrix[0]:
        select_1 = Mat_pop[0][0:job_len]
    else:
        j = 0
        temp += prob_Matrix[0]
        while temp < rand_p:
            j += 1
            temp += prob_Matrix[j]
        select_1 = Mat_pop[j][0:job_len]
    temp = 0.0
    rand_p = random.random()
    if rand_p < prob_Matrix[0]:
        select_2 = Mat_pop[0][0:job_len]
    else:
        j = 0
        temp += prob_Matrix[0]
        while temp < rand_p:
            j += 1
            temp += prob_Matrix[j]
        select_2 = Mat_pop[j][0:job_len]
    return select_1, select_2

def crossover(select_1, select_2, job_len):
    child_1 = [lambda x:0 for x in range(job_len)]
    temp1 = random.randint(1,job_len - 3)
    while True:
        temp2 = random.randint(1,job_len - 3)
        if temp1 != temp2:
            break
    rand_pos1 = max(temp1, temp2)
    rand_pos2 = min(temp1, temp2)
    for i in range(rand_pos2):
        child_1[i] = select_1[i]
    for i in range(job_len - 1, rand_pos1 , -1):
        child_1[i] = select_1[i]
    for i in range(job_len):
        if select_2[i] not in child_1 and rand_pos2 <= job_len - 1:
            child_1[rand_pos2] = select_2[i]
            rand_pos2  += 1
    return child_1

def mutation(child_1):
    job_len = len(child_1)
    temp_individual = [-1 for i in range(job_len)]
    temp1 = random.randint(1,job_len - 2)
    while True:
        temp2 = random.randint(1,job_len - 2)
        if abs(temp1 - temp2) >= 2:
            break
    temp_individual = child_1[:]
    rand_pos1 = max(temp1, temp2)
    rand_pos2 = min(temp1, temp2)
    child_1[rand_pos2] = child_1[rand_pos1]
    for i in range(rand_pos2 + 1, rand_pos1+1):
        child_1[i] = temp_individual[i - 1]
    return child_1

def generate_newpop(num_factory, factory_job_set, prob_Matrix, Matpop):
    len_job = [len(factory_job_set[i]) for i in range(num_factory)]
    cross_prob = 0.9
    mutation_prob = 0.6
    newpop = [[] for i in range(num_factory)]
    for i in range(num_factory):
        for j in range(190):
            #根据crossover概率进行交叉操作
            select_1, select_2 = select_parents(prob_Matrix[i], Matpop[i],len_job[i])
            temp1 = random.random()
            if temp1 <= cross_prob:
                child = crossover(select_1, select_2, len_job[i])
            else:
                temp2 = random.random()
                if temp2 <= 0.5:
                    child = select_1
                else:
                    child = select_2
            #根据mutation概率进行变异操作
            temp3 = random.random()
            if temp3 <= mutation_prob:
                child = mutation(child)
            newpop[i].append(child)
        for j in range(10):
            index = random.randint(0,199)
            newpop[i].append(Matpop[i][index][0:len_job[i]])
    return newpop



def local_search(num_factory, newpop, factory_job_set, num_machine):
    len_job = [len(factory_job_set[i]) for i in range(num_factory)]
    local_search_prob = 0.8
    for i in range(num_factory):
        for j in range(200):
            weights = [0.5, 0.5]
            weights[0] = np.random.random()
            weights[1] = 1 - weights[0]
            sort = list(newpop[i][j][0:len_job[i]])
            C_time, Energy_consumption = TCE(len_job[i], num_machine, sort, test_data, v)
            scalar_fitness = weights[0] * C_time + weights[1] * Energy_consumption
            newpop[i][j].append(C_time)
            newpop[i][j].append(Energy_consumption)
            newpop[i][j].append(scalar_fitness)
        newpop[i] = sorted(newpop[i], key= lambda x:x[len_job[i] + 2])
    for i in range(num_factory):
        for j in range(5):
            r = random.random()
            if r <= local_search_prob:
                k_neighbor = []
                for k in range(2):
                    weights = [0.5, 0.5]
                    weights[0] = np.random.random()
                    weights[1] = 1 - weights[0]
                    child = mutation(newpop[i][j][0:len_job[i]])
                    C_time, Energy_consumption = TCE(len_job[i], num_machine, child, test_data, v)
                    scalar_fitness = weights[0] * C_time + weights[1] * Energy_consumption
                    child.append(C_time)
                    child.append(Energy_consumption)
                    child.append(scalar_fitness)
                    k_neighbor.append(child)
                k_neighbor = sorted(k_neighbor, key=lambda x:x[len_job[i] + 2])
                if k_neighbor[0][len_job[i] + 2] < newpop[i][j][len_job[i] + 2]:
                    newpop[i][j][:] = k_neighbor[0][:]
                else:
                    continue
    return newpop
def update_non_dominated(non_dominated_pop, temp_non_dominated,factory_job_set):
    job_len = [len(factory_job_set[i]) for i in range(num_factory)]
    for i in range(num_factory):
        for j in range(len(temp_non_dominated[i])):
            if temp_non_dominated[i][j][0: job_len[i]+2] not in non_dominated_pop[i]:
                non_dominated_pop[i].append(temp_non_dominated[i][j][0:job_len[i] + 2])
    non_dominated_pop = select_non_dominated_pop(num_factory, len_job, non_dominated_pop)
    return non_dominated_pop

def Japan_Multi_object(pop_gen):
    factory_job_set = NEH2(num_job, num_machine, test_data, num_factory)
    global len_job
    len_job = [len(factory_job_set[i]) for i in range(num_factory)]
    Mat_pop, non_dominated_pop = Multi_initial_Japan(num_machine, num_factory, factory_job_set, test_data)
    prob_Matrix = selection_prob(Mat_pop, num_factory, factory_job_set)
    for gen in range(pop_gen):
        Mat_pop =  generate_newpop(num_factory, factory_job_set, prob_Matrix, Mat_pop)
        Mat_pop = local_search(num_factory, Mat_pop, factory_job_set, num_machine)
        temp_non_dominated = select_non_dominated_pop(num_factory, len_job, Mat_pop)
        non_dominated_pop = update_non_dominated(non_dominated_pop, temp_non_dominated,factory_job_set)
    return non_dominated_pop

#non_dominated_pop = Japan_Multi_object(pop_gen)
#print(non_dominated_pop[0])











