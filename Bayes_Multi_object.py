'''
Green_scheduling

2017.08.07
Author: Yang Xiaolin
'''
from AssignRule import *
from random import choice
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import itertools
def speed_Matrix():
    V = [1, 1.3, 1.55, 1.75, 2.1]
    v = np.zeros((num_job, num_machine))
    for i in range(num_machine):
        for j in range(num_job):
            temp = choice(V)
            v[j][i] = temp
    return v

v = speed_Matrix()

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

def TCE(n, m, sort, test_data,v):
    per_consumption_Standy = 1
    standy_time = 0
    Energy_consumption = 0
    for k in range(m):
        for i in range(n):
            per_consumption_V = 4 * v[i][k] * v[i][k] # 机器加速单位时间能源消耗
            Energy_consumption += test_data[sort[i]][k] / v[i][k] * per_consumption_V
    C_time = Green_Calcfitness(n,m,sort, test_data, v)
    for k in range(m):
        Key_time = C_time
        for i in range(n):
            Key_time -= test_data[sort[i]][k] / v[i][k]
        standy_time += Key_time
    Energy_consumption += standy_time * per_consumption_Standy
    return C_time, Energy_consumption


def green_initial_Bayes(num_machine, num_factory, factory_job_set, test_data, len_job, initial_popsize, update_popsize):
    #每个工厂的工件数
   # len_job = [len(factory_job_set[i]) for i in range(num_factory)]
    #每个工厂的初始数据
    Mat_pop =[[[0 for i in range(len_job[k] + 2) ]for j in range(initial_popsize)] for k in range(num_factory)] #最后两个元素分别是经济指标和绿色指标
    non_dominated_pop = [[] for k in range(num_factory)]
    for i in range(num_factory):
        for j in range(initial_popsize):
            sort = random.sample(factory_job_set[i], len_job[i])
            for k in range(len_job[i] + 1):
                if k == len_job[i]:
                    Mat_pop[i][j][k], Mat_pop[i][j][k+1]= TCE(len_job[i], num_machine, sort, test_data,v)
                else:
                    Mat_pop[i][j][k] = sort[k]
        Mat_pop[i] = sorted(Mat_pop[i], key= lambda x:x[-2]+x[-1])
        Mat_pop[i] = Mat_pop[i][0:update_popsize]
        for j in range(update_popsize):
            compare_fitness1 = Mat_pop[i][j][len_job[i]]
            compare_fitness2 = Mat_pop[i][j][len_job[i]+1]
            b_non_dominated = True
            for k in range(update_popsize):
                if j != k:
                    if Mat_pop[i][k][len_job[i]] <= compare_fitness1 and Mat_pop[i][k][len_job[i]+1] <= compare_fitness2:
                        if Mat_pop[i][k][len_job[i]] < compare_fitness1 or Mat_pop[i][k][len_job[i]+1] < compare_fitness2:
                            b_non_dominated = False
                            break
            if b_non_dominated == True:
                if Mat_pop[i][j] not in non_dominated_pop[i]:
                    non_dominated_pop[i].append(Mat_pop[i][j])

    return Mat_pop[:][0:update_popsize], non_dominated_pop

def select_non_dominated_pop(num_factory, len_job, Mat_pop):
    temp_non_dominated_pop = [[] for k in range(num_factory)]
    for i in range(num_factory):
        for j in range(len(Mat_pop[i])):
            compare_fitness1 = Mat_pop[i][j][len_job[i]]
            compare_fitness2 = Mat_pop[i][j][len_job[i]+1]
            b_non_dominated = True
            for k in range(len(Mat_pop[i])):
                if j != k:
                    if Mat_pop[i][k][len_job[i]] <= compare_fitness1 and Mat_pop[i][k][len_job[i]+1] <= compare_fitness2:
                        if Mat_pop[i][k][len_job[i]] < compare_fitness1 or Mat_pop[i][k][len_job[i]+1] < compare_fitness2:
                            b_non_dominated = False
                            break
            if b_non_dominated == True:
                if Mat_pop[i][j][0:len_job[i] + 2] not in temp_non_dominated_pop[i]:
                    temp_non_dominated_pop[i].append(Mat_pop[i][j][0:len_job[i] + 2])
    return temp_non_dominated_pop

def select_all_f_non_dominated_pop(temp_all_f_dominated):
    #在所有工厂的帕累托解的组合中找总工厂的帕累托解
    temp_non_dominated_pop = []
    len_sol = len(temp_all_f_dominated)
    for i in range(len_sol):
        compare_fitness1 = temp_all_f_dominated[i][-2]
        compare_fitness2 = temp_all_f_dominated[i][-1]
        b_non_dominated = True
        for j in range(len_sol):
            if i != j:
                if temp_all_f_dominated[j][-2] <= compare_fitness1 and temp_all_f_dominated[j][-1] <= compare_fitness2:
                    if temp_all_f_dominated[j][-2] < compare_fitness1 or temp_all_f_dominated[j][-1] < compare_fitness2:
                        b_non_dominated = False
                        break
        if b_non_dominated == True:
            if temp_all_f_dominated[i][:] not in temp_non_dominated_pop:
                temp_non_dominated_pop.append(temp_all_f_dominated[i])
    return temp_non_dominated_pop


def Green_New_pop(prob_mat_first, num_factory, factory_job_set, Mat_pop, len_job):
    # 每个工厂每次更新20个个体
    newpop = [[[-1 for i in range(len_job[k] + 2)] for j in range(local_search_size)] for k in range(num_factory)]
    # 三维数组--第一维：工厂数；第二维：每个工厂的所有相邻关系数组；第三维：与上一个工件的关系
    prob_mat = [[[0.0 for i in range(len_job[k])]
                 for l in range(len_job[k] - 1)] for k in range(num_factory)]
    for i in range(num_factory):
        for num in range(local_search_size):
            temp = 0.0
            # 用二分法轮盘赌确定第一个工件
            r = random.random()
            dichotomy = Roulette_prob(prob_mat_first[i], len_job[i])
            begin = 0
            end = len_job[i] - 1
            j = Roulette_dichotomy(r, dichotomy, begin, end)
            newpop[i][num][0] = factory_job_set[i][j]
            for k in range(1, len_job[i]):
                temp_job = []
                for ii in Mat_pop[i][0:200]:
                    if ii[k] in newpop[i][num]:
                        continue
                    elif ii[k - 1] == newpop[i][num][k - 1]:
                        temp_job.append(ii[k])
                if len(temp_job) == 0:
                    key = True
                    while key:
                        newpop_temp = choice(factory_job_set[i])
                        if newpop_temp not in newpop[i][num]:
                            newpop[i][num][k] = newpop_temp
                            key = False
                    continue
                for m in range(len_job[i]):
                    prob_mat[i][k - 1][m] = temp_job.count(factory_job_set[i][m]) / len(temp_job)
                r = random.random()
                dichotomy = Roulette_prob(prob_mat[i][k - 1], len_job[i])
                begin = 0
                end = len_job[i] - 1
                j = Roulette_dichotomy(r, dichotomy, begin, end)
                newpop[i][num][k] = factory_job_set[i][j]
            newpop[i][num][-2], newpop[i][num][-1] = TCE(len_job[i], num_machine, newpop[i][num], test_data, v)
        newpop[i] = sorted(newpop[i], key= lambda x:x[-2])
    return newpop[:][0:local_search_size]

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def swap_search(newpop, ls_frequency, len_job, num_factory,local_search_size):
    ls_pop = [[[[-1 for i in range(len_job[k] + 2)] for j in range(ls_frequency)] for l in range(local_search_size)]for k in range(num_factory)]
    select_ls_pop = [[[-1 for i in range(len_job[k] + 2)]  for l in range(local_search_size)]for k in range(num_factory)]
    for i in range(num_factory):
        for l in range(local_search_size):
            ls_pop[i][l][0][:] = newpop[i][l][:]        #保优操作
            for j in range(1, ls_frequency):
                temp1 = np.random.randint(0, len_job[i])
                temp2 = np.random.randint(0, len_job[i])
                while temp1 == temp2:
                    temp1 = np.random.randint(0, len_job[i])
                    temp2 = np.random.randint(0, len_job[i])
                temp = newpop[i][l][temp1]
                newpop[i][l][temp1] = newpop[i][l][temp2]
                newpop[i][l][temp2] = temp
                ls_pop[i][l][j][:len_job[i]] = newpop[i][l][:len_job[i]]
                newpop[i][l][:len_job[i]] = ls_pop[i][l][0][:len_job[i]]
                ls_pop[i][l][j][len_job[i]],ls_pop[i][l][j][len_job[i] + 1] = TCE(len_job[i], num_machine, ls_pop[i][l][j][0:len_job[i]], test_data, v)
            select_ls_pop[i][l] = sorted(ls_pop[i][l],key= lambda x:x[len_job[i]])[0]
        select_ls_pop[i] = sorted(select_ls_pop[i],key= lambda x:x[len_job[i]])
    return select_ls_pop

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
def insert_search(newpop, ls_frequency, len_job, num_factory, local_search_size):
    ls_pop = [[[[-1 for i in range(len_job[k] + 2)] for j in range(ls_frequency)] for l in range(local_search_size)]for k in range(num_factory)]
    select_ls_pop = [[[-1 for i in range(len_job[k] + 2)]  for l in range(local_search_size)]for k in range(num_factory)]
    for i in range(num_factory):
        temp_individual = [-1 for i in range(len_job[i])]
        for l in range(local_search_size):
            ls_pop[i][l][0][:] = newpop[i][l][:]        #保优操作
            for j in range(1, ls_frequency):
                temp1 = random.randint(1, len_job[i] - 2)
                while True:
                    temp2 = random.randint(1, len_job[i] - 2)
                    if abs(temp1 - temp2) >= 2:
                        break
                rand_pos1 = max(temp1, temp2)
                rand_pos2 = min(temp1, temp2)
                temp_individual[:] = newpop[i][l][:]
                temp_individual[rand_pos2] = temp_individual[rand_pos1]
                for k in range(rand_pos2 + 1, rand_pos1+1):
                    temp_individual[k] = newpop[i][l][k-1]
                ls_pop[i][l][j][0:len_job[i]] = temp_individual[:]
                ls_pop[i][l][j][len_job[i]], ls_pop[i][l][j][len_job[i] + 1] = TCE(len_job[i], num_machine, ls_pop[i][l][j][0:len_job[i]], test_data, v)
            select_ls_pop[i][l] = sorted(ls_pop[i][l], key=lambda x: x[len_job[i]])[0]
        select_ls_pop[i] = sorted(select_ls_pop[i], key=lambda x: x[len_job[i]])
    return select_ls_pop
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
def interchange(newpop, ls_frequency, len_job, num_factory, local_search_size):
    ls_pop = [[[[-1 for i in range(len_job[k] + 2)] for j in range(ls_frequency)] for l in range(local_search_size)]for k in range(num_factory)]
    select_ls_pop = [[[-1 for i in range(len_job[k] + 2)]  for l in range(local_search_size)]for k in range(num_factory)]
    for i in range(num_factory):
        temp_individual = [-1 for i in range(len_job[i])]
        for l in range(local_search_size):
            ls_pop[i][l][0][:] = newpop[i][l][:]        #保优操作
            for j in range(1, ls_frequency):
                temp = random.randint(0, len_job[i] - 1)
                if temp == 0:
                    temp_individual[:] = newpop[i][l][:]
                    temp_job = temp_individual[temp]
                    temp_individual[temp] = temp_individual[temp + 1]
                    temp_individual[temp + 1] = temp_job
                elif temp == len_job[i] - 1:
                    temp_individual[:] = newpop[i][l][:]
                    temp_job = temp_individual[temp]
                    temp_individual[temp] = temp_individual[temp - 1]
                    temp_individual[temp - 1] = temp_job
                else:
                    r = random.random()
                    if r>0.5:
                        temp_individual[:] = newpop[i][l][:]
                        temp_job = temp_individual[temp]
                        temp_individual[temp] = temp_individual[temp + 1]
                        temp_individual[temp + 1] = temp_job
                    else:
                        temp_individual[:] = newpop[i][l][:]
                        temp_job = temp_individual[temp]
                        temp_individual[temp] = temp_individual[temp - 1]
                        temp_individual[temp - 1] = temp_job
                ls_pop[i][l][j][0:len_job[i]] = temp_individual[:]
                ls_pop[i][l][j][len_job[i]], ls_pop[i][l][j][len_job[i] + 1] = TCE(len_job[i], num_machine, ls_pop[i][l][j][0:len_job[i]], test_data, v)
            select_ls_pop[i][l] = sorted(ls_pop[i][l], key=lambda x: x[len_job[i]])[0]
        select_ls_pop[i] = sorted(select_ls_pop[i], key=lambda x: x[len_job[i]])
    return select_ls_pop
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
def block_3dim(Mat_pop, len_job, update_popsize):
    block = [[] for i in range(num_factory)]
    select_block = [[] for i in range(num_factory)]
    select_location = [[] for i in range(num_factory)]
    for i in range(num_factory):
        block_Matrix = np.zeros((len_job[i] - 2, num_job, num_job, num_job)) #带位置信息的三维block
        for k in range(int(update_popsize/5)):
            for j in range(0,len_job[i] - 2):
                block_Matrix[j, Mat_pop[i][k][j],Mat_pop[i][k][j+1],Mat_pop[i][k][j+2]] += 1
        location, height, raw, column = block_Matrix.shape
        key = True
        while key:
            _positon = np.argmax(block_Matrix)
            loc = int(_positon /height/ raw / column)
            h = int((_positon-(height*raw*column)*loc) / raw / column)
            m, n = divmod((_positon-(height*raw*column)*loc) - (raw * column) * h, column)
            if block_Matrix[loc,h, int(m), int(n)] >= int(update_popsize/5*0.3):
                block[i].append([loc, h, int(m), int(n)])
                #block[i].append(block_Matrix[loc, h, int(m), int(n)])
                block_Matrix[loc, h, int(m), int(n)] = 0
            else:
                key = False
    #---------------------------------------------------------------------------------------------

    for i in range(num_factory):    #组合概率高的block 剔除存在冲突的block
        select_job = set()
        if len(block[i]) == 0:
            continue
        j = 0
        if j == 0:
            for k in range(1,4):
                select_job.add(block[i][0][k])
            select_block[i].append(block[i][0])     #添加第一个块结构
            select_location[i].append(block[i][0][0])       #记录第一个块结构的位置
        j += 1
        len_block = len(block[i])
        while j<len_block:
            #print(block[i][j])
            temp_job = set()
            for k in range(1, 4):
                temp_job.add(block[i][j][k])
            if block[i][j][0] in select_location[i]:
                j += 1
                continue
            elif len(select_job & temp_job) == 0:
                select_block[i].append(block[i][j])
                select_location[i].append(block[i][j][0])
                j += 1
                continue
            else:
                j += 1
                continue
            if block[i][j][1] == select_block[i][j-1][-2] and block[i][j][2] == select_block[i][j-1][-1]:
                if block[i][j][-1] in select_block[i][j-1]:
                    j += 1
                    continue
                else:
                    select_block[i].append(block[i][j])
                    select_location[i].append(block[i][j][0])
                    j += 1
                    continue
            elif block[i][j][1] == select_block[i][j-1][-1]:
                if block[i][j][2] or block[i][j][3] in  select_block[i][j-1]:
                    j += 1
                    continue
                else:
                    select_block[i].append(block[i][j])
                    select_location[i].append(block[i][j][0])
            j += 1
    return select_block

def block_based(block,  Mat_pop, factory_job_set, len_job, update_popsize):
    #根据精英集合构建的快结构生成新种群
    factory_job_set_other = [[] for i in range(num_factory)]
    len_job_other = [0 for i in range(num_factory)]
    block_Mat_pop = [[[-1 for i in range(len_job[k] + 2)] for j in range(update_popsize)] for k in range(num_factory)]
    for i in range(num_factory):
        for j in range(len(block[i])):
            location = block[i][j][0]
            for k in range(1,len(block[i][j])):
                for l in range(update_popsize):
                    block_Mat_pop[i][l][location] = block[i][j][k]
                location += 1
    for i in range(num_factory):
        for j in range(len_job[i]):
            if factory_job_set[i][j] not in block_Mat_pop[i][0]:
                factory_job_set_other[i].append(factory_job_set[i][j])
            len_job_other[i] = len(factory_job_set_other[i])
        for j in range(update_popsize):
            sort = random.sample(factory_job_set_other[i], len_job_other[i])
            index = 0
            for k in range(len_job[i]):
                if block_Mat_pop[i][j][k] == -1:
                    block_Mat_pop[i][j][k] = sort[index]
                    index += 1
                if index == len_job_other[i]:
                    break
    for i in range(num_factory):
        for j in range(update_popsize):
            block_Mat_pop[i][j][len_job[i]], block_Mat_pop[i][j][len_job[i]+1] = TCE(len_job[i], num_machine, block_Mat_pop[i][j][0:len_job[i]],
                                                                                     test_data, v)
        block_Mat_pop[i] = sorted(block_Mat_pop[i], key=lambda x: x[-2])
    return block_Mat_pop

def Green_Bayes_net(pop_gen, ls_frequency, update_popsize):
    global len_job
    factory_job_set = NEH2(num_job, num_machine, test_data, num_factory)
    len_job = [len(factory_job_set[i]) for i in range(num_factory)]
    Mat_pop,  non_dominated_pop= green_initial_Bayes(num_machine, num_factory, factory_job_set, test_data, len_job,initial_popsize, update_popsize)
    temp_list = []
    for gen in range(pop_gen):
        prob_mat_first = Bayes_update(Mat_pop, factory_job_set, num_factory, len_job, update_popsize)
        newpop = Green_New_pop(prob_mat_first, num_factory, factory_job_set, Mat_pop, len_job)
        r = random.random()
        ls_pop = swap_search(newpop, ls_frequency, len_job, num_factory,local_search_size)
        for i in range(num_factory):
            k_index = -1
            for k in range(local_search_size - 1,-1,-1):
                k_index += 1
                for j in range(update_popsize - 1 - k_index,-1,-1):
                    if float(ls_pop[i][k][-2]) < float(Mat_pop[i][j][len_job[i]]) or float(ls_pop[i][k][-1]) < float(Mat_pop[i][j][len_job[i]+1]):
                        temp_list = ls_pop[i][k]
                        for l in range(len(factory_job_set[i]) + 2):
                            Mat_pop[i][j][l] = temp_list[l]
                        temp_list = []
                        break
            Mat_pop[i] = sorted(Mat_pop[i], key= lambda x:x[-2])
        temp_non_dominated = select_non_dominated_pop(num_factory, len_job, Mat_pop)
        for i in range(num_factory):
            for j in range(len(temp_non_dominated[i])):
                if temp_non_dominated[i][j] not in non_dominated_pop[i]:
                    non_dominated_pop[i].append(temp_non_dominated[i][j])
        non_dominated_pop = select_non_dominated_pop(num_factory, len_job, non_dominated_pop)
    demo = block_3dim(Mat_pop, len_job, update_popsize)
    demo1 = block_based(demo, Mat_pop, factory_job_set, len_job, update_popsize)
    for gen in range(50):
        ls_pop = insert_search(demo1, ls_frequency, len_job, num_factory, local_search_size)
        for i in range(num_factory):
            k_index = -1
            for k in range(local_search_size - 1, -1, -1):
                k_index += 1
                for j in range(update_popsize - 1 - k_index, -1, -1):
                    if float(ls_pop[i][k][-2]) < float(demo1[i][j][len_job[i]]) or float(ls_pop[i][k][-1]) < float(
                            demo1[i][j][len_job[i] + 1]):
                        temp_list = ls_pop[i][k]
                        for l in range(len(factory_job_set[i]) + 2):
                            demo1[i][j][l] = temp_list[l]
                        temp_list = []
                        break
            demo1[i] = sorted(demo1[i], key=lambda x: x[-2])
        temp_non_dominated = select_non_dominated_pop(num_factory, len_job, demo1)
        for i in range(num_factory):
            for j in range(len(temp_non_dominated[i])):
                if temp_non_dominated[i][j] not in non_dominated_pop[i]:
                    non_dominated_pop[i].append(temp_non_dominated[i][j])
        non_dominated_pop = select_non_dominated_pop(num_factory, len_job, non_dominated_pop)
    return non_dominated_pop

def All_factory_dominated(non_dominated_pop, num_factory):
    #根据每个工厂的帕累托解确定总工厂的解
    temp_all_f_dominated = []
    result = []
    temp_set = list(itertools.product(non_dominated_pop[0],non_dominated_pop[1]))#lambda x: list(x) for x in non_dominated_pop[0])
    for individual in temp_set:
        individual = sorted(individual,key=lambda x:x[-2])
        sum_green_fitness = 0
        for indi_green in individual:
            sum_green_fitness += indi_green[-1]
        individual[-1][-1] = sum_green_fitness
        temp_all_f_dominated.append(individual[-1])
    result = select_all_f_non_dominated_pop(temp_all_f_dominated)
    print(result)


non_dominated_pop = Green_Bayes_net(pop_gen, ls_frequency,update_popsize)
#print(non_dominated_pop[0])
All_factory_dominated(non_dominated_pop, num_factory)
#print(non_dominated_pop[0])