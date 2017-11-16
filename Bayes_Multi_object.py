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
import math
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
    return int(C_time), int(Energy_consumption)


def green_initial_Bayes(num_machine, num_factory, factory_job_set, test_data, len_job, update_popsize,v):
    Mat_pop =[[[0 for i in range(len_job[k] + 2) ]for j in range(update_popsize)] for k in range(num_factory)] #最后两个元素分别是经济指标和绿色指标
    non_dominated_pop = [[] for k in range(num_factory)]
    for i in range(num_factory):
        for j in range(update_popsize):
            sort = random.sample(factory_job_set[i], len_job[i])
            for k in range(len_job[i] + 1):
                if k == len_job[i]:
                    Mat_pop[i][j][k], Mat_pop[i][j][k+1]= TCE(len_job[i], num_machine, sort, test_data,v)
                else:
                    Mat_pop[i][j][k] = sort[k]
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
                if Mat_pop[i][j][0:len_job[i]+2] not in non_dominated_pop[i]:
                    non_dominated_pop[i].append(Mat_pop[i][j][0:len_job[i]+2])

    return Mat_pop, non_dominated_pop

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

def B_update_non_dominated(B_non_dominated_pop, B_temp_non_dominated,factory_job_set,num_factory):
    B_len_job = [len(factory_job_set[i]) for i in range(num_factory)]
    for i in range(num_factory):
        for j in range(len(B_temp_non_dominated[i])):
            if B_temp_non_dominated[i][j][0:B_len_job[i]+2] not in B_non_dominated_pop[i]:
                B_non_dominated_pop[i].append(B_temp_non_dominated[i][j][0:B_len_job[i]+2])
    B_non_dominated_pop = select_non_dominated_pop(num_factory, B_len_job, B_non_dominated_pop)
    return B_non_dominated_pop

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
            if temp_all_f_dominated[i] not in temp_non_dominated_pop:
                temp_non_dominated_pop.append(temp_all_f_dominated[i])
    return temp_non_dominated_pop

def Bayes_update(Mat_pop, factory_job_set, num_factory, len_job, update_popsize, non_dominated_pop):
    prob_mat_first = [[0 for i in range(len_job[k])]for k in range(num_factory)]
    #返回每个工厂相邻两个工件的所有情况的概率分布
    for i in range(num_factory):
        #确定数据中第一个工件的出现概率
        index = 0
        demo = [ii[0] for ii in non_dominated_pop[i]]
        for job in factory_job_set[i]:
            prob_mat_first[i][index] += demo.count(job) / len(non_dominated_pop[i])
            index += 1
        #每个工厂内分别有对应的（job_len - 1）个关系数组
       # zongshu = sum(prob_mat_first[i])
       # for j in range(len_job[i]):
        #    prob_mat_first[i][j] = prob_mat_first[i][j]/zongshu  #归一化
    return prob_mat_first

def Green_New_pop(prob_mat_first, num_factory, factory_job_set, Mat_pop, len_job,v,update_popsize,local_search_size,num_machine, test_data, non_dominated_pop):
    # 每个工厂每次更新20个个体
    newpop = [[[-1 for i in range(len_job[k] + 2)] for j in range(update_popsize)] for k in range(num_factory)]
    # 三维数组--第一维：工厂数；第二维：每个工厂的所有相邻关系数组；第三维：与上一个工件的关系
    prob_mat = [[[1/len_job[k] for i in range(len_job[k])]
                 for l in range(len_job[k] - 1)] for k in range(num_factory)]
    for i in range(num_factory):
        for num in range(update_popsize):
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
                for ii in non_dominated_pop[i]:
                    if ii[k] in newpop[i][num]:
                        continue
                    elif ii[k - 1] == newpop[i][num][k - 1]:
                        temp_job.append(ii[k])
                if len(temp_job) == 0:
                    shengyu_gongjian = list(set(factory_job_set[i]).difference(set(newpop[i][num])))
                    #key = True
                   # while key:
                    newpop_temp = choice(shengyu_gongjian)
                     #   if newpop_temp not in newpop[i][num]:
                    newpop[i][num][k] = newpop_temp
                            #key = False
                   # continue
                else:
                    for m in range(len_job[i]):
                        if factory_job_set[i][m] in newpop[i][num]:
                            prob_mat[i][k - 1][m] = 0
                        else:
                            prob_mat[i][k - 1][m] += temp_job.count(factory_job_set[i][m]) / len(temp_job)
                    zongshu = sum(prob_mat[i][k - 1])
                    for m in range(len_job[i]):
                        prob_mat[i][k - 1][m] = prob_mat[i][k - 1][m]/zongshu
                    B_index = True
                    while B_index:
                        r = random.random()
                        dichotomy = Roulette_prob(prob_mat[i][k - 1], len_job[i])
                        begin = 0
                        end = len_job[i] - 1
                        j = Roulette_dichotomy(r, dichotomy, begin, end)
                        if factory_job_set[i][j] in newpop[i][num]:
                            B_index = True
                        else:
                            newpop[i][num][k] = factory_job_set[i][j]
                            B_index = False
            newpop[i][num][len_job[i]], newpop[i][num][len_job[i]+1] = TCE(len_job[i], num_machine, newpop[i][num], test_data, v)
    return newpop

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def swap_search(newpop, ls_frequency, len_job, num_factory,local_search_size,v):
    ls_pop = [[[[-1 for i in range(len_job[k] + 3)] for j in range(ls_frequency)] for l in range(local_search_size)]for k in range(num_factory)]
    select_ls_pop = [[[-1 for i in range(len_job[k] + 3)]  for l in range(local_search_size)]for k in range(num_factory)]
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
            fitness1_max = max(ls_pop[i][l], key=lambda x: x[len_job[i]])
            fitness1_min = min(ls_pop[i][l], key=lambda x: x[len_job[i]])
            fitness2_max = max(ls_pop[i][l], key=lambda x: x[len_job[i] + 1])
            fitness2_min = min(ls_pop[i][l], key=lambda x: x[len_job[i] + 1])
            normalized_f1 = int(fitness1_max[len_job[i]] - fitness1_min[len_job[i]])+1
            normalized_f2 = int(fitness2_max[len_job[i] + 1] - fitness2_min[len_job[i] + 1])+1
            for j in range(ls_frequency):
                ls_pop[i][l][j][-1] =  float(ls_pop[i][l][j][len_job[i]]-fitness1_min[len_job[i]])/normalized_f1+\
                                          float(ls_pop[i][l][j][len_job[i]+1]-fitness2_min[len_job[i]+1])/normalized_f2
            select_ls_pop[i][l] = sorted(ls_pop[i][l],key= lambda x:x[-1])[0]
        fitness1_max = max(select_ls_pop[i], key=lambda x: x[len_job[i]])
        fitness1_min = min(select_ls_pop[i], key=lambda x: x[len_job[i]])
        fitness2_max = max(select_ls_pop[i], key=lambda x: x[len_job[i] + 1])
        fitness2_min = min(select_ls_pop[i], key=lambda x: x[len_job[i] + 1])
        normalized_f1 = int(fitness1_max[len_job[i]] - fitness1_min[len_job[i]])+1
        normalized_f2 = int(fitness2_max[len_job[i] + 1] - fitness2_min[len_job[i] + 1])+1
        for l in range(local_search_size):
            select_ls_pop[i][l][-1] = float(select_ls_pop[i][l][len_job[i]] - fitness1_min[len_job[i]]) / normalized_f1 + \
                                 float(select_ls_pop[i][l][len_job[i] + 1] - fitness2_min[len_job[i] + 1]) / normalized_f2
        select_ls_pop[i] = sorted(select_ls_pop[i],key= lambda x:x[-1])
    return select_ls_pop

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
def insert_search(newpop, ls_frequency, len_job, num_factory, local_search_size,v, num_machine, test_data):
    ls_pop = [[[[-1 for i in range(len_job[k] + 3)] for j in range(ls_frequency)] for l in range(local_search_size)]for k in range(num_factory)]
    select_ls_pop = [[[-1 for i in range(len_job[k] + 3)]  for l in range(local_search_size)]for k in range(num_factory)]
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
            fitness1_max = max(ls_pop[i][l], key=lambda x: x[len_job[i]])
            fitness1_min = min(ls_pop[i][l], key=lambda x: x[len_job[i]])
            fitness2_max = max(ls_pop[i][l], key=lambda x: x[len_job[i] + 1])
            fitness2_min = min(ls_pop[i][l], key=lambda x: x[len_job[i] + 1])
            normalized_f1 = int(fitness1_max[len_job[i]] - fitness1_min[len_job[i]])+1
            normalized_f2 = int(fitness2_max[len_job[i] + 1] - fitness2_min[len_job[i] + 1])+1
            for j in range(ls_frequency):
                ls_pop[i][l][j][-1] = float(ls_pop[i][l][j][len_job[i]] - fitness1_min[len_job[i]]) / normalized_f1 + \
                                     float(ls_pop[i][l][j][len_job[i] + 1] - fitness2_min[len_job[i] + 1]) / normalized_f2
            select_ls_pop[i][l] = sorted(ls_pop[i][l], key=lambda x: x[-1])[0]
        fitness1_max = max(select_ls_pop[i], key=lambda x: x[len_job[i]])
        fitness1_min = min(select_ls_pop[i], key=lambda x: x[len_job[i]])
        fitness2_max = max(select_ls_pop[i], key=lambda x: x[len_job[i] + 1])
        fitness2_min = min(select_ls_pop[i], key=lambda x: x[len_job[i] + 1])
        normalized_f1 = int(fitness1_max[len_job[i]] - fitness1_min[len_job[i]])+1
        normalized_f2 = int(fitness2_max[len_job[i] + 1] - fitness2_min[len_job[i] + 1])+1
        for l in range(local_search_size):
            select_ls_pop[i][l][-1] = float(select_ls_pop[i][l][len_job[i]] - fitness1_min[len_job[i]]) / normalized_f1 + \
                                      float(select_ls_pop[i][l][len_job[i] + 1] - fitness2_min[len_job[i] + 1]) / normalized_f2
        select_ls_pop[i] = sorted(select_ls_pop[i], key=lambda x: x[-1])
    return select_ls_pop
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
def interchange(newpop, ls_frequency, len_job, num_factory, local_search_size,v):
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
            select_ls_pop[i][l] = sorted(ls_pop[i][l], key=lambda x: x[len_job[i]+1])[0]
        select_ls_pop[i] = sorted(select_ls_pop[i], key=lambda x: x[len_job[i]+1])
    return select_ls_pop

def interchange_insert(non_dominated_pop, ls_frequency, len_job, num_factory,v,num_machine,test_data):
    ls_pop = [[[-1 for i in range(len_job[k] + 2)]  for l in range(len(non_dominated_pop[k]))] for k in range(num_factory)]
   # select_ls_pop = [[[-1 for i in range(len_job[k] + 3)] for l in range(local_search_size)] for k in
   #                  range(num_factory)]
    select_ls_individual = [[-1 for i in range(len_job[k] + 2)] for k in range(num_factory)]
    for i in range(num_factory):
        temp_individual = [-1 for i in range(len_job[i])]
        for l in range(len(non_dominated_pop[i])):
            ls_pop[i][l][:] = non_dominated_pop[i][l][:]
            temp1 = random.randint(0, len_job[i] - 1)
            temp2 = random.randint(0, len_job[i] - 1)
            while temp1 == temp2:
                temp2 = random.randint(0, len_job[i] - 1)
            temp_job = ls_pop[i][l][temp1]
            ls_pop[i][l][temp1] = ls_pop[i][l][temp2]
            ls_pop[i][l][temp2] = temp_job
            #select_ls_individual[i][:] = ls_pop[i][l][:]
            ls_temp_individual = [-1 for i in range(len_job[i] + 2)]
            #进行10次局部搜索
            for j in range(ls_frequency):
                temp1 = random.randint(0, len_job[i] - 1)
                temp2 = random.randint(0, len_job[i] - 1)
                while temp1 == temp2:
                    temp2 = random.randint(0, len_job[i] - 1)
                rand_pos1 = max(temp1, temp2)
                rand_pos2 = min(temp1, temp2)
                ls_temp_individual[:] = ls_pop[i][l][:]
                ls_temp_individual[rand_pos2] = ls_pop[i][l][rand_pos1]
                for k in range(rand_pos2 + 1, rand_pos1 + 1):
                    ls_temp_individual[k] = ls_pop[i][l][k - 1]
                ls_temp_individual[len_job[i]],ls_temp_individual[len_job[i] + 1] = TCE(len_job[i], num_machine, ls_temp_individual[0:len_job[i]],
                                                                                       test_data, v)
                if ls_temp_individual[len_job[i]]<=non_dominated_pop[i][l][len_job[i]] and  ls_temp_individual[len_job[i]+1]<=non_dominated_pop[i][l][len_job[i]+1]:
                    if ls_temp_individual[len_job[i]]<non_dominated_pop[i][l][len_job[i]] or ls_temp_individual[len_job[i]+1]<non_dominated_pop[i][l][len_job[i]+1]:
                        non_dominated_pop[i][l][:] = ls_temp_individual[:]
                elif (ls_temp_individual[len_job[i]]<=non_dominated_pop[i][l][len_job[i]] and  ls_temp_individual[len_job[i]+1]>=non_dominated_pop[i][l][len_job[i]+1] )or \
                        (ls_temp_individual[len_job[i]]>=non_dominated_pop[i][l][len_job[i]] and  ls_temp_individual[len_job[i]+1]<=non_dominated_pop[i][l][len_job[i]+1]):
                    b = np.random.random()
                    if b <= 0.5:
                        non_dominated_pop[i][l][:] = ls_temp_individual[:]
                ls_pop[i][l][:] = non_dominated_pop[i][l][:]
    return non_dominated_pop




            #------------------------------------------------------------------------------------------------------------------------------------------------------------------

def block_insert(num_machine, demo, demo1,ls_frequency,select_position,num_factory,factory_job_set,len_job,v,test_data,block_number):
    for i in range(num_factory):
        all_position = [k for k in range(len_job[i])]       #所有位置
        complete_individual = [-1 for k in range(len_job[i]+2)]     #初始化一个完整个体
        all_block_position = []     #初始化所有块结构位置
        block_position = list(select_position[i])       #记录每个工厂的块结构位置
        for j in range(len(block_position)):        #记录块结构占的所有位置
            all_block_position.append(block_position[j])
            all_block_position.append(block_position[j] + 1)
            all_block_position.append(block_position[j] + 2)
        shengyu_position = list(set(all_position).difference(set(all_block_position)))      #记录剩余活动位置
        shengyu_position = list(np.sort(shengyu_position))      #对活动位置从小到大排序
        temp_individual = [-1 for k in range(len(shengyu_position))]        #初始化活动位置工件
        for j in range(len(demo1[i])):
            r = random.random()
            if r > 0.7:
                #分解块结构
                all_block_position = []
                block_position = list(select_position[i])
                # 分解块结构操作
                ppp = np.random.randint(1,int(block_number/5)+2)
                for k in range(ppp):
                    break_block = choice(block_position)
                    block_position.remove(break_block)
                for jj in range(len(block_position)):
                    all_block_position.append(block_position[jj])
                    all_block_position.append(block_position[jj] + 1)
                    all_block_position.append(block_position[jj] + 2)
                shengyu_position = list(set(all_position).difference(set(all_block_position)))
                shengyu_position = list(np.sort(shengyu_position))
                #temp_individual = [-1 for k in range(len(shengyu_position))]

            #做interchange扰动
            shengyu_gongjian = []
            ls_pop = [-1 for k in range(len(shengyu_position))]
            for k in range(len(shengyu_position)):
                shengyu_gongjian.append(demo1[i][j][shengyu_position[k]])       #按照活动位置添加活动工件
            temp_individual[:] = shengyu_gongjian[:]        #复制活动工件
            temp1 = np.random.randint(0,len(shengyu_gongjian))
            temp2 = np.random.randint(0, len(shengyu_gongjian))
            while temp2 == temp1:
                temp2 = np.random.randint(0, len(shengyu_gongjian))
            temp_job = temp_individual[temp1]
            temp_individual[temp1] = temp_individual[temp2]
            temp_individual[temp2] = temp_job       #对活动工件进行一次interchange扰动
            for l in range(ls_frequency):
                #做insert操作
                temp1 = np.random.randint(0, len(temp_individual))
                temp2 = np.random.randint(0, len(temp_individual))
                while temp2 == temp1:
                    temp2 = np.random.randint(0, len(temp_individual))
                rand_pos1 = max(temp1, temp2)
                rand_pos2 = min(temp1, temp2)
                ls_pop[:] = temp_individual[:]
                ls_pop[rand_pos2] = temp_individual[rand_pos1]
                for m in range(rand_pos2+1,rand_pos1+1):
                    ls_pop[m] = temp_individual[m-1]
                complete_individual[:] = demo1[i][j][:]
                for m in range(len(shengyu_position)):
                    complete_individual[shengyu_position[m]] = ls_pop[m]
                complete_individual[len_job[i]], complete_individual[len_job[i]+1] = TCE(len_job[i], num_machine, complete_individual[0:len_job[i]],
                                                                                       test_data, v)
                if complete_individual[len_job[i]] <= demo1[i][j][len_job[i]] and complete_individual[len_job[i]+1] <= demo1[i][j][len_job[i]+1]:
                    if complete_individual[len_job[i]] < demo1[i][j][len_job[i]] or complete_individual[len_job[i]+1]< demo1[i][j][len_job[i]+1]:
                        demo1[i][j][:] = complete_individual[:]
                for m in range(len(shengyu_position)):
                    temp_individual[m] = demo1[i][j][shengyu_position[m]]

    return demo1
def block_3dim(non_dominated_pop, len_job, update_popsize,block_number,num_factory, num_job):
    block = [[] for i in range(num_factory)]
    select_block = [[] for i in range(num_factory)]
    select_location = [[] for i in range(num_factory)]
    for i in range(num_factory):
        block_Matrix = np.zeros((len_job[i] - 2, num_job, num_job, num_job)) #带位置信息的三维block
        for k in range(int(len(non_dominated_pop[i]))):
            for j in range(0,len_job[i] - 2):
                block_Matrix[j, non_dominated_pop[i][k][j],non_dominated_pop[i][k][j+1],non_dominated_pop[i][k][j+2]] += 1
        location, height, raw, column = block_Matrix.shape
        key = True
        select_job = []
        #index = 0
        while len(block[i]) < block_number:
            _positon = np.argmax(block_Matrix)
            loc = int(_positon /height/ raw / column)
            h = int((_positon-(height*raw*column)*loc) / raw / column)
            m, n = divmod((_positon-(height*raw*column)*loc) - (raw * column) * h, column)
            if block_Matrix[loc, h, int(m), int(n)] == 0:
                break
            else:
                block[i].append([loc, h, int(m), int(n)])
                select_location[i].append(block[i][-1][0])
                for k in range(1, 4):
                    select_job.append(block[i][-1][k])
                #     #block[i].append(block_Matrix[loc, h, int(m), int(n)])
                block_Matrix[loc, h, int(m), int(n)] = -1
            len_block = len(block[i])
            if len_block == 1:
                continue
            else:
                for k in range(len_block-1):
                    if abs(block[i][-1][0] - block[i][k][0]) < 3:
                        select_job.pop()
                        select_job.pop()
                        select_job.pop()
                        block[i].pop()
                        select_location[i].pop()
                        break
                    else:
                        #other_job = set(set(select_job).difference(set(block[i][-1][1:4])))
                        for ll in range(1,4):
                            if block[i][-1][ll] in select_job[0:-3]:
                                block[i].pop()
                                select_job.pop()
                                select_job.pop()
                                select_job.pop()
                                select_location[i].pop()
                                break
                        break

    return block,select_location

def block_based(block,  non_dominated_pop, factory_job_set, len_job, update_popsize,v,num_factory, num_machine,test_data):
    #根据精英集合构建的快结构生成新种群
    factory_job_set_other = [[] for i in range(num_factory)]
    len_job_other = [0 for i in range(num_factory)]
    block_Mat_pop = [[[-1 for i in range(len_job[k] + 2)] for j in range(update_popsize)] for k in range(num_factory)]
    for i in range(num_factory):
        # for j in range(len(non_dominated_pop[i])):
        #     block_Mat_pop[i][j][:] = non_dominated_pop[i][j][:]
        for j in range(len(block[i])):
            location = block[i][j][0]
            for k in range(1,len(block[i][j])):
                for l in range(update_popsize):
                    block_Mat_pop[i][l][location] = block[i][j][k]
                location += 1
    block_set = set()
    for i in range(num_factory):
        for j in range(len(block[i])):
            for k in range(1,4):
                block_set.add(block[i][j][k])
    for i in range(num_factory):
        for j in range(len_job[i]):
            if factory_job_set[i][j] not in block_set:
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
            block_Mat_pop[i][j][len_job[i]], block_Mat_pop[i][j][len_job[i]+1] = TCE(len_job[i], num_machine, block_Mat_pop[i][j][0:len_job[i]],test_data, v)
    return block_Mat_pop

def Green_Bayes_net(pop_gen, ls_frequency, update_popsize, test_time,v,block_number,Elite_prob,num_job,num_machine, test_data, num_factory,local_search_size):
    test_timeup = time.clock()
    factory_job_set = NEH2(num_job, num_machine, test_data, num_factory,v)
    len_job = [len(factory_job_set[i]) for i in range(num_factory)]
    Mat_pop,  non_dominated_pop= green_initial_Bayes(num_machine, num_factory, factory_job_set, test_data, len_job,update_popsize,v)
    temp_non_dominated = interchange_insert(non_dominated_pop, ls_frequency, len_job, num_factory, v, num_machine,test_data)
    non_dominated_pop = B_update_non_dominated(non_dominated_pop, temp_non_dominated, factory_job_set, num_factory)
    temp_list = []
    Each_gen_pareto = []
    bayes_index = True
    gen_index = -1
    while bayes_index:
        gen_index += 1
        prob_mat_first = Bayes_update(Mat_pop, factory_job_set, num_factory, len_job, update_popsize, non_dominated_pop)
        Mat_pop = Green_New_pop(prob_mat_first, num_factory, factory_job_set, Mat_pop, len_job,v,update_popsize,local_search_size,num_machine, test_data,non_dominated_pop)
        Mat_pop = interchange_insert(Mat_pop, 2, len_job, num_factory, v, num_machine, test_data)
        temp_non_dominated = select_non_dominated_pop(num_factory, len_job, Mat_pop)
        temp_non_dominated = interchange_insert(temp_non_dominated, ls_frequency, len_job, num_factory,v,num_machine,test_data)
        non_dominated_pop = B_update_non_dominated(non_dominated_pop, temp_non_dominated,factory_job_set,num_factory)
        Each_gen_pareto.append(non_dominated_pop)
        distance1 = [0 for i in range(num_factory)]
        distance2 = [0 for i in range(num_factory)]
        yuzhidaishu = [10,20,30,40,50,60,70,80]
        if gen_index in yuzhidaishu:
            for j in range(num_factory):
                for i in range(len(Each_gen_pareto[gen_index][j])):
                    distance1[j] += math.sqrt(int(Each_gen_pareto[gen_index][j][i][-2])*int(Each_gen_pareto[gen_index][j][i][-2])+int(Each_gen_pareto[gen_index][j][i][-1])*int(Each_gen_pareto[gen_index][j][i][-1]))
                distance1[j] = distance1[j] / int(len(Each_gen_pareto[gen_index][j]))
                for i in range(len(Each_gen_pareto[gen_index-10][j])):
                    distance2[j] += math.sqrt(int(Each_gen_pareto[gen_index-10][j][i][-2])*int(Each_gen_pareto[gen_index-10][j][i][-2])+int(Each_gen_pareto[gen_index-10][j][i][-1])*int(Each_gen_pareto[gen_index-10][j][i][-1]))
                distance2[j] = distance2[j] / int(len(Each_gen_pareto[gen_index-10][j]))
                TSD = abs(distance1[j]-distance2[j])/max(distance1[j],distance2[j])
                if TSD < 0.001:
                    bayes_index = False
                    break
        test_timedown = time.clock()
        if float(test_timedown - test_timeup) >= float(test_time):
             break
    # demo ,select_position = block_3dim(non_dominated_pop, len_job, update_popsize, block_number,num_factory, num_job)
    # demo1 = block_based(demo, non_dominated_pop, factory_job_set, len_job, update_popsize,v,num_factory, num_machine,test_data)
    # temp_non_dominated = select_non_dominated_pop(num_factory, len_job, demo1)
    # non_dominated_pop = B_update_non_dominated(non_dominated_pop, temp_non_dominated, factory_job_set, num_factory)
    test_time_mid = time.clock()
    print('单独贝叶斯程序共运行：%s' % (test_time_mid-test_timeup))
    for gen in range(700):
        demo, select_position = block_3dim(non_dominated_pop, len_job, update_popsize, block_number, num_factory,
                                           num_job)
        demo1 = block_based(demo, non_dominated_pop, factory_job_set, len_job, update_popsize, v, num_factory,
                            num_machine, test_data)
        temp_non_dominated = select_non_dominated_pop(num_factory, len_job, demo1)
        non_dominated_pop = B_update_non_dominated(non_dominated_pop, temp_non_dominated, factory_job_set, num_factory)
        demo1 = block_insert(num_machine, demo, demo1,ls_frequency,select_position,num_factory,factory_job_set,len_job,v,test_data,block_number)
        temp_non_dominated = select_non_dominated_pop(num_factory, len_job, demo1)
        non_dominated_pop = B_update_non_dominated(non_dominated_pop, temp_non_dominated,factory_job_set,num_factory)
        test_timedown = time.clock()
        if float(test_timedown-test_timeup) >= float(test_time):
            break
    return non_dominated_pop, float(test_timedown-test_timeup)


def dandu_siwei(pop_gen, ls_frequency, update_popsize, test_time,v,block_number,Elite_prob,num_job,num_machine, test_data, num_factory,local_search_size):
    test_timeup = time.clock()
    factory_job_set = NEH2(num_job, num_machine, test_data, num_factory,v)
    len_job = [len(factory_job_set[i]) for i in range(num_factory)]
    Mat_pop,  non_dominated_pop= green_initial_Bayes(num_machine, num_factory, factory_job_set, test_data, len_job,update_popsize,v)
    temp_non_dominated = interchange_insert(non_dominated_pop, ls_frequency, len_job, num_factory, v, num_machine,test_data)
    non_dominated_pop = B_update_non_dominated(non_dominated_pop, temp_non_dominated, factory_job_set, num_factory)
   # temp_list = []
    for gen in range(700):
        #prob_mat_first = Bayes_update(Mat_pop, factory_job_set, num_factory, len_job, update_popsize, non_dominated_pop)
      #  Mat_pop = Green_New_pop(prob_mat_first, num_factory, factory_job_set, Mat_pop, len_job,v,update_popsize,local_search_size,num_machine, test_data,non_dominated_pop)
       # Mat_pop = interchange_insert(Mat_pop, 2, len_job, num_factory, v, num_machine, test_data)
       # temp_non_dominated = select_non_dominated_pop(num_factory, len_job, Mat_pop)
       # temp_non_dominated = interchange_insert(temp_non_dominated, ls_frequency, len_job, num_factory,v,num_machine,test_data)
       # non_dominated_pop = B_update_non_dominated(non_dominated_pop, temp_non_dominated,factory_job_set,num_factory)
        #test_timedown = time.clock()
       # if float(test_timedown - test_timeup) >= float(test_time):
       #      break
        demo ,select_position = block_3dim(non_dominated_pop, len_job, update_popsize, block_number,num_factory, num_job)
        demo1 = block_based(demo, non_dominated_pop, factory_job_set, len_job, update_popsize,v,num_factory, num_machine,test_data)
        temp_non_dominated = select_non_dominated_pop(num_factory, len_job, demo1)
        non_dominated_pop = B_update_non_dominated(non_dominated_pop, temp_non_dominated, factory_job_set, num_factory)
    #test_time_mid = time.clock()
    #print('单独贝叶斯程序共运行：%s' % (test_time_mid-test_timeup))
    #for gen in range(pop_gen-50):
        demo1 = block_insert(num_machine, demo, demo1,ls_frequency,select_position,num_factory,factory_job_set,len_job,v,test_data)
       # demo1 = interchange_insert(demo1, 100, len_job, num_factory,v,num_machine,test_data)
        # for i in range(num_factory):
        #     for k in range(int(local_search_size)):
        #         for j in range(update_popsize):
        #             if float(ls_pop[i][k][len_job[i]]) <= float(demo1[i][j][len_job[i]]) and float(ls_pop[i][k][len_job[i] + 1]) <= float(demo1[i][j][len_job[i] + 1]):
        #                 if float(ls_pop[i][k][len_job[i]]) < float(demo1[i][j][len_job[i]]) or float(ls_pop[i][k][len_job[i] + 1]) < float(demo1[i][j][len_job[i] + 1]):
        #                     for l in range(len(factory_job_set[i]) + 2):
        #                         demo1[i][j][l] = ls_pop[i][k][l]
        #                     break
        #             if (float(ls_pop[i][k][len_job[i]]) <= float(demo1[i][j][len_job[i]]) and float(ls_pop[i][k][len_job[i] + 1]) >= float(demo1[i][j][len_job[i] + 1])) or\
        #                     (float(ls_pop[i][k][len_job[i]]) >= float(demo1[i][j][len_job[i]]) and float(ls_pop[i][k][len_job[i] + 1]) <= float(demo1[i][j][len_job[i] + 1])):
        #                 b = random.random()
        #                 if b <= 0.5:
        #                     for l in range(len(factory_job_set[i]) + 2):
        #                         demo1[i][j][l] = ls_pop[i][k][l]
        #                     break

        temp_non_dominated = select_non_dominated_pop(num_factory, len_job, demo1)
        non_dominated_pop = B_update_non_dominated(non_dominated_pop, temp_non_dominated,factory_job_set,num_factory)
        test_timedown = time.clock()
        if float(test_timedown-test_timeup) >= float(test_time):
            break
    return non_dominated_pop, float(test_timedown-test_timeup)
def B_All_factory_dominated(pop_gen, ls_frequency,update_popsize, num_factory, test_time,v,block_number,Elite_prob,num_job,num_machine, test_data,local_search_size):
    #根据每个工厂的帕累托解确定总工厂的解
    non_dominated_pop, run_time = Green_Bayes_net(pop_gen, ls_frequency, update_popsize, test_time,v,block_number,Elite_prob,num_job,num_machine, test_data, num_factory,local_search_size)
    print('贝叶斯程序共运行：%s' % (run_time))
    temp_all_f_dominated = []
    result = []
    temp_set = list(itertools.product(non_dominated_pop[0],non_dominated_pop[1]))#lambda x: list(x) for x in non_dominated_pop[0])
    for individual in temp_set:
        individual = sorted(individual,key=lambda x:x[-2])
        parteo_solution = [0 for i in range(len(individual[-1]))]
        sum_green_fitness = 0
        for indi_green in individual:
            sum_green_fitness += indi_green[-1]
        for i in range(len(individual[-1])):
            parteo_solution[i] = individual[-1][i]
        parteo_solution[-1] = sum_green_fitness
        temp_all_f_dominated.append(parteo_solution)
    result = select_all_f_non_dominated_pop(temp_all_f_dominated)
    return result

#B_All_factory_dominated(pop_gen, ls_frequency,update_popsize, num_factory)
