# -*- coding:utf-8 -*-
import LoadData as ld
import numpy as np
import random
from random import choice

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

def CalcFitness(n, m, test_data):
    c_time1 = np.zeros([n, m])
    c_time1[0][0] = test_data[0][0]
    for i in range(1, n):
        c_time1[i][0] = c_time1[i - 1][0] + test_data[i][0]
    for i in range(1, m):
        c_time1[0][i] = c_time1[0][i - 1] + test_data[0][i]
    for i in range(1, n):
        for k in range(1, m):
            c_time1[i][k] = test_data[i][k] + max(c_time1[i - 1][k], c_time1[i][k - 1])
    return c_time1[n - 1][m - 1]

def NEH2(num_job, num_machine, test_data, num_factory,v):
    first_job = []
    factory_job_set = [[] for i in range(num_factory)]
    first_job_index = []
    factory_data = [[] for i in range(num_factory)]
    factory_fit = []
    c_time = np.zeros([num_job, num_machine])
    #确定每一个工厂的第一个排序工件号
    temp = []
    k = 0
    for i in range(num_job):
        temp.append(sum(test_data[i]))
    first_job_index = np.argsort(temp)
    for i in range(num_factory):
        first_job.append(first_job_index[i] )
        factory_job_set[i].append(first_job[i])
        #为每个工厂分配第一个工件
        factory_data[i].append(test_data[first_job_index[i]])
    while True:
        if k == num_job - num_factory :
            break
        for i in range(num_factory):
            factory_job_set[i].append(first_job_index[num_factory + k])
            factory_data[i].append(test_data[first_job_index[num_factory + k]])
            factory_fittness= TCE(len(factory_data[i]),num_machine,factory_job_set[i],test_data,v)
            factory_fit.append(factory_fittness)
        #依次分配剩下的工件
        temp_non_dominated_pop = []
        for i in range(num_factory):
            compare_fitness1 = factory_fit[i][0]
            compare_fitness2 = factory_fit[i][1]
            b_non_dominated = True
            for j in range(num_factory):
                if i != j:
                    if factory_fit[j][0] <= compare_fitness1 and factory_fit[j][1] <= compare_fitness2:
                        if factory_fit[j][0] < compare_fitness1 or factory_fit[j][1] < compare_fitness2:
                            b_non_dominated = False
                            break
            if b_non_dominated == True:
                if factory_fit[i] not in temp_non_dominated_pop:
                    temp_non_dominated_pop.append(factory_fit[i])
        r = random.randint(0, len(temp_non_dominated_pop)-1)
        index = factory_fit.index(temp_non_dominated_pop[r])
        factory_fit = []
        for i in range(num_factory):
            if i == index:
                continue
            else:
                factory_job_set[i].pop()
                factory_data[i].pop()
        k += 1
    return  factory_job_set



def Bayes_update(Mat_pop, factory_job_set, num_factory, len_job, update_popsize):
    prob_mat_first = [[1/len_job[k] for i in range(len_job[k])] for k in range(num_factory)]
    #返回每个工厂相邻两个工件的所有情况的概率分布
    for i in range(num_factory):
        #确定数据中第一个工件的出现概率
        index = 0
        demo = [ii[0] for ii in Mat_pop[i][0:int(update_popsize*0.2)]]
        for job in factory_job_set[i]:
            prob_mat_first[i][index] += demo.count(job) / int(update_popsize*0.2)
            index += 1
        #每个工厂内分别有对应的（job_len - 1）个关系数组
        zongshu = sum(prob_mat_first[i])
        for j in range(len_job[i]):
            prob_mat_first[i][j] = prob_mat_first[i][j]/zongshu  #归一化
    return prob_mat_first

def Roulette_prob(prob_mat, len_job):
    dichotomy = [0.0 for i in range(len_job)]
    dichotomy[0] = prob_mat[0]
    for i in range(1,len_job):
        dichotomy[i] = prob_mat[i] + dichotomy[i - 1]
    return dichotomy

def Roulette_dichotomy(r, dichotomy, begin, end):       #二分法快速轮盘赌
    mid = int((end-begin) / 2) + begin
    if end - begin < 2 and r > dichotomy[begin]:
        return end
    if end - begin < 2 and r <= dichotomy[begin]:
        return begin
    elif r <= dichotomy[mid]:
        return Roulette_dichotomy(r, dichotomy, begin, mid)
    elif r > dichotomy[mid]:
        return Roulette_dichotomy(r, dichotomy, mid, end )

