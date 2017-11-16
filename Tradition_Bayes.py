import time
import random
import numpy as np
from random import choice
import math

def Roulette_prob(prob_mat, num_job):
    dichotomy = [0.0 for i in range(num_job)]
    dichotomy[0] = prob_mat[0]
    for i in range(1,num_job):
        dichotomy[i] = prob_mat[i] + dichotomy[i - 1]
    return dichotomy

def Roulette_dichotomy(r, dichotomy, begin, end):
    mid = int((end-begin) / 2) + begin
    if end - begin < 2 and r > dichotomy[begin]:
        return end
    if end - begin < 2 and r <= dichotomy[begin]:
        return begin
    elif r <= dichotomy[mid]:
        return Roulette_dichotomy(r, dichotomy, begin, mid)
    elif r > dichotomy[mid]:
        return Roulette_dichotomy(r, dichotomy, mid, end )



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
                per_consumption_V = 4 * v[i][k] * v[i][k]
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

def green_initial_Bayes(num_machine, num_factory, test_data, num_job, update_popsize,v):
    Mat_pop =[[0 for i in range(num_job + 2) ]for j in range(update_popsize)]
    job_set = [i for i in range(num_job)]
    non_dominated_pop = []
    for j in range(update_popsize):
        sort = random.sample(job_set, num_job)
        for k in range(num_job + 1):
            if k == num_job:
                Mat_pop[j][k], Mat_pop[j][k+1]= TCE(num_job, num_machine, test_data,v, num_factory, sort)
            else:
                Mat_pop[j][k] = sort[k]
    for j in range(update_popsize):
        compare_fitness1 = Mat_pop[j][num_job]
        compare_fitness2 = Mat_pop[j][num_job+1]
        b_non_dominated = True
        for k in range(update_popsize):
            if j != k:
                if Mat_pop[k][num_job] <= compare_fitness1 and Mat_pop[k][num_job+1] <= compare_fitness2:
                    if Mat_pop[k][num_job] < compare_fitness1 or Mat_pop[k][num_job+1] < compare_fitness2:
                        b_non_dominated = False
                        break
        if b_non_dominated == True:
            if Mat_pop[j][0:num_job+2] not in non_dominated_pop:
                non_dominated_pop.append(Mat_pop[j][0:num_job+2])

    return Mat_pop, non_dominated_pop

def select_non_dominated_pop(num_factory, num_job, Mat_pop):
    temp_non_dominated_pop = []
    #for i in range(num_factory):
    for j in range(len(Mat_pop)):
        compare_fitness1 = Mat_pop[j][num_job]
        compare_fitness2 = Mat_pop[j][num_job+1]
        b_non_dominated = True
        for k in range(len(Mat_pop)):
            if j != k:
                if Mat_pop[k][num_job] <= compare_fitness1 and Mat_pop[k][num_job+1] <= compare_fitness2:
                    if Mat_pop[k][num_job] < compare_fitness1 or Mat_pop[k][num_job+1] < compare_fitness2:
                        b_non_dominated = False
                        break
        if b_non_dominated == True:
            if Mat_pop[j][0:num_job + 2] not in temp_non_dominated_pop:
                temp_non_dominated_pop.append(Mat_pop[j][0:num_job + 2])
    return temp_non_dominated_pop

def B_update_non_dominated(B_non_dominated_pop, B_temp_non_dominated,num_job,num_factory):
    #for i in range(num_factory):
    for j in range(len(B_temp_non_dominated)):
        if B_temp_non_dominated[j][0:num_job+2] not in B_non_dominated_pop:
            B_non_dominated_pop.append(B_temp_non_dominated[j][0:num_job+2])
    B_non_dominated_pop = select_non_dominated_pop(num_factory, num_job, B_non_dominated_pop)
    return B_non_dominated_pop


def Bayes_update(Mat_pop, num_job, num_factory, update_popsize, non_dominated_pop):
    job_set = [i for i in range(num_job)]
    prob_mat_first = [0 for i in range(num_job)]
    index = 0
    demo = [ii[0] for ii in non_dominated_pop]
    for job in job_set:
        prob_mat_first[index] += demo.count(job) / len(non_dominated_pop)
        index += 1
    return prob_mat_first

def Green_New_pop(prob_mat_first, num_factory, num_job, Mat_pop, v,update_popsize,local_search_size,num_machine, test_data, non_dominated_pop):
    job_set = [i for i in range(num_job)]
    newpop = [[-1 for i in range(num_job + 2)] for j in range(update_popsize)]
    prob_mat = [[1/num_job for i in range(num_job)] for l in range(num_job - 1)]
    for num in range(update_popsize):
        temp = 0.0
        r = random.random()
        dichotomy = Roulette_prob(prob_mat_first, num_job)
        begin = 0
        end = num_job - 1
        j = Roulette_dichotomy(r, dichotomy, begin, end)
        newpop[num][0] = job_set[j]
        for k in range(1, num_job):
            temp_job = []
            for ii in non_dominated_pop:
                if ii[k] in newpop[num]:
                    continue
                elif ii[k - 1] == newpop[num][k - 1]:
                    temp_job.append(ii[k])
            if len(temp_job) == 0:
                shengyu_gongjian = list(set(job_set).difference(set(newpop[num])))
                newpop_temp = choice(shengyu_gongjian)
                newpop[num][k] = newpop_temp
            else:
                for m in range(num_job):
                    if job_set[m] in newpop[num]:
                        prob_mat[k - 1][m] = 0
                    else:
                        prob_mat[k - 1][m] += temp_job.count(job_set[m]) / len(temp_job)
                zongshu = sum(prob_mat[k - 1])
                for m in range(num_job):
                    prob_mat[k - 1][m] = prob_mat[k - 1][m]/zongshu
                B_index = True
                while B_index:
                    r = random.random()
                    dichotomy = Roulette_prob(prob_mat[k - 1], num_job)
                    begin = 0
                    end = num_job - 1
                    j = Roulette_dichotomy(r, dichotomy, begin, end)
                    if job_set[j] in newpop[num]:
                        B_index = True
                    else:
                        newpop[num][k] = job_set[j]
                        B_index = False
        newpop[num][num_job], newpop[num][num_job+1] = TCE(num_job, num_machine, test_data,v, num_factory, newpop[num])
    return newpop

def interchange_insert(non_dominated_pop, ls_frequency, num_job, num_factory,v,num_machine,test_data):
    ls_pop = [[-1 for i in range(num_job + 2)]  for l in range(len(non_dominated_pop))]
    select_ls_individual = [-1 for i in range(num_job + 2)]
    temp_individual = [-1 for i in range(num_job)]
    for l in range(len(non_dominated_pop)):
        ls_pop[l][:] = non_dominated_pop[l][:]
        temp1 = random.randint(0, num_job - 1)
        temp2 = random.randint(0, num_job - 1)
        while temp1 == temp2:
            temp2 = random.randint(0, num_job - 1)
        temp_job = ls_pop[l][temp1]
        ls_pop[l][temp1] = ls_pop[l][temp2]
        ls_pop[l][temp2] = temp_job
        ls_temp_individual = [-1 for i in range(num_job + 2)]
        for j in range(ls_frequency):
            temp1 = random.randint(0, num_job - 1)
            temp2 = random.randint(0, num_job - 1)
            while temp1 == temp2:
                temp2 = random.randint(0, num_job - 1)
            rand_pos1 = max(temp1, temp2)
            rand_pos2 = min(temp1, temp2)
            ls_temp_individual[:] = ls_pop[l][:]
            ls_temp_individual[rand_pos2] = ls_pop[l][rand_pos1]
            for k in range(rand_pos2 + 1, rand_pos1 + 1):
                ls_temp_individual[k] = ls_pop[l][k - 1]
            ls_temp_individual[num_job],ls_temp_individual[num_job + 1] = TCE(num_job, num_machine, test_data,v, num_factory, ls_temp_individual[0:num_job])
            if ls_temp_individual[num_job]<=non_dominated_pop[l][num_job] and  ls_temp_individual[num_job+1]<=non_dominated_pop[l][num_job+1]:
                if ls_temp_individual[num_job]<non_dominated_pop[l][num_job] or ls_temp_individual[num_job+1]<non_dominated_pop[l][num_job+1]:
                    non_dominated_pop[l][:] = ls_temp_individual[:]
            elif (ls_temp_individual[num_job]<=non_dominated_pop[l][num_job] and  ls_temp_individual[num_job+1]>=non_dominated_pop[l][num_job+1] )or \
                    (ls_temp_individual[num_job]>=non_dominated_pop[l][num_job] and  ls_temp_individual[num_job+1]<=non_dominated_pop[l][num_job+1]):
                b = np.random.random()
                if b <= 0.5:
                    non_dominated_pop[l][:] = ls_temp_individual[:]
            ls_pop[l][:] = non_dominated_pop[l][:]
    return non_dominated_pop

def block_insert(num_machine, demo, demo1,ls_frequency,select_position,num_factory,num_job,v,test_data,block_number):
    for i in range(num_factory):
        all_position = [k for k in range(num_job)]
        complete_individual = [-1 for k in range(num_job+2)]
        all_block_position = []
        block_position = list(select_position)
        for j in range(len(block_position)):
            all_block_position.append(block_position[j])
            all_block_position.append(block_position[j] + 1)
            all_block_position.append(block_position[j] + 2)
        shengyu_position = list(set(all_position).difference(set(all_block_position)))
        shengyu_position = list(np.sort(shengyu_position))
        temp_individual = [-1 for k in range(len(shengyu_position))]
        for j in range(len(demo1)):
            r = random.random()
            if r > 0.7:
                all_block_position = []
                block_position = list(select_position)
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
            shengyu_gongjian = []
            ls_pop = [-1 for k in range(len(shengyu_position))]
            for k in range(len(shengyu_position)):
                shengyu_gongjian.append(demo1[j][shengyu_position[k]])
            temp_individual[:] = shengyu_gongjian[:]
            temp1 = np.random.randint(0,len(shengyu_gongjian))
            temp2 = np.random.randint(0, len(shengyu_gongjian))
            while temp2 == temp1:
                temp2 = np.random.randint(0, len(shengyu_gongjian))
            temp_job = temp_individual[temp1]
            temp_individual[temp1] = temp_individual[temp2]
            temp_individual[temp2] = temp_job
            for l in range(ls_frequency):
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
                complete_individual[:] = demo1[j][:]
                for m in range(len(shengyu_position)):
                    complete_individual[shengyu_position[m]] = ls_pop[m]
                complete_individual[num_job], complete_individual[num_job+1] = TCE(num_job, num_machine, test_data,v, num_factory, complete_individual[0:num_job])
                if complete_individual[num_job] <= demo1[j][num_job] and complete_individual[num_job+1] <= demo1[j][num_job+1]:
                    if complete_individual[num_job] < demo1[j][num_job] or complete_individual[num_job+1]< demo1[j][num_job+1]:
                        demo1[j][:] = complete_individual[:]
                for m in range(len(shengyu_position)):
                    temp_individual[m] = demo1[j][shengyu_position[m]]

    return demo1

def block_3dim(non_dominated_pop, update_popsize,block_number,num_factory, num_job):
    block = []
    select_block = []
    select_location = []
    block_Matrix = np.zeros((num_job - 2, num_job, num_job, num_job))
    for k in range(int(len(non_dominated_pop))):
        for j in range(0,num_job - 2):
            block_Matrix[j, non_dominated_pop[k][j],non_dominated_pop[k][j+1],non_dominated_pop[k][j+2]] += 1
    location, height, raw, column = block_Matrix.shape
    key = True
    select_job = []
    while len(block) < block_number:
        _positon = np.argmax(block_Matrix)
        loc = int(_positon /height/ raw / column)
        h = int((_positon-(height*raw*column)*loc) / raw / column)
        m, n = divmod((_positon-(height*raw*column)*loc) - (raw * column) * h, column)
        if block_Matrix[loc, h, int(m), int(n)] == 0:
            break
        else:
            block.append([loc, h, int(m), int(n)])
            select_location.append(block[-1][0])
            for k in range(1, 4):
                select_job.append(block[-1][k])
            block_Matrix[loc, h, int(m), int(n)] = -1
        len_block = len(block)
        if len_block == 1:
            continue
        else:
            for k in range(len_block-1):
                if abs(block[-1][0] - block[k][0]) < 3:
                    select_job.pop()
                    select_job.pop()
                    select_job.pop()
                    block.pop()
                    select_location.pop()
                    break
                else:
                    for ll in range(1,4):
                        if block[-1][ll] in select_job[0:-3]:
                            block.pop()
                            select_job.pop()
                            select_job.pop()
                            select_job.pop()
                            select_location.pop()
                            break
                    break

    return block,select_location

def block_based(block,  non_dominated_pop, num_job, update_popsize,v,num_factory, num_machine,test_data):
    job_set = [i for i in range(num_job)]
    factory_job_set_other = []
    num_job_other = 0
    block_Mat_pop = [[-1 for i in range(num_job + 2)] for j in range(update_popsize)]
    for j in range(len(block)):
        location = block[j][0]
        for k in range(1,len(block[j])):
            for l in range(update_popsize):
                block_Mat_pop[l][location] = block[j][k]
            location += 1
    block_set = set()
    for j in range(len(block)):
        for k in range(1,4):
            block_set.add(block[j][k])
    for j in range(num_job):
        if job_set[j] not in block_set:
            factory_job_set_other.append(job_set[j])
    num_job_other = len(factory_job_set_other)
    for j in range(update_popsize):
        sort = random.sample(job_set, num_job_other)
        index = 0
        for k in range(num_job):
            if block_Mat_pop[j][k] == -1:
                block_Mat_pop[j][k] = sort[index]
                index += 1
            if index == num_job_other:
                break
    for j in range(update_popsize):
        block_Mat_pop[j][num_job], block_Mat_pop[j][num_job+1] = TCE(num_job, num_machine, test_data,v, num_factory, block_Mat_pop[j][0:num_job])
    return block_Mat_pop

def Green_Bayes_net(pop_gen, ls_frequency, update_popsize, test_time,v,block_number,Elite_prob,num_job,num_machine, test_data, num_factory,local_search_size):
    test_timeup = time.clock()
    Mat_pop,  non_dominated_pop= green_initial_Bayes(num_machine, num_factory, test_data, num_job, update_popsize,v)
    temp_non_dominated = interchange_insert(non_dominated_pop, ls_frequency, num_job, num_factory, v, num_machine,test_data)
    non_dominated_pop = B_update_non_dominated(non_dominated_pop, temp_non_dominated, num_job, num_factory)
    temp_list = []
    Each_gen_pareto = []
    bayes_index = True
    gen_index = -1
    while bayes_index:
        gen_index += 1
        prob_mat_first = Bayes_update(Mat_pop, num_job, num_factory, update_popsize, non_dominated_pop)
        Mat_pop = Green_New_pop(prob_mat_first, num_factory, num_job, Mat_pop, v,update_popsize,local_search_size,num_machine, test_data, non_dominated_pop)
        Mat_pop = interchange_insert(Mat_pop, 2, num_job, num_factory, v, num_machine, test_data)
        temp_non_dominated = select_non_dominated_pop(num_factory, num_job, Mat_pop)
        temp_non_dominated = interchange_insert(temp_non_dominated, ls_frequency, num_job, num_factory,v,num_machine,test_data)
        non_dominated_pop = B_update_non_dominated(non_dominated_pop, temp_non_dominated,num_job,num_factory)
        Each_gen_pareto.append(non_dominated_pop)
        distance1 = 0
        distance2 = 0
        yuzhidaishu = [10,20,30,40,50,60,70,80]
        if gen_index in yuzhidaishu:
            distance1 = 0
            distance2 = 0
            for i in range(len(Each_gen_pareto[gen_index])):
                distance1 += math.sqrt(int(Each_gen_pareto[gen_index][i][-2])*int(Each_gen_pareto[gen_index][i][-2])+int(Each_gen_pareto[gen_index][i][-1])*int(Each_gen_pareto[gen_index][i][-1]))
            distance1 = distance1 / int(len(Each_gen_pareto[gen_index]))
            for i in range(len(Each_gen_pareto[gen_index-10])):
                distance2 += math.sqrt(int(Each_gen_pareto[gen_index-10][i][-2])*int(Each_gen_pareto[gen_index-10][i][-2])+int(Each_gen_pareto[gen_index-10][i][-1])*int(Each_gen_pareto[gen_index-10][i][-1]))
            distance2 = distance2 / int(len(Each_gen_pareto[gen_index-10]))
            TSD = abs(distance1-distance2)/max(distance1,distance2)
            if TSD < 0.01:
                bayes_index = False
                break
        test_timedown = time.clock()
        if float(test_timedown - test_timeup) >= float(test_time):
             break
    test_time_mid = time.clock()
    print('Only Bayes run %s' % (test_time_mid-test_timeup))
    for gen in range(700):
        demo, select_position = block_3dim(non_dominated_pop, update_popsize,block_number,num_factory, num_job)
        demo1 = block_based(demo, non_dominated_pop, num_job, update_popsize,v,num_factory, num_machine,test_data)
        temp_non_dominated = select_non_dominated_pop(num_factory, num_job, demo1)
        non_dominated_pop = B_update_non_dominated(non_dominated_pop, temp_non_dominated, num_job, num_factory)
        demo1 = block_insert(num_machine, demo, demo1,ls_frequency,select_position,num_factory,num_job,v,test_data,block_number)
        temp_non_dominated = select_non_dominated_pop(num_factory, num_job, demo1)
        non_dominated_pop = B_update_non_dominated(non_dominated_pop, temp_non_dominated,num_job,num_factory)
        test_timedown = time.clock()
        if float(test_timedown-test_timeup) >= float(test_time):
            break
    return non_dominated_pop#, float(test_timedown-test_timeup)