from AssignRule import *
import time
from itertools import combinations
import itertools
#from New_Bayes import *
#from Japan_Multi_object import *


def single_fitness(num_machine, num_factory, GGA_popsize, GGA_Mat_pop, GGA_factory_job_set):
    s = [0 for i in range(GGA_popsize)]
    r = [0 for i in range(GGA_popsize)]
    d = [0 for i in range(GGA_popsize)]
    GGA_sum1 = [0 for i in range(GGA_popsize)]
    GGA_sigma = [[0 for i in range(GGA_popsize)] for j in range(GGA_popsize)]
    GGA_len_job = [len(GGA_factory_job_set[i]) for i in range(num_factory)]
    for i in range(num_factory):
        for j in range(GGA_popsize):
            for k in range(GGA_popsize):
                if j != k:
                    if GGA_Mat_pop[i][j][GGA_len_job[i]] <= GGA_Mat_pop[i][k][GGA_len_job[i]] and \
                       GGA_Mat_pop[i][j][GGA_len_job[i]+1] <= GGA_Mat_pop[i][k][GGA_len_job[i]+1]:
                        if GGA_Mat_pop[i][j][GGA_len_job[i]] < GGA_Mat_pop[i][k][GGA_len_job[i]] or \
                           GGA_Mat_pop[i][j][GGA_len_job[i] + 1] < GGA_Mat_pop[i][k][GGA_len_job[i] + 1]:
                            s[j] += 1
        for k in range(GGA_popsize):
            for j in range(GGA_popsize):
                if j != k:
                    if GGA_Mat_pop[i][j][GGA_len_job[i]] <= GGA_Mat_pop[i][k][GGA_len_job[i]] and \
                       GGA_Mat_pop[i][j][GGA_len_job[i]+1] <= GGA_Mat_pop[i][k][GGA_len_job[i]+1]:
                        if GGA_Mat_pop[i][j][GGA_len_job[i]] < GGA_Mat_pop[i][k][GGA_len_job[i]] or \
                           GGA_Mat_pop[i][j][GGA_len_job[i] + 1] < GGA_Mat_pop[i][k][GGA_len_job[i] + 1]:
                            r[k] = r[k] + s[j]

        for j in range(GGA_popsize):
            for k in range(GGA_popsize):
                if j != k:
                    GGA_sigma[j][k] = np.sqrt((GGA_Mat_pop[i][j][GGA_len_job[i]] - GGA_Mat_pop[i][k][GGA_len_job[i]])*(GGA_Mat_pop[i][j][GGA_len_job[i]] - GGA_Mat_pop[i][k][GGA_len_job[i]])+ \
                                          (GGA_Mat_pop[i][j][GGA_len_job[i]+1] - GGA_Mat_pop[i][k][GGA_len_job[i]+1]) * (GGA_Mat_pop[i][j][GGA_len_job[i]+1] - GGA_Mat_pop[i][k][GGA_len_job[i]+1]))
            GGA_sigma[j] = np.sort(GGA_sigma[j])
            kk = int(np.sqrt(GGA_popsize))
            for k in range(kk):
                GGA_sum1[j] = GGA_sum1[j] + GGA_sigma[j][k]
            d[j] = 1/(GGA_sum1[j] + 2)
            GGA_Mat_pop[i][j][-1] = r[j] + d[j]
    return GGA_Mat_pop

def Multi_initial_GGA(num_machine, num_factory, GGA_factory_job_set, test_data, GGA_popsize,v):
    #每个工厂的工件数
    GGA_len_job = [len(GGA_factory_job_set[i]) for i in range(num_factory)]
    #每个工厂的初始数据
    GGA_Mat_pop =[[[0 for i in range(GGA_len_job[k] + 3) ]for j in range(GGA_popsize)] for k in range(num_factory)]
    GGA_non_dominated_pop = [[] for k in range(num_factory)]
    for i in range(num_factory):
        for j in range(GGA_popsize):
            GGA_sort = random.sample(GGA_factory_job_set[i], GGA_len_job[i])
            for k in range(GGA_len_job[i] + 1):
                if k == GGA_len_job[i]:
                    GGA_Mat_pop[i][j][k], GGA_Mat_pop[i][j][k+1]= TCE(GGA_len_job[i], num_machine, GGA_sort, test_data,v)
                else:
                    GGA_Mat_pop[i][j][k] = GGA_sort[k]
        for j in range(GGA_popsize):
            GGA_compare_fitness1 = GGA_Mat_pop[i][j][GGA_len_job[i]]
            GGA_compare_fitness2 = GGA_Mat_pop[i][j][GGA_len_job[i]+1]
            b_non_dominated = True
            for k in range(GGA_popsize):
                if j != k:
                    if GGA_Mat_pop[i][k][GGA_len_job[i]] <= GGA_compare_fitness1 and GGA_Mat_pop[i][k][GGA_len_job[i]+1] <= GGA_compare_fitness2:
                        if GGA_Mat_pop[i][k][GGA_len_job[i]] < GGA_compare_fitness1 or GGA_Mat_pop[i][k][GGA_len_job[i]+1] < GGA_compare_fitness2:
                            b_non_dominated = False
                            break
            if b_non_dominated == True:
                if GGA_Mat_pop[i][j][0:GGA_len_job[i]+2] not in GGA_non_dominated_pop[i]:
                    GGA_non_dominated_pop[i].append(GGA_Mat_pop[i][j][0:GGA_len_job[i]+2])
    GGA_Mat_pop = single_fitness(num_machine, num_factory, GGA_popsize, GGA_Mat_pop, GGA_factory_job_set)
    return GGA_Mat_pop, GGA_non_dominated_pop






def select(num_machine, num_factory, GGA_popsize, GGA_Mat_pop, GGA_factory_job_set):
    GGA_len_job = [len(GGA_factory_job_set[i]) for i in range(num_factory)]
    GGA_Mat_pop1 =[[[0 for i in range(GGA_len_job[k] + 3) ]for j in range(GGA_popsize)] for k in range(num_factory)]
    GGA_select_1 = [0 for k in range(num_factory)]
    GGA_select_2 = [0 for k in range(num_factory)]
    for i in range(num_factory):
        for j in range(GGA_popsize):
            for k in range(GGA_len_job[i]+3):
                GGA_Mat_pop1[i][j][k] = GGA_Mat_pop[i][j][k]
    for i in range(num_factory):
        GGA_Mat_pop1[i] = sorted(GGA_Mat_pop1[i], key= lambda x:x[-1])
    for i in range(num_factory):
        u = np.random.randint(int(GGA_popsize/2+1))
        v = np.random.randint(int(GGA_popsize/2+1))
        while u == v:
            u = np.random.randint(int(GGA_popsize / 2 + 1))
            v = np.random.randint(int(GGA_popsize / 2 + 1))
        if GGA_Mat_pop1[i][u][GGA_len_job[i]+2] < GGA_Mat_pop1[i][v][GGA_len_job[i]+2]:
            GGA_select_1[i] = u
        elif GGA_Mat_pop1[i][u][GGA_len_job[i]+2] > GGA_Mat_pop1[i][v][GGA_len_job[i]+2]:
            GGA_select_1[i] = v
        else:
            b = np.random.random()
            if b < 0.5:
                GGA_select_1[i] = u
            else:
                GGA_select_1[i] = v
        b_label = True
        while b_label:
            u = np.random.randint(int(GGA_popsize / 2 + 1))
            v = np.random.randint(int(GGA_popsize / 2 + 1))
            while u == v:
                u = np.random.randint(int(GGA_popsize / 2 + 1))
                v = np.random.randint(int(GGA_popsize / 2 + 1))
            if GGA_Mat_pop1[i][u][GGA_len_job[i] + 2] < GGA_Mat_pop1[i][v][GGA_len_job[i] + 2]:
                GGA_select_2[i] = u
            elif GGA_Mat_pop1[i][u][GGA_len_job[i] + 2] > GGA_Mat_pop1[i][v][GGA_len_job[i] + 2]:
                GGA_select_2[i] = v
            else:
                b = np.random.random()
                if b < 0.5:
                    GGA_select_2[i] = u
                else:
                    GGA_select_2[i] = v
            if GGA_select_1[i] == GGA_select_2[i]:
                continue
            else:
                b_label = False
    return GGA_select_1, GGA_select_2

def crossover(GGA_Mat_pop,num_factory, GGA_select_1, GGA_select_2,GGA_factory_job_set):
    job_len = [len(GGA_factory_job_set[i]) for i in range(num_factory)]
    child = [[0 for i in range(job_len[k]+3)] for k in range(num_factory)]
    for i in range(num_factory):
        parent1 = [0 for k in range(job_len[i] + 3)]
        parent2 = [0 for k in range(job_len[i] + 3)]
        parent1[:] = GGA_Mat_pop[i][GGA_select_1[i]][:]
        parent2[:] = GGA_Mat_pop[i][GGA_select_2[i]][:]
        temp1 = random.randint(1,job_len[i] - 3)
        while True:
            temp2 = random.randint(1,job_len[i] - 3)
            if temp1 != temp2:
                break
        rand_pos1 = max(temp1, temp2)
        rand_pos2 = min(temp1, temp2)
        for j in range(rand_pos2):
            child[i][j] = parent1[j]
        for j in range(job_len[i] - 1, rand_pos1 , -1):
            child[i][j] = parent1[j]
        for j in range(job_len[i]):
            if parent2[j] not in child[i] and rand_pos2 <= job_len[i] - 1:
                child[i][rand_pos2] = parent2[j]
                rand_pos2  += 1
    return child

def mutation(child, num_factory, GGA_factory_job_set):
    job_len = [len(GGA_factory_job_set[i]) for i in range(num_factory)]
    for i in range(num_factory):
        temp_individual = [-1 for i in range(job_len[i])]
        temp1 = random.randint(1,job_len[i] - 2)
        while True:
            temp2 = random.randint(1,job_len[i] - 2)
            if abs(temp1 - temp2) >= 2:
                break
        temp_individual = child[i][:]
        rand_pos1 = max(temp1, temp2)
        rand_pos2 = min(temp1, temp2)
        child[i][rand_pos2] = child[i][rand_pos1]
        for j in range(rand_pos2 + 1, rand_pos1+1):
            child[i][j] = temp_individual[j - 1]
    return child

def GGA_select_non_dominated_pop(num_factory, GGA_len_job, GGA_Mat_pop):
    GGA_temp_non_dominated_pop = [[] for k in range(num_factory)]
    for i in range(num_factory):
        for j in range(len(GGA_Mat_pop[i])):
            compare_fitness1 = GGA_Mat_pop[i][j][GGA_len_job[i]]
            compare_fitness2 = GGA_Mat_pop[i][j][GGA_len_job[i]+1]
            b_non_dominated = True
            for k in range(len(GGA_Mat_pop[i])):
                if j != k:
                    if GGA_Mat_pop[i][k][GGA_len_job[i]] <= compare_fitness1 and GGA_Mat_pop[i][k][GGA_len_job[i]+1] <= compare_fitness2:
                        if GGA_Mat_pop[i][k][GGA_len_job[i]] < compare_fitness1 or GGA_Mat_pop[i][k][GGA_len_job[i]+1] < compare_fitness2:
                            b_non_dominated = False
                            break
            if b_non_dominated == True:
                if GGA_Mat_pop[i][j][0:GGA_len_job[i]+2] not in GGA_temp_non_dominated_pop[i]:
                    GGA_temp_non_dominated_pop[i].append(GGA_Mat_pop[i][j][0:GGA_len_job[i]+2])
    return GGA_temp_non_dominated_pop

def GGA_update_non_dominated(GGA_non_dominated_pop, GGA_temp_non_dominated,factory_job_set,num_factory):
    GGA_len_job = [len(factory_job_set[i]) for i in range(num_factory)]
    for i in range(num_factory):
        for j in range(len(GGA_temp_non_dominated[i])):
            if GGA_temp_non_dominated[i][j][0:GGA_len_job[i]+2] not in GGA_non_dominated_pop[i]:
                GGA_non_dominated_pop[i].append(GGA_temp_non_dominated[i][j][0:GGA_len_job[i]+2])
    GGA_non_dominated_pop = GGA_select_non_dominated_pop(num_factory, GGA_len_job, GGA_non_dominated_pop)
    return GGA_non_dominated_pop

def GGA_select_all_f_non_dominated_pop(GGA_temp_all_f_dominated):
    #在所有工厂的帕累托解的组合中找总工厂的帕累托解
    GGA_temp_non_dominated_pop = []
    len_sol = len(GGA_temp_all_f_dominated)
    for i in range(len_sol):
        compare_fitness1 = GGA_temp_all_f_dominated[i][-2]
        compare_fitness2 = GGA_temp_all_f_dominated[i][-1]
        b_non_dominated = True
        for j in range(len_sol):
            if i != j:
                if GGA_temp_all_f_dominated[j][-2] <= compare_fitness1 and GGA_temp_all_f_dominated[j][-1] <= compare_fitness2:
                    if GGA_temp_all_f_dominated[j][-2] < compare_fitness1 or GGA_temp_all_f_dominated[j][-1] < compare_fitness2:
                        b_non_dominated = False
                        break
        if b_non_dominated == True:
            if GGA_temp_all_f_dominated[i] not in GGA_temp_non_dominated_pop:
                GGA_temp_non_dominated_pop.append(GGA_temp_all_f_dominated[i])
    return GGA_temp_non_dominated_pop


def generation_GGA(num_factory,GGA_popsize,GGA_factory_job_set,GGA_Mat_pop,v,num_machine, test_data):
    GGA_len_job = [len(GGA_factory_job_set[i]) for i in range(num_factory)]
    GGA_newpop =[[[0 for i in range(GGA_len_job[k] + 3) ]for j in range(GGA_popsize)] for k in range(num_factory)]
    mutation_prob = 0.6
    for j in range(GGA_popsize):
        GGA_select_1, GGA_select_2 = select(num_machine, num_factory, GGA_popsize, GGA_Mat_pop, GGA_factory_job_set)
        child = crossover(GGA_Mat_pop,num_factory, GGA_select_1, GGA_select_2,GGA_factory_job_set)
        temp_random = np.random.random()
        if temp_random < mutation_prob:
            child = mutation(child, num_factory, GGA_factory_job_set)
        for i in range(num_factory):
            C_time, Energy_consumption = TCE(GGA_len_job[i], num_machine, child[i], test_data,v)
            child[i][GGA_len_job[i]] = C_time
            child[i][GGA_len_job[i]+1] = Energy_consumption
            GGA_newpop[i][j][:] = child[i][:]
    for i in range(num_factory):
        for j in range(GGA_popsize):
            if GGA_Mat_pop[i][j][GGA_len_job[i]]>=GGA_newpop[i][j][GGA_len_job[i]] and GGA_Mat_pop[i][j][GGA_len_job[i]+1]>=GGA_newpop[i][j][GGA_len_job[i]+1]:
                if GGA_Mat_pop[i][j][GGA_len_job[i]]>GGA_newpop[i][j][GGA_len_job[i]] or GGA_Mat_pop[i][j][GGA_len_job[i]+1]>GGA_newpop[i][j][GGA_len_job[i]+1]:
                    GGA_Mat_pop[i][j][:] = GGA_newpop[i][j][:]
            if (GGA_Mat_pop[i][j][GGA_len_job[i]]>GGA_newpop[i][j][GGA_len_job[i]] and GGA_Mat_pop[i][j][GGA_len_job[i]+1]<GGA_newpop[i][j][GGA_len_job[i]+1]) or \
                    (GGA_Mat_pop[i][j][GGA_len_job[i]] < GGA_newpop[i][j][GGA_len_job[i]] and GGA_Mat_pop[i][j][GGA_len_job[i] + 1] > GGA_newpop[i][j][GGA_len_job[i] + 1]):
                b = np.random.random()
                if b < 0.5:
                    GGA_Mat_pop[i][j][:] = GGA_newpop[i][j][:]
    return GGA_Mat_pop

def GGA_main(pop_gen, num_job,num_machine, num_factory,test_data, GGA_popsize, test_time,v):
    test_timeup = time.clock()
    GGA_factory_job_set = NEH2(num_job, num_machine, test_data, num_factory,v)
    GGA_len_job = [len(GGA_factory_job_set[i]) for i in range(num_factory)]
    GGA_Mat_pop, GGA_non_dominated_pop =  Multi_initial_GGA(num_machine, num_factory, GGA_factory_job_set, test_data, GGA_popsize,v)
    for i in range(pop_gen):
        GGA_Mat_pop = generation_GGA(num_factory,GGA_popsize,GGA_factory_job_set,GGA_Mat_pop,v,num_machine, test_data)
        GGA_temp_non_dominated = GGA_select_non_dominated_pop(num_factory, GGA_len_job, GGA_Mat_pop)
        GGA_non_dominated_pop = GGA_update_non_dominated(GGA_non_dominated_pop, GGA_temp_non_dominated, GGA_factory_job_set,num_factory)
        test_timedown = time.clock()
        if float(test_timedown - test_timeup) >= float(test_time):
            break
    return GGA_non_dominated_pop, float(test_timedown - test_timeup)

def G_All_factory_dominated(pop_gen, num_job,num_machine, num_factory,test_data, GGA_popsize, test_time,v):
    #根据每个工厂的帕累托解确定总工厂的解
    GGA_non_dominated_pop,run_time = GGA_main(pop_gen, num_job,num_machine, num_factory,test_data, GGA_popsize, test_time,v)
    print('GGA程序共运行：%s' % (run_time))
    GGA_temp_all_f_dominated = []
    result = []
    temp_set = list(itertools.product(GGA_non_dominated_pop[0],GGA_non_dominated_pop[1]))#lambda x: list(x) for x in GGA_non_dominated_pop[0])
    for individual in temp_set:
        individual = sorted(individual,key=lambda x:x[-2])
        parteo_solution = [0 for i in range(len(individual[-1]))]
        sum_green_fitness = 0
        for indi_green in individual:
            sum_green_fitness += indi_green[-1]
        for i in range(len(individual[-1])):
            parteo_solution[i] = individual[-1][i]
        parteo_solution[-1] = sum_green_fitness
        GGA_temp_all_f_dominated.append(parteo_solution)
    result = GGA_select_all_f_non_dominated_pop(GGA_temp_all_f_dominated)
    return result

#GGA = G_All_factory_dominated(pop_gen, num_job,num_machine, num_factory,test_data, GGA_popsize)