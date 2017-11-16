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

def single_fitness(num_machine, num_factory, GGA_popsize, GGA_Mat_pop, num_job):
    s = [0 for i in range(GGA_popsize)]
    r = [0 for i in range(GGA_popsize)]
    d = [0 for i in range(GGA_popsize)]
    GGA_sum1 = [0 for i in range(GGA_popsize)]
    GGA_sigma = [[0 for i in range(GGA_popsize)] for j in range(GGA_popsize)]
    for j in range(GGA_popsize):
        for k in range(GGA_popsize):
            if j != k:
                if GGA_Mat_pop[j][num_job] <= GGA_Mat_pop[k][num_job] and \
                   GGA_Mat_pop[j][num_job+1] <= GGA_Mat_pop[k][num_job+1]:
                    if GGA_Mat_pop[j][num_job] < GGA_Mat_pop[k][num_job] or \
                       GGA_Mat_pop[j][num_job + 1] < GGA_Mat_pop[k][num_job + 1]:
                        s[j] += 1
    for k in range(GGA_popsize):
        for j in range(GGA_popsize):
            if j != k:
                if GGA_Mat_pop[j][num_job] <= GGA_Mat_pop[k][num_job] and \
                   GGA_Mat_pop[j][num_job+1] <= GGA_Mat_pop[k][num_job+1]:
                    if GGA_Mat_pop[j][num_job] < GGA_Mat_pop[k][num_job] or \
                       GGA_Mat_pop[j][num_job + 1] < GGA_Mat_pop[k][num_job + 1]:
                        r[k] = r[k] + s[j]

    for j in range(GGA_popsize):
        for k in range(GGA_popsize):
            if j != k:
                GGA_sigma[j][k] = np.sqrt((GGA_Mat_pop[j][num_job] - GGA_Mat_pop[k][num_job])*(GGA_Mat_pop[j][num_job] - GGA_Mat_pop[k][num_job])+ \
                                      (GGA_Mat_pop[j][num_job+1] - GGA_Mat_pop[k][num_job+1]) * (GGA_Mat_pop[j][num_job+1] - GGA_Mat_pop[k][num_job+1]))
        GGA_sigma[j] = np.sort(GGA_sigma[j])
        kk = int(np.sqrt(GGA_popsize))
        for k in range(kk):
            GGA_sum1[j] = GGA_sum1[j] + GGA_sigma[j][k]
        d[j] = 1/(GGA_sum1[j] + 2)
        GGA_Mat_pop[j][-1] = r[j] + d[j]
    return GGA_Mat_pop

def Multi_initial_GGA(num_machine, num_factory, num_job, test_data, GGA_popsize,v):
    job_set = [i for i in range(num_job)]
    GGA_Mat_pop =[[0 for i in range(num_job + 3) ]for j in range(GGA_popsize)]
    GGA_non_dominated_pop = []
    for j in range(GGA_popsize):
        GGA_sort = random.sample(job_set, num_job)
        for k in range(num_job + 1):
            if k == num_job:
                GGA_Mat_pop[j][k], GGA_Mat_pop[j][k+1]= TCE(num_job, num_machine, test_data,v, num_factory, GGA_sort)
            else:
                GGA_Mat_pop[j][k] = GGA_sort[k]
    for j in range(GGA_popsize):
        GGA_compare_fitness1 = GGA_Mat_pop[j][num_job]
        GGA_compare_fitness2 = GGA_Mat_pop[j][num_job+1]
        b_non_dominated = True
        for k in range(GGA_popsize):
            if j != k:
                if GGA_Mat_pop[k][num_job] <= GGA_compare_fitness1 and GGA_Mat_pop[k][num_job+1] <= GGA_compare_fitness2:
                    if GGA_Mat_pop[k][num_job] < GGA_compare_fitness1 or GGA_Mat_pop[k][num_job+1] < GGA_compare_fitness2:
                        b_non_dominated = False
                        break
        if b_non_dominated == True:
            if GGA_Mat_pop[j][0:num_job+2] not in GGA_non_dominated_pop:
                GGA_non_dominated_pop.append(GGA_Mat_pop[j][0:num_job+2])
    GGA_Mat_pop = single_fitness(num_machine, num_factory, GGA_popsize, GGA_Mat_pop, num_job)
    return GGA_Mat_pop, GGA_non_dominated_pop


def select(num_machine, num_factory, GGA_popsize, GGA_Mat_pop, num_job):
    #GGA_len_job = [len(GGA_factory_job_set[i]) for i in range(num_factory)]
    GGA_Mat_pop1 =[[0 for i in range(num_job + 3) ]for j in range(GGA_popsize)]
    GGA_select_1 = 0
    GGA_select_2 = 0
    #for i in range(num_factory):
    for j in range(GGA_popsize):
        for k in range(num_job+3):
            GGA_Mat_pop1[j][k] = GGA_Mat_pop[j][k]
    #for i in range(num_factory):
    GGA_Mat_pop1 = sorted(GGA_Mat_pop1, key= lambda x:x[-1])
    #for i in range(num_factory):
    u = np.random.randint(int(GGA_popsize/2+1))
    v = np.random.randint(int(GGA_popsize/2+1))
    while u == v:
        u = np.random.randint(int(GGA_popsize / 2 + 1))
        v = np.random.randint(int(GGA_popsize / 2 + 1))
    if GGA_Mat_pop1[u][num_job+2] < GGA_Mat_pop1[v][num_job+2]:
        GGA_select_1 = u
    elif GGA_Mat_pop1[u][num_job+2] > GGA_Mat_pop1[v][num_job+2]:
        GGA_select_1 = v
    else:
        b = np.random.random()
        if b < 0.5:
            GGA_select_1 = u
        else:
            GGA_select_1 = v
    b_label = True
    while b_label:
        u = np.random.randint(int(GGA_popsize / 2 + 1))
        v = np.random.randint(int(GGA_popsize / 2 + 1))
        while u == v:
            u = np.random.randint(int(GGA_popsize / 2 + 1))
            v = np.random.randint(int(GGA_popsize / 2 + 1))
        if GGA_Mat_pop1[u][num_job + 2] < GGA_Mat_pop1[v][num_job + 2]:
            GGA_select_2 = u
        elif GGA_Mat_pop1[u][num_job + 2] > GGA_Mat_pop1[v][num_job + 2]:
            GGA_select_2 = v
        else:
            b = np.random.random()
            if b < 0.5:
                GGA_select_2 = u
            else:
                GGA_select_2 = v
        if GGA_select_1 == GGA_select_2:
            continue
        else:
            b_label = False
    return GGA_select_1, GGA_select_2

def crossover(GGA_Mat_pop,num_factory, GGA_select_1, GGA_select_2, num_job):
    #job_len = [len(GGA_factory_job_set[i]) for i in range(num_factory)]
    child = [0 for i in range(num_job+3)]
    #for i in range(num_factory):
    parent1 = [0 for k in range(num_job + 3)]
    parent2 = [0 for k in range(num_job + 3)]
    parent1[:] = GGA_Mat_pop[GGA_select_1][:]
    parent2[:] = GGA_Mat_pop[GGA_select_2][:]
    temp1 = random.randint(1,num_job - 3)
    while True:
        temp2 = random.randint(1,num_job - 3)
        if temp1 != temp2:
            break
    rand_pos1 = max(temp1, temp2)
    rand_pos2 = min(temp1, temp2)
    for j in range(rand_pos2):
        child[j] = parent1[j]
    for j in range(num_job - 1, rand_pos1 , -1):
        child[j] = parent1[j]
    for j in range(num_job):
        if parent2[j] not in child and rand_pos2 <= num_job - 1:
            child[rand_pos2] = parent2[j]
            rand_pos2  += 1
    return child

def mutation(child, num_factory, num_job):
    #for i in range(num_factory):
    temp_individual = [-1 for i in range(num_job)]
    temp1 = random.randint(1,num_job - 2)
    while True:
        temp2 = random.randint(1,num_job - 2)
        if abs(temp1 - temp2) >= 2:
            break
    temp_individual = child[:]
    rand_pos1 = max(temp1, temp2)
    rand_pos2 = min(temp1, temp2)
    child[rand_pos2] = child[rand_pos1]
    for j in range(rand_pos2 + 1, rand_pos1+1):
        child[j] = temp_individual[j - 1]
    return child

def GGA_select_non_dominated_pop(num_factory, num_job, GGA_Mat_pop):
    GGA_temp_non_dominated_pop = []
    #for i in range(num_factory):
    for j in range(len(GGA_Mat_pop)):
        compare_fitness1 = GGA_Mat_pop[j][num_job]
        compare_fitness2 = GGA_Mat_pop[j][num_job+1]
        b_non_dominated = True
        for k in range(len(GGA_Mat_pop)):
            if j != k:
                if GGA_Mat_pop[k][num_job] <= compare_fitness1 and GGA_Mat_pop[k][num_job+1] <= compare_fitness2:
                    if GGA_Mat_pop[k][num_job] < compare_fitness1 or GGA_Mat_pop[k][num_job+1] < compare_fitness2:
                        b_non_dominated = False
                        break
        if b_non_dominated == True:
            if GGA_Mat_pop[j][0:num_job+2] not in GGA_temp_non_dominated_pop:
                GGA_temp_non_dominated_pop.append(GGA_Mat_pop[j][0:num_job+2])
    return GGA_temp_non_dominated_pop

def GGA_update_non_dominated(GGA_non_dominated_pop, GGA_temp_non_dominated,num_job,num_factory):
    #GGA_len_job = [len(factory_job_set[i]) for i in range(num_factory)]
    #for i in range(num_factory):
    for j in range(len(GGA_temp_non_dominated)):
        if GGA_temp_non_dominated[j][0:num_job+2] not in GGA_non_dominated_pop:
            GGA_non_dominated_pop.append(GGA_temp_non_dominated[j][0:num_job+2])
    GGA_non_dominated_pop = GGA_select_non_dominated_pop(num_factory, num_job, GGA_non_dominated_pop)
    return GGA_non_dominated_pop

def generation_GGA(num_factory,GGA_popsize,num_job,GGA_Mat_pop,v,num_machine, test_data):
    GGA_newpop =[[0 for i in range(num_job + 3) ]for j in range(GGA_popsize)]
    mutation_prob = 0.6
    for j in range(GGA_popsize):
        GGA_select_1, GGA_select_2 = select(num_machine, num_factory, GGA_popsize, GGA_Mat_pop, num_job)
        child = crossover(GGA_Mat_pop,num_factory, GGA_select_1, GGA_select_2,num_job)
        temp_random = np.random.random()
        if temp_random < mutation_prob:
            child = mutation(child, num_factory, num_job)
        #for i in range(num_factory):
        C_time, Energy_consumption = TCE(num_job, num_machine, test_data,v, num_factory, child)
        child[num_job] = C_time
        child[num_job+1] = Energy_consumption
        GGA_newpop[j][:] = child[:]
    #for i in range(num_factory):
    for j in range(GGA_popsize):
        if GGA_Mat_pop[j][num_job]>=GGA_newpop[j][num_job] and GGA_Mat_pop[j][num_job+1]>=GGA_newpop[j][num_job+1]:
            if GGA_Mat_pop[j][num_job]>GGA_newpop[j][num_job] or GGA_Mat_pop[j][num_job+1]>GGA_newpop[j][num_job+1]:
                GGA_Mat_pop[j][:] = GGA_newpop[j][:]
        if (GGA_Mat_pop[j][num_job]>GGA_newpop[j][num_job] and GGA_Mat_pop[j][num_job+1]<GGA_newpop[j][num_job+1]) or \
                (GGA_Mat_pop[j][num_job] < GGA_newpop[j][num_job] and GGA_Mat_pop[j][num_job + 1] > GGA_newpop[j][num_job + 1]):
            b = np.random.random()
            if b < 0.5:
                GGA_Mat_pop[j][:] = GGA_newpop[j][:]
    return GGA_Mat_pop

def GGA_main(pop_gen, num_job,num_machine, num_factory,test_data, GGA_popsize, test_time,v):
    test_timeup = time.clock()
    #GGA_factory_job_set = NEH2(num_job, num_machine, test_data, num_factory,v)
    GGA_Mat_pop, GGA_non_dominated_pop =  Multi_initial_GGA(num_machine, num_factory, num_job, test_data, GGA_popsize,v)
    for i in range(pop_gen):
        GGA_Mat_pop = generation_GGA(num_factory,GGA_popsize,num_job,GGA_Mat_pop,v,num_machine, test_data)
        GGA_temp_non_dominated = GGA_select_non_dominated_pop(num_factory, num_job, GGA_Mat_pop)
        GGA_non_dominated_pop = GGA_update_non_dominated(GGA_non_dominated_pop, GGA_temp_non_dominated, num_job,num_factory)
        test_timedown = time.clock()
        if float(test_timedown - test_timeup) >= float(test_time):
            break
    return GGA_non_dominated_pop#, float(test_timedown - test_timeup))