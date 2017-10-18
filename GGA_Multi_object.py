from AssignRule import *
from Bayes_Multi_object import *
from Japan_Multi_object import *

factory_job_set =  NEH2(num_job, num_machine, test_data, num_factory)

def Multi_initial_GGA(num_machine, num_factory, factory_job_set, test_data, GGA_popsize):
    #每个工厂的工件数
    len_job = [len(factory_job_set[i]) for i in range(num_factory)]
    #每个工厂的初始数据
    Mat_pop =[[[0 for i in range(len_job[k] + 3) ]for j in range(GGA_popsize)] for k in range(num_factory)]
    non_dominated_pop = [[] for k in range(num_factory)]
    for i in range(num_factory):
        for j in range(GGA_popsize):
            sort = random.sample(factory_job_set[i], len_job[i])
            for k in range(len_job[i] + 1):
                if k == len_job[i]:
                    Mat_pop[i][j][k], Mat_pop[i][j][k+1]= TCE(len_job[i], num_machine, sort, test_data,v)
                else:
                    Mat_pop[i][j][k] = sort[k]
        Mat_pop[i] = sorted(Mat_pop[i], key= lambda x:x[-2])
        for j in range(GGA_popsize):
            compare_fitness1 = Mat_pop[i][j][len_job[i]]
            compare_fitness2 = Mat_pop[i][j][len_job[i]+1]
            b_non_dominated = True
            for k in range(GGA_popsize):
                if j != k:
                    if Mat_pop[i][k][len_job[i]] <= compare_fitness1 and Mat_pop[i][k][len_job[i]+1] <= compare_fitness2:
                        if Mat_pop[i][k][len_job[i]] < compare_fitness1 or Mat_pop[i][k][len_job[i]+1] < compare_fitness2:
                            b_non_dominated = False
                            break
            if b_non_dominated == True:
                if Mat_pop[i][j] not in non_dominated_pop[i]:
                    non_dominated_pop[i].append(Mat_pop[i][j])

    return Mat_pop, non_dominated_pop




def single_fitness(num_machine, num_factory, GGA_popsize, Mat_pop, factory_job_set):
    s = [0 for i in range(GGA_popsize)]
    r = [0 for i in range(GGA_popsize)]
    d = [0 for i in range(GGA_popsize)]
    sum1 = [0 for i in range(GGA_popsize)]
    sigma = [[0 for i in range(GGA_popsize)] for j in range(GGA_popsize)]
    len_job = [len(factory_job_set[i]) for i in range(num_factory)]
    for i in range(num_factory):
        for j in range(GGA_popsize):
            for k in range(GGA_popsize):
                if j != k:
                    if Mat_pop[i][j][len_job[i]] <= Mat_pop[i][k][len_job[i]] and \
                       Mat_pop[i][j][len_job[i]+1] <= Mat_pop[i][k][len_job[i]+1]:
                        if Mat_pop[i][j][len_job[i]] < Mat_pop[i][k][len_job[i]] or \
                           Mat_pop[i][j][len_job[i] + 1] < Mat_pop[i][k][len_job[i] + 1]:
                            s[j] += 1
        for k in range(GGA_popsize):
            for j in range(GGA_popsize):
                if j != k:
                    if Mat_pop[i][j][len_job[i]] <= Mat_pop[i][k][len_job[i]] and \
                       Mat_pop[i][j][len_job[i]+1] <= Mat_pop[i][k][len_job[i]+1]:
                        if Mat_pop[i][j][len_job[i]] < Mat_pop[i][k][len_job[i]] or \
                           Mat_pop[i][j][len_job[i] + 1] < Mat_pop[i][k][len_job[i] + 1]:
                            r[k] = r[k] + s[j]

        for j in range(GGA_popsize):
            for k in range(GGA_popsize):
                if j != k:
                    sigma[j][k] = np.sqrt((Mat_pop[i][j][len_job[i]] - Mat_pop[i][k][len_job[i]])^2+ \
                                          (Mat_pop[i][j][len_job[i]+1] - Mat_pop[i][k][len_job[i]+1]) ^ 2)
            sigma[j] = np.sort(sigma[j])
            kk = int(np.sqrt(GGA_popsize))
            for k in range(kk):
                sum1[j] = sum1[j] + sigma[j][k]
            d[j] = 1/(sum1[j] + 2)
            Mat_pop[i][j][len_job[i]+2] = r[j] + d[j]
    return Mat_pop

def select(num_machine, num_factory, GGA_popsize, Mat_pop, factory_job_set):
    len_job = [len(factory_job_set[i]) for i in range(num_factory)]
    Mat_pop1 =[[[0 for i in range(len_job[k] + 3) ]for j in range(GGA_popsize)] for k in range(num_factory)]
    select_1 = [0 for k in range(num_factory)]
    select_2 = [0 for k in range(num_factory)]
    for i in range(num_factory):
        for j in range(GGA_popsize):
            for k in range(len_job[i]+3):
                Mat_pop1[i][j][k] = Mat_pop[i][j][k]
    for i in range(num_factory):
        Mat_pop1[i] = sorted(Mat_pop1[i], key= lambda x:x[-1])
    for i in range(num_factory):
        u = np.random.randint(int(GGA_popsize/2+1))
        v = np.random.randint(int(GGA_popsize/2+1))
        while u == v:
            u = np.random.randint(int(GGA_popsize / 2 + 1))
            v = np.random.randint(int(GGA_popsize / 2 + 1))
        if Mat_pop1[i][u][len_job[i]+2] < Mat_pop1[i][v][len_job[i]+2]:
            select_1[i] = u
        elif Mat_pop1[i][u][len_job[i]+2] > Mat_pop1[i][v][len_job[i]+2]:
            select_1[i] = v
        else:
            b = np.random.random()
            if b < 0.5:
                select_1[i] = u
            else:
                select_1[i] = v
        b_label = True
        while b_label:
            u = np.random.randint(int(GGA_popsize / 2 + 1))
            v = np.random.randint(int(GGA_popsize / 2 + 1))
            while u == v:
                u = np.random.randint(int(GGA_popsize / 2 + 1))
                v = np.random.randint(int(GGA_popsize / 2 + 1))
            if Mat_pop1[i][u][len_job[i] + 2] < Mat_pop1[i][v][len_job[i] + 2]:
                select_2[i] = u
            elif Mat_pop1[i][u][len_job[i] + 2] > Mat_pop1[i][v][len_job[i] + 2]:
                select_2[i] = v
            else:
                b = np.random.random()
                if b < 0.5:
                    select_2[i] = u
                else:
                    select_2[i] = v
            if select_1[i] == select_2[i]:
                continue
            else:
                b_label = False
    return select_1, select_2

def crossover(Mat_pop,num_factory, select_1, select_2,factory_job_set):
    job_len = [len(factory_job_set[i]) for i in range(num_factory)]
    child = [[0 for i in range(job_len[k]+3)] for k in range(num_factory)]
    for i in range(num_factory):
        parent1 = [0 for k in range(job_len[i] + 3)]
        parent2 = [0 for k in range(job_len[i] + 3)]
        parent1[:] = Mat_pop[i][select_1[i]][:]
        parent2[:] = Mat_pop[i][select_2[i]][:]
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

def mutation(child, num_factory, factory_job_set):
    job_len = [len(factory_job_set[i]) for i in range(num_factory)]
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



def generation_GGA(num_factory,GGA_popsize,factory_job_set,Mat_pop):
    len_job = [len(factory_job_set[i]) for i in range(num_factory)]
    newpop =[[[0 for i in range(len_job[k] + 3) ]for j in range(GGA_popsize)] for k in range(num_factory)]
    mutation_prob = 0.6
    for j in range(GGA_popsize):
        select_1, select_2 = select(num_machine, num_factory, GGA_popsize, Mat_pop, factory_job_set)
        child = crossover(Mat_pop,num_factory, select_1, select_2,factory_job_set)
        temp_random = np.random.random()
        if temp_random < mutation_prob:
            child = mutation(child, num_factory, factory_job_set)
        for i in range(num_factory):
            C_time, Energy_consumption = TCE(len_job[i], num_machine, child[i], test_data,v)
            child[i][len_job[i]] = C_time
            child[i][len_job[i]+1] = Energy_consumption
            newpop[i][j][:] = child[i][:]
    for i in range(num_factory):
        for j in range(GGA_popsize):
            if Mat_pop[i][j][len_job[i]]>=newpop[i][j][len_job[i]] and Mat_pop[i][j][len_job[i]+1]>=newpop[i][j][len_job[i]+1]:
                if Mat_pop[i][j][len_job[i]]>newpop[i][j][len_job[i]] or Mat_pop[i][j][len_job[i]+1]>newpop[i][j][len_job[i]+1]:
                    Mat_pop[i][j][:] = newpop[i][j][:]
            if Mat_pop[i][j][len_job[i]]>newpop[i][j][len_job[i]] and Mat_pop[i][j][len_job[i]+1]<newpop[i][j][len_job[i]+1] or \
               Mat_pop[i][j][len_job[i]] < newpop[i][j][len_job[i]] and Mat_pop[i][j][len_job[i] + 1] > newpop[i][j][len_job[i] + 1]:
                b = np.random.random()
                if b < 0.5:
                    Mat_pop[i][j][:] = newpop[i][j][:]
    return Mat_pop

def GGA_main(pop_gen, num_job,num_machine, num_factory,test_data, GGA_popsize):
    factory_job_set = NEH2(num_job, num_machine, test_data, num_factory)
    len_job = [len(factory_job_set[i]) for i in range(num_factory)]
    Mat_pop, non_dominated_pop =  Multi_initial_GGA(num_machine, num_factory, factory_job_set, test_data, GGA_popsize)
    for i in range(pop_gen):
        Mat_pop = generation_GGA(num_factory,GGA_popsize,factory_job_set,Mat_pop)
        temp_non_dominated = select_non_dominated_pop(num_factory, len_job, Mat_pop)
        non_dominated_pop = update_non_dominated(non_dominated_pop, temp_non_dominated, factory_job_set)
    return non_dominated_pop

#dedd = GGA_main(pop_gen, num_job,num_machine, num_factory,test_data, GGA_popsize)