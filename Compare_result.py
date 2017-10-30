import matplotlib.pyplot as plt
import Bayes_Multi_object as test_B
import Japan_Multi_object as test_J
import GGA_Multi_object  as test_G
import numpy as np
from AssignRule import *
import xlwt
import LoadData as ld
book = xlwt.Workbook(encoding = 'utf-8')
sheet = book.add_sheet('data')
num_job_list = [20,30,30,50,50,50,70,70,70,100,100]
num_machine_list =[5,5,10,5,10,20,5,10,20,10,20]
test_time_list = [20.0,30.0,30.0,50.0,50.0,50.0,70.0,70.0,140.0,200.0,200.0]

sheet.write(0, 1, 'bayes1')
sheet.write(0, 2, 'bayes2')
sheet.write(0, 3, 'japan1')
sheet.write(0, 4, 'japan2')
sheet.write(0, 5, 'gga1')
sheet.write(0, 6, 'gga2')
for i_index in range(len(num_job_list)):
   # global num_machine,num_job,num_factory,update_popsize,GGA_popsize,local_search_size,ls_frequency,pop_gen,Elite_prob,block_number
    num_job = num_job_list[i_index]
    num_machine = num_machine_list[i_index]
    num_factory = 2
    update_popsize = 200
    GGA_popsize = 200
    local_search_size = 20
    ls_frequency = 10
    pop_gen = 700
    Elite_prob = 0.2
    block_number = 3
    V = [1, 1.1, 1.2, 1.3, 1.4]
    v = np.zeros((num_job, num_machine))
    for iii in range(num_machine):
        for jjj in range(num_job):
            temp = choice(V)
            v[jjj][iii] = temp
    test_data = ld.LoadData(num_job, num_machine)
    bayes1 = 0
    bayes2 = 0
    japan1 = 0
    japan2 = 0
    gga1 = 0
    gga2 = 0
    for run_number in range(20):
        #time1 = time.clock()
        test_time = test_time_list[i_index]
        Japan = test_J.J_All_factory_dominated(num_factory,pop_gen,test_time,v,num_job, num_machine, test_data)
       # time2 = time.clock()
       # print('日本人程序共运行：%s'%(time2-time1))
       # time1 = time.clock()
        Bayes = test_B.B_All_factory_dominated(pop_gen, ls_frequency,update_popsize, num_factory, test_time,v,block_number,Elite_prob,num_job,num_machine, test_data,local_search_size)
       # time2 = time.clock()
       # print('贝叶斯程序共运行：%s'%(time2-time1))
       # time1 = time.clock()
        GGA = test_G.G_All_factory_dominated(pop_gen, num_job,num_machine, num_factory,test_data, GGA_popsize,test_time,v)
       # time2 = time.clock()
       # print('GGA程序共运行：%s'%(time2-time1))
        Japan_x = []
        Japan_y = []
        Bayes_x = []
        Bayes_y = []
        GGA_x = []
        GGA_y = []
        all_algorithm_set_x = []
        all_algorithm_set_y = []
        for individual in Japan:
            Japan_x.append(individual[-2])
            Japan_y.append(individual[-1])
            all_algorithm_set_x.append(individual[-2])
            all_algorithm_set_y.append(individual[-1])
        for individual in Bayes:
            Bayes_x.append(individual[-2])
            Bayes_y.append(individual[-1])
            all_algorithm_set_x.append(individual[-2])
            all_algorithm_set_y.append(individual[-1])
        for individual in GGA:
            GGA_x.append(individual[-2])
            GGA_y.append(individual[-1])
            all_algorithm_set_x.append(individual[-2])
            all_algorithm_set_y.append(individual[-1])
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_xlabel('Fitness')
        # ax.set_ylabel('Carbon')
        # ax.plot(Japan_x, Japan_y, 'rD',label='Japan')
        # ax.plot(Bayes_x, Bayes_y, 'gD',label='Bayes')
        # ax.plot(GGA_x, GGA_y,'yD',label='GGA')
        # plt.grid()
        # plt.legend()
        # plt.show()
        # bayes1 = 0
        # bayes2 = 0
        # japan1 = 0
        # japan2 = 0
        # gga1 = 0
        # gga2 = 0
        num_Japan_sol = len(Japan_x)
        num_Bayes_sol = len(Bayes_x)
        num_GGA_sol = len(GGA_x)
        num_all_sol = len(all_algorithm_set_x)
        num_Bayes_pareto = 0
        num_Japan_pareto = 0
        num_GGA_pareto = 0
        for i in range(num_Bayes_sol):
            fitness1 = Bayes_x[i]
            fitness2 = Bayes_y[i]
            for j in range(num_all_sol):
                if all_algorithm_set_x[j] <= fitness1 and all_algorithm_set_y[j] <= fitness2:
                    if all_algorithm_set_x[j] < fitness1 or all_algorithm_set_y[j]< fitness2:
                        num_Bayes_pareto += 1
                        break
        for i in range(num_Japan_sol):
            fitness1 = Japan_x[i]
            fitness2 = Japan_y[i]
            for j in range(num_all_sol):
                if all_algorithm_set_x[j] <= fitness1 and all_algorithm_set_y[j] <= fitness2:
                    if all_algorithm_set_x[j] < fitness1 or all_algorithm_set_y[j] < fitness2:
                        num_Japan_pareto += 1
                        break
        for i in range(num_GGA_sol):
            fitness1 = GGA_x[i]
            fitness2 = GGA_y[i]
            for j in range(num_all_sol):
                if all_algorithm_set_x[j] <= fitness1 and all_algorithm_set_y[j] <= fitness2:
                    if all_algorithm_set_x[j] < fitness1 or all_algorithm_set_y[j] < fitness2:
                        num_GGA_pareto += 1
                        break
        bayes1+=((num_Bayes_sol-num_Bayes_pareto)/num_Bayes_sol)
        bayes2+=(num_Bayes_sol-num_Bayes_pareto)
        japan1+=((num_Japan_sol-num_Japan_pareto)/num_Japan_sol)
        japan2+=(num_Japan_sol-num_Japan_pareto)
        gga1+=((num_GGA_sol-num_GGA_pareto)/num_GGA_sol)
        gga2+=(num_GGA_sol-num_GGA_pareto)
        print('Bayes评价指标1： %.2f'%((num_Bayes_sol-num_Bayes_pareto)/num_Bayes_sol))
        print('Bayes评价指标2： %s'%(num_Bayes_sol-num_Bayes_pareto))
        print('Japan评价指标1： %.2f'%((num_Japan_sol-num_Japan_pareto)/num_Japan_sol))
        print('Japan评价指标2： %s'%(num_Japan_sol-num_Japan_pareto))
        print('GGA评价指标1： %.2f'%((num_GGA_sol-num_GGA_pareto)/num_GGA_sol))
        print('GGA评价指标2： %s'%(num_GGA_sol-num_GGA_pareto))
    result_bayes1 = bayes1/20
    result_bayes2 = bayes2/20
    result_japan1 = japan1/20
    result_japan2 = japan2/20
    result_gga1 = gga1/20
    result_gga2 = gga2/20
    sheet.write(i_index + 1, 0, '%s_%s' % (num_job, num_machine))
    sheet.write(i_index + 1, 1, str(result_bayes1))
    sheet.write(i_index + 1, 2, str(result_bayes2))
    sheet.write(i_index + 1, 3, str(result_japan1))
    sheet.write(i_index + 1, 4, str(result_japan2))
    sheet.write(i_index + 1, 5, str(result_gga1))
    sheet.write(i_index + 1, 6, str(result_gga2))
    book.save('data\\data.xls')

