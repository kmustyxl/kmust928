import matplotlib.pyplot as plt
import Bayes_Multi_object as test_B
import Japan_Multi_object as test_J
import GGA_Multi_object  as test_G
import numpy as np
from AssignRule import *
import xlwt
import xlrd
#import xlutils.copy
import LoadData as ld
book = xlwt.Workbook(encoding = 'utf-8')
sheet = book.add_sheet('data')
num_job_list =[20,30,30]#,50,50,50,70,70,70,100,100]
num_machine_list =[5,5,10]#,5,10,20,5,10,20,10,20]
test_time_list =[30.0,45.0,45.0]#,75.0,75.0,75.0,105.0,105.0,105.0,150.0,150.0]

sheet.write(0, 1, 'Bayes')
sheet.write(0, 6, 'Japan')
sheet.write(0, 11, 'GGA')
sheet.write(1, 1, 'R_N')
sheet.write(1, 2, 'N_N')
sheet.write(1, 3, 'MID')
sheet.write(1, 4, 'SNS')
sheet.write(1, 5, 'RAS')
sheet.write(1, 6, 'R_N')
sheet.write(1, 7, 'N_N')
sheet.write(1, 8, 'MID')
sheet.write(1, 9, 'SNS')
sheet.write(1, 10, 'RAS')
sheet.write(1, 11, 'R_N')
sheet.write(1, 12, 'N_N')
sheet.write(1, 13, 'MID')
sheet.write(1, 14, 'SNS')
sheet.write(1, 15, 'RAS')

def MID(X,Y):
    n = len(X)
    C = 0
    for i in range(n):
        C += np.sqrt(X[i]*X[i]+Y[i]*Y[i])
    mid = C/n
    return mid

def SNS(MID,X,Y):
    n = len(X)
    fenzi = 0
    for i in range(n):
        C = np.sqrt(X[i]*X[i]+Y[i]*Y[i])
        fenzi += (MID - C)*(MID - C)
    sns = np.sqrt(fenzi/(n-1))
    return sns

def RAS(X,Y):
    n = len(X)
    fenzi = 0
    for i in range(n):
        minfit = min(X[i], Y[i])
        fenzi += ((X[i]-minfit)/minfit + (Y[i]-minfit)/minfit)
    ras = fenzi/n
    return ras
for i_index in range(len(num_job_list)):
    num_job = num_job_list[i_index]
    num_machine = num_machine_list[i_index]
    num_factory = 2
    update_popsize = 100
    GGA_popsize = 100
    local_search_size = 100
    ls_frequency = 200
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
    bayes_mid = 0
    bayes_sns = 0
    bayes_ras = 0
    japan_mid = 0
    japan_sns = 0
    japan_ras = 0
    gga_mid = 0
    gga_sns = 0
    gga_ras = 0
    for run_number in range(20):
        test_time = test_time_list[i_index]
        Japan = test_J.J_All_factory_dominated(num_factory,pop_gen,test_time,v,num_job, num_machine, test_data)
        #Japan,timee = test_J.Japan_Multi_object(pop_gen, test_time,v,num_job, num_machine, test_data, num_factory)
        Bayes = test_B.B_All_factory_dominated(pop_gen, ls_frequency,update_popsize, num_factory, test_time,v,block_number,Elite_prob,num_job,num_machine, test_data,local_search_size)
        #Bayes ,timmmm = test_B.Green_Bayes_net(pop_gen, ls_frequency, update_popsize, test_time,v,block_number,Elite_prob,num_job,num_machine, test_data, num_factory,local_search_size)
        GGA = test_G.G_All_factory_dominated(pop_gen, num_job,num_machine, num_factory,test_data, GGA_popsize,test_time,v)
        #GGA,tttt = test_G.GGA_main(pop_gen, num_job,num_machine, num_factory,test_data, GGA_popsize, test_time,v)
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
        bayes_mid_temp = MID(Bayes_x, Bayes_y)
        bayes_mid += bayes_mid_temp
        bayes_sns_temp = SNS(bayes_mid_temp,Bayes_x, Bayes_y)
        bayes_sns += bayes_sns_temp
        bayes_ras_temp = RAS(Bayes_x, Bayes_y)
        bayes_ras += bayes_ras_temp

        japan_mid_temp = MID(Japan_x, Japan_y)
        japan_mid += japan_mid_temp
        japan_sns_temp = SNS(japan_mid_temp, Japan_x, Japan_y)
        japan_sns += japan_sns_temp
        japan_ras_temp = RAS(Japan_x, Japan_y)
        japan_ras += japan_ras_temp

        gga_mid_temp = MID(GGA_x, GGA_y)
        gga_mid += gga_mid_temp
        gga_sns_temp = SNS(gga_mid_temp, GGA_x, GGA_y)
        gga_sns += gga_sns_temp
        gga_ras_temp = RAS(GGA_x, GGA_y)
        gga_ras += gga_ras_temp

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
    result_bayes_mid = bayes_mid/20
    result_bayes_sns = bayes_sns/20
    result_bayes_ras = bayes_ras/20
    result_japan_mid = japan_mid/20
    result_japan_sns = japan_sns/20
    result_japan_ras = japan_ras/20
    result_gga_mid = gga_mid/20
    result_gga_sns = gga_sns/20
    result_gga_ras = gga_ras/20
    sheet.write(i_index + 2, 0, '%s_%s' % (num_job, num_machine))
    sheet.write(i_index + 2, 1, str(result_bayes1))
    sheet.write(i_index + 2, 2, str(result_bayes2))
    sheet.write(i_index + 2, 3, str(result_bayes_mid))
    sheet.write(i_index + 2, 4, str(result_bayes_sns))
    sheet.write(i_index + 2, 5, str(result_bayes_ras))
    sheet.write(i_index + 2, 6, str(result_japan1))
    sheet.write(i_index + 2, 7, str(result_japan2))
    sheet.write(i_index + 2, 8, str(result_japan_mid))
    sheet.write(i_index + 2, 9, str(result_japan_sns))
    sheet.write(i_index + 2, 10, str(result_japan_ras))
    sheet.write(i_index + 2, 11, str(result_gga1))
    sheet.write(i_index + 2, 12, str(result_gga2))
    sheet.write(i_index + 2, 13, str(result_gga_mid))
    sheet.write(i_index + 2, 14, str(result_gga_sns))
    sheet.write(i_index + 2, 15, str(result_gga_ras))
book.save('data\\data.xls')

