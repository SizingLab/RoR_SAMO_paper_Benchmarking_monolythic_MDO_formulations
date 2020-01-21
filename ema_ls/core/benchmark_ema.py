from os import path, remove
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib
matplotlib.rcParams['text.latex.unicode'] = True
from pyDOE2 import lhs

from ema.core.problems.nvh_problem import NVHProblem
from ema.core.problems.mdf_problem import MDFProblem
from ema.core.problems.idf_problem import IDFProblem
from ema.core.problems.hybrid_problem import HybridProblem


class BenchmarkEMA():

    def run_analysis(self, problems, initial_values={}):
        for prob in problems:

            try:
                 prob.run_benchmark(initial_values)
            except:
                 prob.post_analysis_results['success'] = False
        
        return problems

    def convergence_plot(self, prob_list, file_name='convergence_characteristics'):
        objective_value = 15.385
        fig1, ax1 = plt.subplots()

        for prob in prob_list:
            if prob.name == 'MDF':
                num_iterations = len(prob.post_analysis_results['obj_convergence'])
                mdf_convergence = prob.post_analysis_results['obj_convergence']
                x_mdf = [i for i in range(0, num_iterations)]
                y_mdf = np.absolute((abs(mdf_convergence) - abs(objective_value))) / objective_value
                ax1.plot(x_mdf, y_mdf, 'o-', label='MDF')
            elif prob.name == 'IDF':
                num_iterations = len(prob.post_analysis_results['obj_convergence'])
                idf_convergence = prob.post_analysis_results['obj_convergence'] 
                x_idf = [i for i in range(0, num_iterations)]
                y_idf = np.absolute((abs(idf_convergence) - abs(objective_value))) / objective_value
                ax1.plot(x_idf, y_idf, '-v', label='IDF')
            elif prob.name == 'HYBRID':
                num_iterations = len(prob.post_analysis_results['obj_convergence'])
                hybrid_convergence = prob.post_analysis_results['obj_convergence']
                x_hybrid = [i for i in range(0, num_iterations)]
                y_hybrid = np.absolute((abs(hybrid_convergence) - abs(objective_value))) / objective_value
                ax1.plot(x_hybrid, y_hybrid, '-s', label='HYBRID')
            elif prob.name == 'NVH':
                num_iterations = len(prob.post_analysis_results['obj_convergence'])
                nvh_convergence = prob.post_analysis_results['obj_convergence']
                x_nvh = [i for i in range(0, num_iterations)]
                y_nvh = np.absolute((abs(nvh_convergence) - abs(objective_value))) / objective_value
                ax1.plot(x_nvh, y_nvh, '-*', label='NVH')


        ax1.legend()
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        # if prob.optimizer == 'COBYLA':
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_yticks([1e-7, 1e-6, 1e-5, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])

        plt.xlabel('Major iteration number')
        plt.ylabel('log(Relative error)')

        plt.gca().xaxis.grid(True, linestyle=':')
        plt.gca().yaxis.grid(True, linestyle=':')

        # plt.show()
        plt.savefig('ema/figs/' + file_name + '.pdf')

    def run_optimization_benchmark(self, log_file_name='ema/results/log_file_optimization.txt'):
        if path.exists(log_file_name):
            remove(log_file_name)
        s = '################### Running optimization benchmark ###################'

        log_file = open(log_file_name, 'a+')
        log_file.writelines(s)  
        log_file.close()
        print(s)

        test_cases = [('SLSQP', 'full_analytic', False), ('SLSQP', 'semi_analytic_fd', False)]
        for test_case in test_cases:
            optimizer = test_case[0]
            derivative_method = test_case[1]
            blackbox = test_case[2]
            initial_values = {'F_ema': 7.e4*1.0, 'N_red': 2.0, 'J_mot': 0.0124695, 'T_em': 55.6515}
            prob_list = [
                MDFProblem(name='MDF', optimizer=optimizer, derivative_method=derivative_method,
                           blackbox=blackbox, log_file=log_file_name),
                IDFProblem(name='IDF', optimizer=optimizer, derivative_method=derivative_method,
                           blackbox=blackbox, log_file=log_file_name),
                HybridProblem(name='HYBRID', optimizer=optimizer, derivative_method=derivative_method,
                              blackbox=blackbox, log_file=log_file_name),
                NVHProblem(name='NVH',  optimizer=optimizer, derivative_method=derivative_method,
                            blackbox=blackbox, log_file=log_file_name)
                ]
            self.run_analysis(prob_list, initial_values=initial_values)
            self.convergence_plot(prob_list,
                                  file_name='convergence_characteristics_' + optimizer + '_' + derivative_method)

    def run_initial_values_benchmark(self, log_file_name='ema/results/log_file_initial_values.txt'):
        if path.exists(log_file_name):
            remove(log_file_name)
        s = '################### Running robustness to initial values test ################### \n'

        logfile = open(log_file_name, 'a+')
        logfile.writelines(s)
        logfile.close()
        print(s)

        # Set to true to print detail of each sample
        print_option = False

        test_cases = [('SLSQP', 'full_analytic', False), ('SLSQP', 'semi_analytic_fd', False)]
        for test_case in test_cases:
            optimizer = test_case[0]
            derivative_method = test_case[1]
            blackbox = test_case[2]
            prob_type = 'MDO'

            # Number of samples
            N = 50
            res_num_compute = {
                'MDF': [],
                'IDF': [],
                'HYBRID': [],
                'NVH': []
            }

            # Scale to make sure that a solution exist
            k_os = 1.

            # Bounds of the variables of the DOE
            # lower_bound = 0.1
            # upper_bound = 1.

            variables = {
                'N_red': (0.1, 8.),
                'J_mot': (1e-6, 1.e-1),
                'T_em': (0.1, 100.)
            }

            # DOE
            doe = lhs(len(variables.keys()), samples=N, criterion='center')

            # Perform an analysis for each sample
            for sample in doe:
                initial_values = copy.copy(variables)
                for i, (key, value) in enumerate(initial_values.items()):
                    upper_bound = value[1]
                    lower_bound = value[0]
                    initial_values[key] = (upper_bound - lower_bound) * sample[i] + lower_bound

                prob_list = [
                    MDFProblem(name='MDF', optimizer=optimizer, derivative_method=derivative_method,
                               blackbox=blackbox, log_file=log_file_name, scale=k_os, print=print_option),
                    IDFProblem(name='IDF', optimizer=optimizer, derivative_method=derivative_method,
                               blackbox=blackbox, log_file=log_file_name, scale=k_os, print=print_option),
                    HybridProblem(name='HYBRID', optimizer=optimizer, derivative_method=derivative_method,
                                  blackbox=blackbox, log_file=log_file_name, scale=k_os, print=print_option),
                    NVHProblem(name='NVH', optimizer=optimizer, derivative_method=derivative_method,
                               blackbox=blackbox, log_file=log_file_name, scale=k_os, print=print_option)]

                self.run_analysis(prob_list, initial_values=initial_values)
                
                for prob in prob_list:
                    if prob.post_analysis_results['success']:
                        res_num_compute[prob.name].append(prob.post_analysis_results['num_compute'])
            
            
            # Result analysis
            for i, (key, value) in enumerate(res_num_compute.items()):
                s = '> ---------- Running ' + prob_type + ' using ' + \
                    key + ' formulation and ' + optimizer + ' optimizer with ' + \
                    derivative_method + ' at scale ' + str(k_os) + ' ------------ \n'
                max_num_compute = max(value)
                min_num_compute = min(value)
                mean_num_compute = np.mean(value)
                median_num_compute = np.median(value)
                percentage_of_success = len(value) / N * 100.
                res = 'Max number of evaluations : ' + str(max_num_compute) + '\n' \
                    'Min number of evaluations : ' + str(min_num_compute) + '\n' \
                    'Mean number of evaluations : ' + str(mean_num_compute) + '\n' \
                    'Median number of evaluations : ' + str(median_num_compute) + '\n' \
                    'Percentage of success : ' + str(percentage_of_success) + '\n'
                s += s + res
                logfile = open(log_file_name, 'a+')
                logfile.writelines(s)  
                logfile.close()
                print(s)


    def run_scaling_benchmark(self, log_file_name='ema/results/log_file_scaling.txt'):
        if path.exists(log_file_name):
            remove(log_file_name)
        s = '################### Running robustness to scale change test ###################\n'

        log_file = open(log_file_name, 'a+')
        log_file.writelines(s)  
        log_file.close()
        print(s)

        test_cases = [('SLSQP', 'full_analytic', False), ('SLSQP', 'semi_analytic_fd', False)]
        for test_case in test_cases:
            optimizer = test_case[0]
            derivative_method = test_case[1]
            blackbox = test_case[2]

            k_F_ema = [1., 2., 5., 10.]

            for k_os in k_F_ema:
                initial_values = {'F_ema': 7.e4 * k_os, 'N_red': 2.0, 'J_mot': 0.0124695, 'T_em': 55.6515}
                prob_list = [
                    MDFProblem(name='MDF', optimizer=optimizer, derivative_method=derivative_method,
                               blackbox=blackbox, log_file=log_file_name, scale=k_os),
                    IDFProblem(name='IDF', optimizer=optimizer, derivative_method=derivative_method,
                               blackbox=blackbox, log_file=log_file_name, scale=k_os),
                    HybridProblem(name='HYBRID', optimizer=optimizer, derivative_method=derivative_method,
                                  blackbox=blackbox, log_file=log_file_name, scale=k_os),
                    NVHProblem(name='NVH',  optimizer=optimizer, derivative_method=derivative_method,
                               blackbox=blackbox, log_file=log_file_name, scale=k_os)]
                self.run_analysis(prob_list, initial_values=initial_values)

    def run(self):
        self.run_optimization_benchmark()
        self.run_initial_values_benchmark()
        self.run_scaling_benchmark()
