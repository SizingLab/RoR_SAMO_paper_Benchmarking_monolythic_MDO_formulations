from os import path, remove

import numpy as np
import matplotlib
matplotlib.rcParams['text.latex.unicode'] = True
from pyDOE2 import lhs

from sellar.core.problems.nvh_problem import NVHProblem
from sellar.core.problems.mdf_problem import MDFProblem
from sellar.core.problems.idf_problem import IDFProblem
from sellar.core.problems.hybrid_problem import HybridProblem


class BenchmarkSellar():

    def run_analysis(self, problems, initial_values={}):
        for prob in problems:
            prob.run_benchmark(initial_values)

        return problems

    def run_optimization_benchmark(self, log_file_name='sellar/results/log_file_optimization.txt'):
        if path.exists(log_file_name):
            remove(log_file_name)
        s = '################### Running optimization benchmark ###################'

        log_file = open(log_file_name, 'a+')
        log_file.writelines(s)
        log_file.close()
        print(s)

        test_cases = [('SLSQP', 'full_analytic', False), ('SLSQP', 'semi_analytic_fd', False),
                      ('SLSQP', 'monolythic_fd', True), ('COBYLA', 'derivative_free', False)]
        for test_case in test_cases:
            optimizer = test_case[0]
            derivative_method = test_case[1]
            blackbox = test_case[2]
            initial_values = {'x': 5.0, 'z': np.array([5.0, 2.0]), 'y1': 0.0, 'y1': 0.0}
            prob_list = [
                MDFProblem(name='MDF', optimizer=optimizer, derivative_method=derivative_method,
                           blackbox=blackbox, log_file=log_file_name),
                IDFProblem(name='IDF', optimizer=optimizer, derivative_method=derivative_method,
                           blackbox=blackbox, log_file=log_file_name),
                HybridProblem(name='HYBRID', optimizer=optimizer, derivative_method=derivative_method,
                              blackbox=blackbox, log_file=log_file_name),
                NVHProblem(name='NVH', optimizer=optimizer, derivative_method=derivative_method,
                           blackbox=blackbox, log_file=log_file_name)]
            self.run_analysis(prob_list, initial_values=initial_values)

    def run_initial_values_benchmark(self, log_file_name='sellar/results/log_file_initial_values.txt'):
        if path.exists(log_file_name):
            remove(log_file_name)
        s = '################### Running robustness to initial values test ################### \n'

        logfile = open(log_file_name, 'a+')
        logfile.writelines(s)
        logfile.close()
        print(s)

        test_cases = [('SLSQP', 'full_analytic', False), ('SLSQP', 'semi_analytic_fd', False),
                      ('SLSQP', 'monolythic_fd', True), ('COBYLA', 'derivative_free', False)]
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

            variables = {
                'x': (0., 10.),
                'z1': (-10.0, 10.),
                'z2': (0., 10.)
            }

            # DOE
            doe = lhs(len(variables.keys()), samples=N, criterion='center')

            initial_values = {
                'x': (0., 10.),
                'z': np.zeros(2)
            }
            # Perform an analysis for each sample
            for sample in doe:
                lower_bound = variables['x'][0]
                upper_bound = variables['x'][1]
                initial_values['x'] = (upper_bound - lower_bound) * sample[0] + lower_bound

                lower_bound = variables['z1'][0]
                upper_bound = variables['z1'][1]
                z1 = (upper_bound - lower_bound) * sample[1] + lower_bound

                lower_bound = variables['z2'][0]
                upper_bound = variables['z2'][1]
                z2 = (upper_bound - lower_bound) * sample[2] + lower_bound

                initial_values['z'] = np.array([z1, z2])

                prob_list = [
                    MDFProblem(name='MDF', optimizer=optimizer, derivative_method=derivative_method,
                               blackbox=blackbox, log_file=log_file_name, print=False),
                    IDFProblem(name='IDF', optimizer=optimizer, derivative_method=derivative_method,
                               blackbox=blackbox, log_file=log_file_name, print=False),
                    HybridProblem(name='HYBRID', optimizer=optimizer, derivative_method=derivative_method,
                                  blackbox=blackbox, log_file=log_file_name, print=False),
                    NVHProblem(name='NVH', optimizer=optimizer, derivative_method=derivative_method,
                               blackbox=blackbox, log_file=log_file_name, print=False)]

                self.run_analysis(prob_list, initial_values=initial_values)

                for prob in prob_list:
                    if prob.post_analysis_results['success']:
                        res_num_compute[prob.name].append(prob.post_analysis_results['num_compute'])

            # Result analysis
            for i, (key, value) in enumerate(res_num_compute.items()):
                s = '> ---------- Running ' + prob_type + ' using ' + key + ' formulation and ' + \
                    optimizer + ' optimizer with ' + derivative_method + ' ------------ \n'
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

    def run(self):
        self.run_optimization_benchmark()
        self.run_initial_values_benchmark()
