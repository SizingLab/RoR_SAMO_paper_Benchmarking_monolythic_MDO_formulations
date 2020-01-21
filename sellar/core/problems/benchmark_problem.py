from openmdao.api import Problem
import matplotlib.pyplot as plt
import numpy as np

class BenchmarkProblem(object):

    def __init__(self, name='', prob_type='MDO', tol=1.e-8, optimizer='SLSQP', derivative_method='full_analytic', blackbox=False, plot=False, log_file='log_file.txt', scale=1., print=True):

        self.name = name
        self.prob_type = prob_type
        self.tol = tol
        self.optimizer = optimizer
        self.derivative_method = derivative_method
        self.blackbox = blackbox
        self.problem = Problem()
        self.post_analysis_results = {}
        self.plot = plot
        self.log_file = log_file
        self.scale = scale
        self.print = print
        self.optimal_value = 3.18339395

    def initialize_problem(self):
        # To be override by child class
        pass

    def run(self, initial_values={}):

        # Setting initial values
        for i, (variable, value) in enumerate(initial_values.items()):
            self.problem[variable] = value
       
        self.problem.run_driver()

    def post_analysis(self):
        if self.print:
            log_file = open(self.log_file, 'a+')
            s = '---------- Post Analysis ------------ \n'
            print(s)
            log_file.writelines(s)
            log_file.writelines('obj = ' + str(self.problem['obj']) + ', x = ' + \
                                str(self.problem['x']) + ', z1 = ' + str(self.problem['z'][0]) + \
                                ', z2 = ' + str(self.problem['z'][1]) + '\n')
            log_file.close()
        self.post_analysis_results['abs_error'] = self.absolute_error()
        self.post_analysis_results['num_compute'], \
            self.post_analysis_results['num_compute_partials'] = self.number_of_evaluations()
        self.post_analysis_results['obj_convergence'] = self.convergence_plot()
        self.post_analysis_results['success'] = self.check_success()
        if self.print:
            s = '------------------------------------- \n'
            log_file = open(self.log_file, 'a+')
            log_file.writelines(s)
            print(s)
            log_file.close()

    def absolute_error(self):

        optimal_value = self.optimal_value
        computed_optimal_value = self.problem['obj']
        absolute_error = abs(abs(computed_optimal_value) - abs(optimal_value))[0]

        s = '---------- Absolute error ----------- \n' + \
            'Absolute error: ' + str(absolute_error) + '\n' + \
            '------------------------------------- \n'
        if self.print:
            print(s)
            log_file = open(self.log_file, 'a+')
            log_file.writelines(s)
            log_file.close()

        return absolute_error

    def number_of_evaluations(self):
        # Number of counts, -1 due to setup
        if not self.blackbox:
            num_compute = self.problem.model.d1.num_compute - 1
            num_compute_partials = self.problem.model.d1.num_compute_partials
        else:
            num_compute = self.problem.model.problem.model.d1.num_compute - 1
            num_compute_partials = len(self.problem.model.sellar_blackbox.major_iterations) -2


        s = '------- Number of evaluations ------- \n' + \
            'Number of function evaluations: ' + str(num_compute) + '\n' + \
            'Number of derivative evaluations: ' + str(num_compute_partials) + '\n' + \
            '------------------------------------- \n'
        if self.print:
            print(s)
            
            log_file = open(self.log_file, 'a+')
            log_file.writelines(s)
            log_file.close()

        return num_compute, num_compute_partials

    def convergence_plot(self):

        obj_hist = []
        if not self.blackbox:
            obj = self.problem.model.obj_comp
        else:
            obj = self.problem.model.sellar_blackbox

        # Save only major iterations objective values
        for i in obj.obj_hist.keys():
            # COBYLA has no major iterations
            if self.derivative_method == 'derivative_free':
                if i >= 1:
                    obj_hist.append(obj.obj_hist[i])

            else:
                # Case where there are more derivatives calls than major iterations
                if i in obj.major_iterations:
                    obj_hist.append(obj.obj_hist[i])
         
        objective_value = self.optimal_value

        obj_hist = np.array(obj_hist)

        if self.plot == True:
            num_iterations = len(obj_hist)
            
            obj_hist = np.array(obj_hist)

            x = [i for i in range(num_iterations)]
            y = np.absolute((obj_hist - objective_value)) / objective_value
            

            fig1, ax1 = plt.subplots()

            ax1.plot(x, y, '-*', label=self.name)

            ax1.legend()
            ax1.set_xticks(x)
            ax1.set_yscale('log')
            ax1.set_yticks([1.e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])

            plt.xlabel('Iteration number')
            plt.ylabel('log(Relative error)')

            plt.gca().yaxis.grid(True, linestyle=':')

            plt.show()
        
        return obj_hist
    
    def check_success(self):
        # To be override by child class
        pass

    def run_benchmark(self, initial_values):
        if self.print:
            s = '> ---------- Running ' + self.prob_type + ' using ' + self.name + \
                ' formulation and ' + self.optimizer + ' optimizer with ' + \
                self.derivative_method + ' at scale ' + str(self.scale) + ' ------------ \n'
            print(s)
            log_file = open(self.log_file, 'a+')
            log_file.writelines(s)
            log_file.close()
        self.run(initial_values)
        self.post_analysis()
        if self.print:
            s = '< ------------------------------------------- End of ' + self.name \
                + ' ------------------------------------------- \n \n'
            log_file = open(self.log_file, 'a+')
            log_file.writelines(s)  
            log_file.close()
            print(s)