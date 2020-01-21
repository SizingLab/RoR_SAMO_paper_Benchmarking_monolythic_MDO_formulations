from openmdao.api import Problem
import matplotlib.pyplot as plt
import numpy as np


class BenchmarkProblem(object):

    def __init__(self, name='', prob_type='MDO', tol=1.e-6, optimizer='SLSQP',
                 derivative_method='full_analytic', blackbox=False, plot=False,
                 log_file='log_file.txt', scale=1., print=True,
                 N=100, t=np.linspace(0., 1.5, num=1000)):

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
        self.optimal_value = 15.381369 # 15.388218 # 19.5091407
        self.N = N
        self.t = t

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
            log_file.writelines('M_mot = ' + str(self.problem['M_mot']) + ', N_red = ' + str(
                self.problem['N_red']) + ', W_mot_constr = ' + str(np.max(self.problem['W_mot_constr'])) + '\n')
            log_file.close()
        self.post_analysis_results['abs_error'] = self.absolute_error()
        self.post_analysis_results['num_compute'], self.post_analysis_results[
            'num_compute_partials'] = self.number_of_evaluations()
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
        computed_optimal_value = self.problem['M_mot']
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
            num_compute = self.problem.model.motor_torque.num_compute - 1
            num_compute_partials = self.problem.model.motor_torque.num_compute_partials
        else:
            num_compute = self.problem.model.problem.model.motor_torque.num_compute - 1
            num_compute_partials = len(self.problem.model.actuator_blackbox.major_iterations) - 1

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
            motor_mass = self.problem.model.motor_mass
        else:
            motor_mass = self.problem.model.actuator_blackbox

        # Save only major iterations objective values
        for i in motor_mass.obj_hist.keys():
            # COBYLA has no major iterations
            if self.derivative_method == 'derivative_free':
                if i >= 1:
                    obj_hist.append(motor_mass.obj_hist[i])

            else:
                # Case where there are more derivatives calls than major iterations
                if i in motor_mass.major_iterations:
                    obj_hist.append(motor_mass.obj_hist[i])

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
            s = '> ---------- Running ' + self.prob_type + ' using ' + self.name + ' formulation and ' + self.optimizer + ' optimizer with ' + self.derivative_method + ' at scale ' + str(
                self.scale) + ' ------------ \n'
            print(s)
            log_file = open(self.log_file, 'a+')
            log_file.writelines(s)
            log_file.close()
        self.run(initial_values)
        self.post_analysis()
        if self.print:
            s = '< ------------------------------------------- End of ' + self.name + ' ------------------------------------------- \n \n'
            log_file = open(self.log_file, 'a+')
            log_file.writelines(s)
            log_file.close()
            print(s)
