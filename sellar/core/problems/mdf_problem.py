import math
from openmdao.api import ScipyOptimizeDriver
import numpy as np

from sellar.core.problems.benchmark_problem import BenchmarkProblem
from sellar.core.models.mdf_models import Sellar, SellarBlackBox


class MDFProblem(BenchmarkProblem):

    def __init__(self, **kwargs):
        super(MDFProblem, self).__init__(**kwargs)
        if not self.blackbox:
            self.problem.model = Sellar(optimizer=self.optimizer, derivative_method=self.derivative_method)
        else:
            self.problem.model = SellarBlackBox(optimizer=self.optimizer, derivative_method=self.derivative_method)
        self.initialize_problem()

    def initialize_problem(self):

        model = self.problem.model
        problem = self.problem

        # Adding design variables
        model.add_design_var('x', lower=0., upper=10.)
        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        
        # Adding constraints
        model.add_constraint('con1', upper=0.)
        model.add_constraint('con2', upper=0.)

        # Adding objective
        model.add_objective('obj')

        # Setting optimizer
        problem.driver = ScipyOptimizeDriver()
        if self.optimizer == 'COBYLA':
            problem.driver.options['optimizer'] = 'COBYLA'
            problem.driver.options['maxiter'] = 1000
        elif self.optimizer == 'SLSQP':
            problem.driver.options['optimizer'] = 'SLSQP'
            problem.driver.options['maxiter'] = 500
        else:
            raise('Unknown optimizer' + self.optimizer)
        problem.driver.options['tol'] = self.tol

        # More functions than design variables
        problem.setup(mode='fwd')

    def number_of_evaluations(self):

        # Number of counts, -1 due to setup
        if not self.blackbox:
            num_compute = self.problem.model.MDA.d1.num_compute - 1 
            num_compute_partials = self.problem.model.MDA.d1.num_compute_partials - 1
        else:
            num_compute = self.problem.model.problem.model.MDA.d1.num_compute - 1 
            num_compute_partials = self.problem.model.problem.model.MDA.d1.num_compute_partials - 1
            
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

    def check_success(self):
        # Custom verification of the optimization results
        tol = self.tol

        relative_error = abs((abs(self.problem['obj']) - abs(self.optimal_value))) / self.optimal_value
        success = math.isclose(relative_error, 0., abs_tol=tol)

        z1 = self.problem['z'][0]
        z2 = self.problem['z'][1]
        y1 = self.problem['y1'][0]
        y2 = self.problem['y2'][0]
        x = self.problem['x'][0]

        y1_real = z1**2 + z2 + x - 0.2*y2
        y2_real = y1**.5 + z1 + z2

        relative_error = abs((abs(self.problem['y1'][0]) - abs(y1_real))) / y1_real
        success = success and math.isclose(relative_error, 0., abs_tol=tol)
        relative_error = abs((abs(self.problem['y2'][0]) - abs(y2_real))) / y2_real
        success = success and math.isclose(relative_error, 0., abs_tol=tol)

        success = success and (str(self.problem['y1'][0]) != 'nan') and (str(self.problem['y1'][0]) != 'inf')

        s = 'Success in solving system consistency: ' + str(success) + '\n' \
            'y1 value: ' + str(self.problem['y1'][0]) + '\n' \
            'y2 value: ' + str(self.problem['y2'][0]) + '\n'
        
        if self.print:
            log_file = open(self.log_file, 'a+')        
            log_file.writelines(s)
            log_file.close()
            print(s)

        return success

