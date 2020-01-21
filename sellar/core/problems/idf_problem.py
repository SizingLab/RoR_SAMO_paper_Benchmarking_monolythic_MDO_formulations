import math
from openmdao.api import ScipyOptimizeDriver
import numpy as np

from sellar.core.problems.benchmark_problem import BenchmarkProblem
from sellar.core.models.idf_models import Sellar, SellarBlackBox


class IDFProblem(BenchmarkProblem):

    def __init__(self, **kwargs):
        super(IDFProblem, self).__init__(**kwargs)
        if not self.blackbox:
            self.problem.model = Sellar(derivative_method=self.derivative_method)
        else:
            self.problem.model = SellarBlackBox(derivative_method=self.derivative_method)
        self.initialize_problem()

    def initialize_problem(self):

        model = self.problem.model
        problem = self.problem

        # Adding design variables
        model.add_design_var('x', lower=0., upper=10.)
        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('y1', lower=-50., upper=50.)
        model.add_design_var('y2', lower=-50., upper=50.)

        # Adding constraints
        model.add_constraint('con1', upper=0.)
        model.add_constraint('con2', upper=0.)
        model.add_constraint('y1_c_constr', lower=0., upper=0.)
        model.add_constraint('y2_c_constr', lower=0., upper=0.)

        # Adding objective
        model.add_objective('obj')

        # Setting approx_totals if monolythic finite difference requested
        # if self.derivative_method == 'monolythic_fd' and not self.blackbox:
        #     model.approx_totals(method='fd')

        # Setting optimizer
        problem.driver = ScipyOptimizeDriver()
        if self.optimizer == 'COBYLA':
            problem.driver.options['optimizer'] = 'COBYLA'
            problem.driver.options['maxiter'] = 1000
        elif self.optimizer == 'SLSQP':
            problem.driver.options['optimizer'] = 'SLSQP'
            problem.driver.options['maxiter'] = 500
        else:
            raise ('Unknown optimizer' + self.optimizer)
        problem.driver.options['tol'] = self.tol

        # More functions than design variables
        problem.setup(mode='fwd')

    def check_success(self):
        # Custom verification of the optimization results
        tol = self.tol

        relative_error = abs((abs(self.problem['obj']) - abs(self.optimal_value))) / self.optimal_value
        success = math.isclose(relative_error, 0., abs_tol=tol)
        relative_error = abs((abs(self.problem['y1_c_constr'])) / self.problem['y1'])
        success = success and math.isclose(relative_error, 0., abs_tol=tol)
        relative_error = abs((abs(self.problem['y2_c_constr'])) / self.problem['y2'])
        success = success and math.isclose(relative_error, 0., abs_tol=tol)

        s = 'Success in solving system consistency: ' + str(success) + '\n' \
                                                                       'Y1 consistency constraint: ' + str(
            self.problem['y1_c_constr'][0]) + '\n' \
                                              'Y2 consistency constraint: ' + \
            str(self.problem['y2_c_constr'][0]) + '\n'

        if self.print:
            log_file = open(self.log_file, 'a+')
            log_file.writelines(s)
            log_file.close()
            print(s)

        return success
