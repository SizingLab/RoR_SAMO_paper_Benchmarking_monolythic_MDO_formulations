import math
import numpy as np
import matplotlib.pyplot as plt
from openmdao.api import ScipyOptimizeDriver

from ema.core.problems.benchmark_problem import BenchmarkProblem
from ema.core.models.hybrid_models import Actuator, ActuatorBlackBox


class HybridProblem(BenchmarkProblem):

    def __init__(self, **kwargs):
        super(HybridProblem, self).__init__(**kwargs)

        if not self.blackbox:
            self.problem.model = Actuator(derivative_method=self.derivative_method,
                                          N=self.N,
                                          t=self.t)
        else:
            self.problem.model = ActuatorBlackBox(derivative_method=self.derivative_method,
                                                  N=self.N,
                                                  t=self.t)
        self.initialize_problem()

    def initialize_problem(self):

        model = self.problem.model
        problem = self.problem

        model.add_design_var('a_n', lower=-1e0, upper=1e0)
        model.add_design_var('b_n', lower=-1e0, upper=1e0)
        model.add_constraint('X_final', lower=0.15, upper=0.15)
        model.add_constraint('V_final', lower=0.0, upper=0.0)

        # Adding design variables
        if self.prob_type == 'MDO':
            model.add_design_var('N_red', lower=1.0, upper=8.)
        model.add_design_var('J_mot', lower=1.e-6, upper=1.e-1 * self.scale ** (5.0 / 3.5))

        # Adding constraints
        if self.prob_type == 'MDO':
            model.add_constraint('W_mot_constr', upper=0.)
        model.add_constraint('J_mot_c_constr', lower=0., upper=0.)

        # Adding objective
        model.add_objective('M_mot')

        # Setting approx_totals if monolythic finite difference requested
        if self.derivative_method == 'monolythic_fd' and not self.blackbox:
            model.approx_totals(method='fd')

        # Setting optimizer
        problem.driver = ScipyOptimizeDriver()
        if self.optimizer == 'COBYLA':
            problem.driver.options['optimizer'] = 'COBYLA'
            problem.driver.options['maxiter'] = 50000
        elif self.optimizer == 'SLSQP':
            problem.driver.options['optimizer'] = 'SLSQP'
            problem.driver.options['maxiter'] = 10000
        else:
            raise ('Unknown optimizer' + self.optimizer)
        problem.driver.options['tol'] = self.tol

        # More design variables than functions but the adjoint fails
        problem.setup(mode='fwd')

    def check_success(self):
        # Custom verification of the optimization results
        tol = self.tol

        relative_error = abs(self.problem['J_mot_c_constr']) / self.problem['J_mot']
        success = math.isclose(relative_error, 0., abs_tol=tol)

        if self.prob_type == 'MDO':
            if self.scale == 1.0:
                relative_error = abs(abs(self.problem['M_mot']) - abs(self.optimal_value)) / self.optimal_value
                # * 1000 due to scaler
                success = success and math.isclose(relative_error, 0., abs_tol=tol * 1000)
            relative_error = abs(self.problem['W_mot_constr']) / self.problem['W_mot']
            success = success and math.isclose(relative_error, 0., abs_tol=tol)

            relative_error = abs(self.problem['V_final'])
            success = success and math.isclose(relative_error, 0., abs_tol=tol)
            relative_error = (abs(self.problem['X_final']) - 0.15) / 0.15
            success = success and math.isclose(relative_error, 0., abs_tol=tol)

        s = 'Success in solving system consistency: ' + str(success) + '\n' \
                                                                       'Motor mass: ' + str(
            self.problem['M_mot'][0]) + '\n' \
                                        'A_rms: ' + str(self.problem['A_rms'][0]) + '\n' \
                                                                                    'T_em: ' + str(
            self.problem['T_em'][0]) + '\n' \
                                       'X_final: ' + str(self.problem['X_final'][0]) + '\n' \
                                                                                       'V_final: ' + str(
            self.problem['V_final'][0]) + '\n' \
                                          'V_max: ' + str(np.max(self.problem['V_ema'])) + '\n' \
                                                                                           'N_red: ' + str(
            self.problem['N_red'][0]) + '\n' \
                                        'Motor inertia consistency constraint: ' + str(
            self.problem['J_mot_c_constr'][0]) + '\n' \
                                                 'Motor speed constraint: ' + str(
            np.max(self.problem['W_mot_constr'][0])) + '\n'

        if self.plot:
            t = self.t
            X_ema = self.problem['X_ema']
            V_ema = self.problem['V_ema']
            A_ema = self.problem['A_ema']
            f, (ax1, ax2, ax3) = plt.subplots(3, 1)

            ax1.plot(t, X_ema)
            ax2.plot(t, V_ema)
            ax3.plot(t, A_ema)

            plt.show()

        if self.print:
            log_file = open(self.log_file, 'a+')
            log_file.writelines(s)
            log_file.close()

            print(s)

        return success
