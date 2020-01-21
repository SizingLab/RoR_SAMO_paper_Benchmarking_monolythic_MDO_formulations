import math
import numpy as np
from openmdao.api import ScipyOptimizeDriver

from ema.core.problems.benchmark_problem import BenchmarkProblem
from ema.core.models.mdf_models import Actuator, ActuatorBlackBox
import matplotlib.pyplot as plt

class MDFProblem(BenchmarkProblem):

    def __init__(self, **kwargs):
        super(MDFProblem, self).__init__(**kwargs)

        if not self.blackbox:
            self.problem.model = Actuator(optimizer=self.optimizer, derivative_method=self.derivative_method,
                                          N=self.N, t=self.t)
        else:
            self.problem.model = ActuatorBlackBox(optimizer=self.optimizer,
                                                  derivative_method=self.derivative_method,N=self.N,
                                                  t=self.t)
        self.initialize_problem()

    def initialize_problem(self):

        model = self.problem.model
        problem = self.problem

        model.add_design_var('a_n', lower=-1e0, upper=1e0)
        model.add_design_var('b_n', lower=-1e0, upper=1e0)
        model.add_constraint('X_final', lower=0.15, upper=0.15)
        model.add_constraint('V_final', lower=0.0, upper=0.0)

        if self.prob_type == 'MDO':
            # Adding design variables
            model.add_design_var('N_red', lower=1., upper=8.)

            # Adding constraints
            model.add_constraint('W_mot_constr', upper=0.)

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
                raise('Unknown optimizer' + self.optimizer)
            problem.driver.options['tol'] = self.tol

        # More design variables than functions but the adjoint fails
        problem.setup(mode='fwd')

    def number_of_evaluations(self):

        # Number of counts, -1 due to setup
        if not self.blackbox:
            num_compute = self.problem.model.MDA.motor_torque.num_compute - 1
            num_compute_partials = self.problem.model.MDA.motor_torque.num_compute_partials - 1
        else:
            num_compute = self.problem.model.problem.model.MDA.motor_torque.num_compute - 1
            num_compute_partials = self.problem.model.problem.model.MDA.motor_torque.num_compute_partials - 1

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
        T_em_real = (np.mean((self.problem['J_mot'] * self.problem['A_rms'] *
                              self.problem['N_red'] / self.problem['p'] + self.problem['F_ema'] *
                              self.problem['p'] / self.problem['N_red']) ** 2)) ** (1 / 2)
        J_mot_real = self.problem['J_mot_ref'] * (abs(self.problem['T_em']) /
                                                  self.problem['T_em_ref']) ** (5.0 / 3.5)

        success = (str(self.problem['T_em'][0]) != 'nan') and (str(self.problem['T_em'][0]) != 'inf')

        relative_error = abs(abs(self.problem['T_em'][0]) - abs(T_em_real)) / self.problem['T_em']
        success = success and math.isclose(relative_error, 0., abs_tol=tol)
        relative_error = abs(abs(self.problem['J_mot'][0]) - abs(J_mot_real)) / self.problem['J_mot']
        success = success and math.isclose(relative_error, 0., abs_tol=tol)

        relative_error = abs(self.problem['V_final'])
        success = success and math.isclose(relative_error, 0., abs_tol=tol)
        relative_error = (abs(self.problem['X_final']) - 0.15) / 0.15
        success = success and math.isclose(relative_error, 0., abs_tol=tol)

        if self.prob_type == 'MDO':
            if self.scale == 1.0:
                relative_error = abs(abs(self.problem['M_mot']) - \
                                     abs(self.optimal_value)) / self.optimal_value
                # * 1000 due to scaler
                success = success and math.isclose(relative_error, 0., abs_tol=tol * 1000)
            relative_error = abs(self.problem['W_mot_constr']) / self.problem['W_mot']
            success = success and math.isclose(relative_error, 0., abs_tol=tol)

        s = 'Success in solving system consistency: ' + str(success) + '\n' \
            'Motor mass: ' + str(self.problem['M_mot'][0]) + '\n' \
            'Motor torque value: ' + str(self.problem['T_em'][0]) + '\n' \
            'Motor inertia value: ' + str(self.problem['J_mot'][0]) + '\n' \
            'A_rms: ' + str(self.problem['A_rms'][0]) + '\n' \
            'T_em: ' + str(self.problem['T_em'][0]) + '\n' \
            'X_final: ' + str(self.problem['X_final'][0]) + '\n' \
            'V_final: ' + str(self.problem['V_final'][0]) + '\n' \
            'V_max: ' + str(np.max(self.problem['V_ema'])) + '\n' \
            'N_red: ' + str(self.problem['N_red'][0]) + '\n' \
            'Motor speed constraint: ' + str(np.max(self.problem['W_mot_constr'])) + '\n'

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

