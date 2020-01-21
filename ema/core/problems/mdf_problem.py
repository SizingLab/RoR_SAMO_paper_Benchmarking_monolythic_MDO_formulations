import math
from openmdao.api import ScipyOptimizeDriver

from ema.core.problems.benchmark_problem import BenchmarkProblem
from ema.core.models.mdf_models import Actuator, ActuatorBlackBox


class MDFProblem(BenchmarkProblem):

    def __init__(self, **kwargs):
        super(MDFProblem, self).__init__(**kwargs)
        if not self.blackbox:
            self.problem.model = Actuator(optimizer=self.optimizer, derivative_method=self.derivative_method)
        else:
            self.problem.model = ActuatorBlackBox(optimizer=self.optimizer, derivative_method=self.derivative_method)
        self.initialize_problem()

    def initialize_problem(self):

        model = self.problem.model
        problem = self.problem

        if self.prob_type == 'MDO':
            # Adding design variables
            model.add_design_var('N_red', lower=1.0, upper=10.)

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

        T_em_real = self.problem['J_mot'] * self.problem['A_max'] * \
                    self.problem['N_red'] / self.problem['p'] + self.problem['F_ema'] * \
                    self.problem['p'] / self.problem['N_red']
        J_mot_real = self.problem['J_mot_ref'] * (abs(self.problem['T_em']) / \
                                                  self.problem['T_em_ref']) ** (5.0 / 3.5)

        success = (str(self.problem['T_em'][0]) != 'nan') and (str(self.problem['T_em'][0]) != 'inf')

        relative_error = abs(abs(self.problem['T_em'][0]) - abs(T_em_real)) / self.problem['T_em']
        success = success and math.isclose(relative_error, 0., abs_tol=tol)
        relative_error = abs(abs(self.problem['J_mot'][0]) - abs(J_mot_real)) / self.problem['J_mot']
        success = success and math.isclose(relative_error, 0., abs_tol=tol)

        if self.prob_type == 'MDO':
            if self.scale == 1.0:
                relative_error = abs(abs(self.problem['M_mot']) - \
                                     abs(self.optimal_value)) / self.optimal_value
                success = success and math.isclose(relative_error, 0., abs_tol=tol)
            relative_error = abs(self.problem['W_mot_constr']) / self.problem['W_mot']
            success = success and math.isclose(relative_error, 0., abs_tol=tol)

        s = 'Success in solving system consistency: ' + str(success) + '\n' \
            'Motor torque value: ' + str(self.problem['T_em'][0]) + '\n' \
            'Motor inertia value: ' + str(self.problem['J_mot'][0]) + '\n' \
            'Motor speed constraint: ' + str(self.problem['W_mot_constr'][0]) + '\n'
        
        if self.print:
            log_file = open(self.log_file, 'a+')        
            log_file.writelines(s)
            log_file.close()
            
            print(s)

        return success

