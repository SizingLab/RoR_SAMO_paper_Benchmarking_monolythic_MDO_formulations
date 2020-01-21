import math
from openmdao.api import ScipyOptimizeDriver

from ema.core.problems.benchmark_problem import BenchmarkProblem
from ema.core.models.nvh_models import Actuator, ActuatorBlackBox


class NVHProblem(BenchmarkProblem):

    def __init__(self, **kwargs):
        super(NVHProblem, self).__init__(**kwargs)
        if not self.blackbox:
            self.problem.model = Actuator(derivative_method=self.derivative_method)
        else:
            self.problem.model = ActuatorBlackBox(derivative_method=self.derivative_method)
        self.initialize_problem()

    def initialize_problem(self):

        model = self.problem.model
        problem = self.problem

        # Adding design variables
        if self.prob_type == 'MDO':
            model.add_design_var('N_red', lower=1.0, upper=10.) 
        model.add_design_var('k_os', lower=0.1, upper=5.)

        # Adding constraints
        if self.prob_type == 'MDO':
            model.add_constraint('W_mot_constr', upper=0.)
        model.add_constraint('T_em_constr', upper=0.)

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
    
    def check_success(self):
        # Custom verification of the optimization results
        tol = self.tol

        relative_error = abs(self.problem['T_em_constr']) / self.problem['T_em']
        success = math.isclose(relative_error, 0., abs_tol=tol)

        if self.prob_type == 'MDO':
            if self.scale == 1.0:
                relative_error = abs(abs(self.problem['M_mot']) - \
                                     abs(self.optimal_value)) / self.optimal_value
                success = success and math.isclose(relative_error, 0., abs_tol=tol)
            relative_error = abs(self.problem['W_mot_constr']) / self.problem['W_mot']
            success = success and math.isclose(relative_error, 0., abs_tol=tol)

        s = 'Success in solving system consistency: ' + str(success) + '\n' \
            'Normalized variable k_os value: ' + str(self.problem['k_os'][0]) + '\n' \
            'Torque consistency constraint T_em_constr value: ' + str(self.problem['T_em_constr'][0]) + '\n' \
            'Motor speed constraint: ' + str(self.problem['W_mot_constr'][0]) + '\n'
        
        if self.print:
            log_file = open(self.log_file, 'a+')        
            log_file.writelines(s)
            log_file.close()
            
            print(s)

        return success

