from openmdao.api import ExplicitComponent, Group, IndepVarComp, NewtonSolver, \
    DirectSolver, NonlinearBlockGS, Problem
from .common_models import MotorTorque, MotorInertia, MotorSpeed, MotorMass, MotorSpeedConstraint


class MDA(Group):
    """ Group that performs the MDA."""

    def __init__(self, optimizer='SLSQP', derivative_method='full_analytic', **kwargs):
        super(MDA, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.derivative_method = derivative_method

    def setup(self):
        self.add_subsystem('motor_torque',
                           MotorTorque(derivative_method=self.derivative_method),
                           promotes=['T_em', 'J_mot', 'A_max', 'N_red', 'p', 'F_ema'])
        self.add_subsystem('motor_inertia',
                           MotorInertia(derivative_method=self.derivative_method),
                           promotes=['J_mot', 'J_mot_ref', 'T_em', 'T_em_ref'])

        if self.optimizer == 'SLSQP':
            # Non-linear Solver
            self.nonlinear_solver = NewtonSolver()
            self.nonlinear_solver.options['atol'] = 1.0e-10
            self.nonlinear_solver.options['maxiter'] = 500
            self.nonlinear_solver.options['iprint'] = 0

            # Linear Solver
            self.linear_solver = DirectSolver()
            self.linear_solver.options['iprint'] = 0

        elif self.optimizer == 'COBYLA':
            # Non-linear Solver
            self.nonlinear_solver = NonlinearBlockGS()
            self.nonlinear_solver.options['atol'] = 1.0e-10
            self.nonlinear_solver.options['maxiter'] = 500
            self.nonlinear_solver.options['iprint'] = 0
        else:
            raise ('Unknown optimizer' + self.optimizer)


class Actuator(Group):
    """ Group containing the Actuator."""

    def __init__(self, optimizer='SLSQP', derivative_method='full_analytic', **kwargs):
        super(Actuator, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.derivative_method = derivative_method

    def setup(self):
        # Design variables
        self.add_subsystem('pN_red', IndepVarComp('N_red', 1.), promotes=['N_red'])

        # Fixed parameters
        self.add_subsystem('pF_ema', IndepVarComp('F_ema', 7.e4), promotes=['F_ema'])
        self.add_subsystem('pA_max', IndepVarComp('A_max', 2.), promotes=['A_max'])
        self.add_subsystem('pp', IndepVarComp('p', 1.59e-3), promotes=['p'])
        self.add_subsystem('pT_em_ref', IndepVarComp('T_em_ref', 13.4), promotes=['T_em_ref'])
        self.add_subsystem('pJ_mot_ref', IndepVarComp('J_mot_ref', 2.9e-4), promotes=['J_mot_ref'])
        self.add_subsystem('pW_mot_ref', IndepVarComp('W_mot_ref', 754.), promotes=['W_mot_ref'])
        self.add_subsystem('pM_mot_ref', IndepVarComp('M_mot_ref', 3.8), promotes=['M_mot_ref'])
        self.add_subsystem('pV_max', IndepVarComp('V_max', 2.e-1), promotes=['V_max'])

        # Subsystems
        self.add_subsystem('MDA',
                           MDA(optimizer=self.optimizer, derivative_method=self.derivative_method),
                           promotes=['T_em', 'J_mot', 'A_max', 'N_red', 'p', 'F_ema', 'J_mot_ref', 'T_em_ref'])
        self.add_subsystem('motor_speed',
                           MotorSpeed(derivative_method=self.derivative_method),
                           promotes=['W_mot', 'W_mot_ref', 'T_em', 'T_em_ref'])
        self.add_subsystem('motor_mass',
                           MotorMass(derivative_method=self.derivative_method),
                           promotes=['M_mot', 'M_mot_ref', 'T_em', 'T_em_ref'])
        self.add_subsystem('motor_speed_constraint',
                           MotorSpeedConstraint(derivative_method=self.derivative_method),
                           promotes=['W_mot_constr', 'W_mot', 'V_max', 'N_red', 'p'])


class ActuatorBlackBoxComponent(ExplicitComponent):

    def __init__(self, derivative_method='full_analytic', problem=None, **kwargs):
        super(ActuatorBlackBoxComponent, self).__init__(**kwargs)
        self.derivative_method = derivative_method
        self.problem = problem

    def setup(self):
        # Inputs
        self.add_input('N_red', val=1.)
        self.add_input('F_ema', val=1.)
        self.add_input('A_max', val=1.)
        self.add_input('p', val=1.)
        self.add_input('T_em_ref', val=1.)
        self.add_input('J_mot_ref', val=1.)
        self.add_input('W_mot_ref', val=1.)
        self.add_input('M_mot_ref', val=1.)
        self.add_input('V_max', val=1.)

        # Outputs
        self.add_output('T_em', val=1.)
        self.add_output('J_mot', val=1.)
        self.add_output('W_mot', val=1.)
        self.add_output('M_mot', val=1., ref=100)
        self.add_output('W_mot_constr', val=1.)

        self.problem.setup()

        if self.derivative_method == 'full_analytic':
            self.declare_partials('*', '*', method='exact')
        else:
            self.declare_partials('*', '*', method='fd')

        self.obj_hist = {}

        self.num_iter = 0

        # First index is one because of init
        self.major_iterations = [1]

    def compute(self, inputs, outputs):
        p = self.problem

        # Reading inputs
        p['N_red'] = inputs['N_red']
        p['F_ema'] = inputs['F_ema']
        p['A_max'] = inputs['A_max']
        p['p'] = inputs['p']
        p['T_em_ref'] = inputs['T_em_ref']
        p['J_mot_ref'] = inputs['J_mot_ref']
        p['W_mot_ref'] = inputs['W_mot_ref']
        p['M_mot_ref'] = inputs['M_mot_ref']
        p['V_max'] = inputs['V_max']

        # Running the analysis
        p.run_driver()

        # Updating outputs
        outputs['T_em'] = p['T_em']
        outputs['J_mot'] = p['J_mot']
        outputs['W_mot'] = p['W_mot']
        outputs['M_mot'] = p['M_mot']
        outputs['W_mot_constr'] = p['W_mot_constr']
        self.obj_hist[self.num_iter] = outputs['M_mot'][0]
        self.num_iter += 1

    def compute_partials(self, inputs, partials):

        self.major_iterations.append(self.num_iter)


class ActuatorBlackBox(Group):
    """ Group containing the Actuator."""

    def __init__(self, optimizer='SLSQP', derivative_method='full_analytic', **kwargs):
        super(ActuatorBlackBox, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.derivative_method = derivative_method
        self.problem = Problem()
        self.problem.model = Actuator(optimizer=self.optimizer, derivative_method=self.derivative_method)

    def setup(self):
        # Design variables
        self.add_subsystem('pN_red', IndepVarComp('N_red', 1.), promotes=['N_red'])

        # Fixed parameters
        self.add_subsystem('pF_ema', IndepVarComp('F_ema', 7.e4), promotes=['F_ema'])
        self.add_subsystem('pA_max', IndepVarComp('A_max', 2.), promotes=['A_max'])
        self.add_subsystem('pp', IndepVarComp('p', 1.59e-3), promotes=['p'])
        self.add_subsystem('pT_em_ref', IndepVarComp('T_em_ref', 13.4), promotes=['T_em_ref'])
        self.add_subsystem('pJ_mot_ref', IndepVarComp('J_mot_ref', 2.9e-4), promotes=['J_mot_ref'])
        self.add_subsystem('pW_mot_ref', IndepVarComp('W_mot_ref', 754.), promotes=['W_mot_ref'])
        self.add_subsystem('pM_mot_ref', IndepVarComp('M_mot_ref', 3.8), promotes=['M_mot_ref'])
        self.add_subsystem('pV_max', IndepVarComp('V_max', 2.e-1), promotes=['V_max'])

        # Subsystems
        self.add_subsystem('actuator_blackbox',
                           ActuatorBlackBoxComponent(derivative_method=self.derivative_method,
                                                     problem=self.problem), \
                           promotes=['T_em', 'J_mot', 'A_max', 'N_red', 'p', 'F_ema', 'T_em_ref', \
                                     'J_mot_ref', 'W_mot_ref', 'M_mot_ref', 'V_max', 'W_mot', \
                                     'M_mot', 'W_mot_constr'])
