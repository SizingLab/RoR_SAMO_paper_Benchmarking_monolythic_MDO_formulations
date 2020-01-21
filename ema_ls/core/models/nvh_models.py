import numpy as np
from openmdao.api import ExplicitComponent, Group, IndepVarComp, Problem
from .common_models import Trajectory, MotorInertia, MotorSpeed, MotorMass, MotorSpeedConstraint


class MotorTorque(ExplicitComponent):
    """ExplicitComponent that computes the motor torque."""

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(MotorTorque, self).__init__(**kwargs)
        self.derivative_method = derivative_method

    def setup(self):

        # Attributes to count number of compute and compute_partials performed
        self.num_compute = 0
        self.num_compute_partials = 0

        # Oversizing factor
        self.add_input('k_os', val=1.1)

        # Reducer reduction ratio
        self.add_input('N_red', val=1.)

        # Screw pitch
        self.add_input('p', val=1.)

        # Actuator load
        self.add_input('F_ema', val=1.0)

        # Motor electromagnetic torque
        self.add_output('T_em', val=1.0)

        if self.derivative_method == 'full_analytic':
            self.declare_partials('*', '*', method='exact')
        else:
            self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """Evaluates the equation
        T_em = k_os * F_ema * p/N_red"""

        k_os = inputs['k_os']
        N_red = inputs['N_red']
        p = inputs['p']
        F_ema = inputs['F_ema']

        outputs['T_em'] = k_os * F_ema * p / N_red

        self.num_compute += 1

    def compute_partials(self, inputs, partials):
        """ Jacobian for motor torque."""

        partials['T_em', 'k_os'] = inputs['F_ema'] * inputs['p'] / inputs['N_red']
        partials['T_em', 'N_red'] = - inputs['k_os'] * inputs['F_ema'] * inputs['p'] / inputs['N_red'] ** 2.0
        partials['T_em', 'p'] = inputs['k_os'] * inputs['F_ema'] / inputs['N_red']
        partials['T_em', 'F_ema'] = inputs['k_os'] * inputs['p'] / inputs['N_red']

        self.num_compute_partials += 1


class MotorTorqueConsistencyConstraint(ExplicitComponent):
    """ExplicitComponent that computes the motor torque."""

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(MotorTorqueConsistencyConstraint, self).__init__(**kwargs)
        self.derivative_method = derivative_method

    def setup(self):

        # Attributes to count number of compute and compute_partials performed
        self.num_compute = 0
        self.num_compute_partials = 0
        # Electromagnetic torque
        self.add_input('T_em', val=1.)

        # Motor inertia
        self.add_input('J_mot', val=1.)

        # Max acceleration at actuator level
        self.add_input('A_rms', val=1.)

        # Reducer reduction ratio
        self.add_input('N_red', val=1.)

        # Screw pitch
        self.add_input('p', val=1.)

        # Actuator load
        self.add_input('F_ema', val=1.)

        # Motor electromagnetic torque
        self.add_output('T_em_constr', val=1.)

        if self.derivative_method == 'full_analytic':
            self.declare_partials('*', '*', method='exact')
        else:
            self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """Evaluates the equation
        T_em = J_mot * A_max * N_red/p + F_ema * p/N_red"""

        T_em = inputs['T_em']
        J_mot = inputs['J_mot']
        A_rms = inputs['A_rms']
        N_red = inputs['N_red']
        p = inputs['p']
        F_ema = inputs['F_ema']

        T_em_real = J_mot * A_rms * N_red / p + F_ema * p / N_red
        T_em_constr = T_em_real - T_em

        outputs['T_em_constr'] = T_em_constr

    def compute_partials(self, inputs, partials):
        """ Jacobian for motor torque."""

        partials['T_em_constr', 'T_em'] = -1.0

        partials['T_em_constr', 'J_mot'] = inputs['A_rms'] * \
                                           inputs['N_red'] / inputs['p']
        partials['T_em_constr', 'A_rms'] = inputs['J_mot'] * \
                                           inputs['N_red'] / inputs['p']
        partials['T_em_constr', 'N_red'] = inputs['J_mot'] * \
                                           inputs['A_rms'] / inputs['p'] \
                                           - inputs['F_ema'] * inputs['p'] / \
                                           inputs[
                                               'N_red'] ** 2.0
        partials['T_em_constr', 'p'] = -inputs['J_mot'] * \
                                       inputs['A_rms'] * \
                                       inputs['N_red'] / inputs['p'] ** 2.0 \
                                       + inputs['F_ema'] / \
                                       inputs['N_red']
        partials['T_em_constr', 'F_ema'] = inputs['p'] / inputs['N_red']


class Actuator(Group):
    """ Group containing the Actuator."""

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(Actuator, self).__init__(**kwargs)
        self.derivative_method = derivative_method

    def initialize(self):
        # Number of sinusoids
        self.options.declare('N', default=100, types=int)
        self.options.declare('t', default=np.linspace(0, 10, num=100))

    def setup(self):
        N = self.options['N']

        # Design variables
        self.add_subsystem('pN_red', IndepVarComp('N_red', 1.), promotes=['N_red'])
        self.add_subsystem('pk_os', IndepVarComp('k_os', 55.6515 / (1.59e-3 * 7.e4 / 2.0)), promotes=['k_os'])
        self.add_subsystem('pa_n', IndepVarComp('a_n', np.full((N, 1), 1e-6)), promotes=['a_n'])
        self.add_subsystem('pb_n', IndepVarComp('b_n', np.full((N, 1), 1e-6)), promotes=['b_n'])

        # Fixed parameters
        self.add_subsystem('pF_ema', IndepVarComp('F_ema', 7.e4), promotes=['F_ema'])
        self.add_subsystem('pp', IndepVarComp('p', 1.59e-3), promotes=['p'])
        self.add_subsystem('pT_em_ref', IndepVarComp('T_em_ref', 4.), promotes=['T_em_ref'])
        self.add_subsystem('pJ_mot_ref', IndepVarComp('J_mot_ref', 2.9e-4), promotes=['J_mot_ref'])
        self.add_subsystem('pW_mot_ref', IndepVarComp('W_mot_ref', 754.), promotes=['W_mot_ref'])
        self.add_subsystem('pM_mot_ref', IndepVarComp('M_mot_ref', 3.8), promotes=['M_mot_ref'])

        # Subsystems
        self.add_subsystem('trajectory',
                           Trajectory(derivative_method=self.derivative_method,
                                      N=self.options['N'],
                                      t=self.options['t']),
                           promotes=['a_n', 'b_n', 'X_final',
                                     'X_ema', 'V_ema', 'A_ema',
                                     'V_final', 'A_rms', 'V_max'])
        self.add_subsystem('motor_torque',
                           MotorTorque(derivative_method=self.derivative_method),
                           promotes=['T_em', 'k_os', 'N_red', 'p', 'F_ema'])
        self.add_subsystem('motor_inertia',
                           MotorInertia(derivative_method=self.derivative_method),
                           promotes=['J_mot', 'J_mot_ref', 'T_em', 'T_em_ref'])
        self.add_subsystem('motor_speed',
                           MotorSpeed(derivative_method=self.derivative_method),
                           promotes=['W_mot', 'W_mot_ref', 'T_em', 'T_em_ref'])
        self.add_subsystem('motor_mass',
                           MotorMass(derivative_method=self.derivative_method),
                           promotes=['M_mot', 'M_mot_ref', 'T_em', 'T_em_ref'])
        self.add_subsystem('motor_speed_constraint',
                           MotorSpeedConstraint(derivative_method=self.derivative_method),
                           promotes=['W_mot_constr', 'W_mot', 'V_max', 'N_red', 'p'])
        self.add_subsystem('motor_torque_consistency_constraint',
                           MotorTorqueConsistencyConstraint(derivative_method=self.derivative_method),
                           promotes=['T_em_constr', 'T_em', 'J_mot', 'A_rms', 'N_red', 'p', 'F_ema'])


class ActuatorBlackBoxComponent(ExplicitComponent):

    def __init__(self, derivative_method='full_analytic', problem=None, **kwargs):
        super(ActuatorBlackBoxComponent, self).__init__(**kwargs)
        self.derivative_method = derivative_method
        self.problem = problem

    def initialize(self):
        # Number of sinusoids
        self.options.declare('N', default=100, types=int)
        self.options.declare('t', default=np.linspace(0, 10, num=100))

    def setup(self):
        N_t = len(self.options['t'])

        # Inputs
        self.add_input('N_red', val=1.)
        self.add_input('k_os', val=1.)
        self.add_input('a_n', val=1., shape=(self.options['N'], 1))
        self.add_input('b_n', val=1., shape=(self.options['N'], 1))
        self.add_input('F_ema', val=1.)
        self.add_input('p', val=1.)
        self.add_input('T_em_ref', val=1.)
        self.add_input('J_mot_ref', val=1.)
        self.add_input('W_mot_ref', val=1.)
        self.add_input('M_mot_ref', val=1.)

        # Outputs
        self.add_output('T_em', val=1.)
        self.add_output('J_mot', val=1.)
        self.add_output('W_mot', val=1.)
        self.add_output('M_mot', val=1., ref=100)
        self.add_output('X_ema', val=1., shape=N_t)
        self.add_output('V_ema', val=1., shape=N_t)
        self.add_output('A_ema', val=1., shape=N_t)
        self.add_output('A_rms', val=1.)
        self.add_output('V_max', val=1.)
        self.add_output('W_mot_constr', val=1.)
        self.add_output('T_em_constr', val=1.)
        self.add_output('X_final', val=1.)
        self.add_output('V_final', val=1.)

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
        p['k_os'] = inputs['k_os']
        p['a_n'] = inputs['a_n']
        p['b_n'] = inputs['b_n']
        p['F_ema'] = inputs['F_ema']
        p['p'] = inputs['p']
        p['T_em_ref'] = inputs['T_em_ref']
        p['J_mot_ref'] = inputs['J_mot_ref']
        p['W_mot_ref'] = inputs['W_mot_ref']
        p['M_mot_ref'] = inputs['M_mot_ref']

        # Running the analysis
        p.run_driver()

        # Updating outputs
        outputs['T_em'] = p['T_em']
        outputs['J_mot'] = p['J_mot']
        outputs['W_mot'] = p['W_mot']
        outputs['M_mot'] = p['M_mot']
        outputs['X_ema'] = p['X_ema']
        outputs['V_ema'] = p['V_ema']
        outputs['A_ema'] = p['A_ema']
        outputs['A_rms'] = p['A_rms']
        outputs['V_max'] = p['V_max']
        outputs['W_mot_constr'] = p['W_mot_constr']
        outputs['T_em_constr'] = p['T_em_constr']
        outputs['X_final'] = p['X_final']
        outputs['V_final'] = p['V_final']
        self.obj_hist[self.num_iter] = outputs['M_mot'][0]
        self.num_iter += 1

    def compute_partials(self, inputs, partials):

        self.major_iterations.append(self.num_iter)


class ActuatorBlackBox(Group):
    """ Group containing the Actuator."""

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(ActuatorBlackBox, self).__init__(**kwargs)
        self.derivative_method = derivative_method
        self.problem = Problem()
        self.problem.model = Actuator(derivative_method=self.derivative_method,
                                      N = self.options['N'],
                                      t = self.options['t'])

    def initialize(self):
        # Number of sinusoids
        self.options.declare('N', default=100, types=int)
        self.options.declare('t', default=np.linspace(0, 10, num=100))

    def setup(self):
        N = self.options['N']

        # Design variables
        self.add_subsystem('pN_red', IndepVarComp('N_red', 1.), promotes=['N_red'])
        self.add_subsystem('pk_os', IndepVarComp('k_os', 1.5), promotes=['k_os'])
        self.add_subsystem('pa_n', IndepVarComp('a_n', np.full((N, 1), 1e-6)), promotes=['a_n'])
        self.add_subsystem('pb_n', IndepVarComp('b_n', np.full((N, 1), 1e-6)), promotes=['b_n'])

        # Fixed parameters
        self.add_subsystem('pF_ema', IndepVarComp('F_ema', 7.e4), promotes=['F_ema'])
        self.add_subsystem('pp', IndepVarComp('p', 1.59e-3), promotes=['p'])
        self.add_subsystem('pT_em_ref', IndepVarComp('T_em_ref', 4.), promotes=['T_em_ref'])
        self.add_subsystem('pJ_mot_ref', IndepVarComp('J_mot_ref', 2.9e-4), promotes=['J_mot_ref'])
        self.add_subsystem('pW_mot_ref', IndepVarComp('W_mot_ref', 754.), promotes=['W_mot_ref'])
        self.add_subsystem('pM_mot_ref', IndepVarComp('M_mot_ref', 3.8), promotes=['M_mot_ref'])

        # Subsystems
        self.add_subsystem('actuator_blackbox',
                           ActuatorBlackBoxComponent(derivative_method=self.derivative_method,
                                                     N=self.options['N'],
                                                     t=self.options['t'],
                                                     problem=self.problem),
                           promotes=['T_em', 'J_mot', 'a_n', 'b_n', 'A_rms', 'N_red', 'p', 'F_ema', 'T_em_ref',
                                     # 'X_ema', 'V_ema', 'A_ema',
                                     'J_mot_ref', 'W_mot_ref', 'M_mot_ref', 'V_max', 'W_mot',
                                     'M_mot', 'W_mot_constr', 'T_em_constr', 'X_final', 'V_final'])
