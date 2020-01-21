import numpy as np
from openmdao.api import ExplicitComponent


class Trajectory(ExplicitComponent):
    """ExplicitComponent that computes the trajectory."""

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(Trajectory, self).__init__(**kwargs)
        self.derivative_method = derivative_method

    def initialize(self):
        # Number of sinusoids
        self.options.declare('N', default=100, types=int)
        self.options.declare('t', default=np.linspace(0, 10, num=100))

    def setup(self):
        N = self.options['N']
        t = self.options['t']
        N_t = len(t)

        # Coefficients an
        self.add_input('a_n', val=1., shape=N)

        # Coefficients bn
        self.add_input('b_n', val=1., shape=N)

        # EMA position
        self.add_output('X_ema', val=1., shape=N_t)

        # EMA final position
        self.add_output('X_final', val=1.)

        # EMA speed
        self.add_output('V_ema', val=1., shape=N_t)

        # EMA max speed
        self.add_output('V_max', val=1.)

        # EMA final speed
        self.add_output('V_final', val=1.)

        # EMA acceleration
        self.add_output('A_ema', val=1., shape=N_t)

        # EMA rms acceleration
        self.add_output('A_rms', val=1.)

        if self.derivative_method == 'full_analytic':
            self.declare_partials('*', '*', method='exact')
        else:
            self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """Evaluates Fourier series for
        acceleration, speed and position"""

        N = self.options['N']
        t = self.options['t']
        N_t = len(t)

        a_n = inputs['a_n']
        b_n = inputs['b_n']

        X_ema = np.linspace(0, 0, num=N_t)
        V_ema = np.linspace(0, 0, num=N_t)
        A_ema = np.linspace(0, 0, num=N_t)

        for n in range(1, N):
            # A_ema(t=0) = 0
            A_ema += a_n[n - 1] * np.cos(n * t) + b_n[n - 1] * np.sin(n * t) - a_n[n - 1]

            # V_ema(t=0) = 0
            V_ema += a_n[n - 1] * 1 / n * np.sin(n * t) - b_n[n - 1] * 1 / n * np.cos(n * t) + b_n[n - 1] * 1 / n - a_n[
                n - 1] * t

            # X_ema(t=0) = 0
            X_ema += - a_n[n - 1] * (1 / n) ** 2 * np.cos(n * t) - b_n[n - 1] * (1 / n) ** 2 * np.sin(n * t) + a_n[
                n - 1] * (1 / n) ** 2 - 1 / 2 * a_n[n - 1] * t ** 2 + b_n[n - 1] * 1 / n * t

        X_final = X_ema[-1]
        V_final = V_ema[-1]
        A_rms = np.sqrt(np.mean(A_ema ** 2))

        i_max = np.argmax(V_ema)
        V_max = V_ema[i_max]

        outputs['X_ema'] = X_ema
        outputs['X_final'] = X_final
        outputs['V_ema'] = V_ema
        outputs['V_max'] = V_max
        outputs['V_final'] = V_final
        outputs['A_ema'] = A_ema
        outputs['A_rms'] = A_rms

    def compute_partials(self, inputs, partials):

        N = self.options['N']
        t = self.options['t']
        N_t = len(t)

        a_n = inputs['a_n']
        b_n = inputs['b_n']

        d_X_ema_d_a_n = np.zeros((N_t, N))
        d_X_ema_d_b_n = np.zeros((N_t, N))

        V_ema = np.linspace(0, 0, num=N_t)
        d_V_ema_d_a_n = np.zeros((N_t, N))
        d_V_ema_d_b_n = np.zeros((N_t, N))

        A_ema = np.linspace(0, 0, num=N_t)
        d_A_ema_d_a_n = np.zeros((N_t, N))
        d_A_ema_d_b_n = np.zeros((N_t, N))
        d_A_rms_d_a_n = np.zeros((1, N))
        d_A_rms_d_b_n = np.zeros((1, N))

        for n in range(1, N):
            # A_ema(t=0) = 0
            A_ema += a_n[n - 1] * np.cos(n * t) + b_n[n - 1] * np.sin(n * t) - a_n[n - 1]
            d_A_ema_d_a_n[:, n - 1] = np.cos(n * t) - 1
            d_A_ema_d_b_n[:, n - 1] = np.sin(n * t)

            # V_ema(t=0) = 0
            V_ema += a_n[n - 1] * 1 / n * np.sin(n * t) - b_n[n - 1] * 1 / n * np.cos(n * t) + b_n[n - 1] * 1 / n - a_n[
                n - 1] * t
            d_V_ema_d_a_n[:, n - 1] = 1 / n * np.sin(n * t) - t
            d_V_ema_d_b_n[:, n - 1] = - 1 / n * np.cos(n * t) + 1 / n

            # X_ema(t=0) = 0
            d_X_ema_d_a_n[:, n - 1] = - 1 / n ** 2 * np.cos(n * t) + 1 / n ** 2 - 1 / 2 * t ** 2
            d_X_ema_d_b_n[:, n - 1] = - 1 / n ** 2 * np.sin(n * t) + 1 / n * t

        for n in range(1, N):
            d_A_rms_d_a_n[0, n - 1] = 1 / 2 * (np.mean(A_ema ** 2)) ** (-1 / 2) * np.mean(
                2 * A_ema * d_A_ema_d_a_n[:, n - 1])
            d_A_rms_d_b_n[0, n - 1] = 1 / 2 * (np.mean(A_ema ** 2)) ** (-1 / 2) * np.mean(
                2 * A_ema * d_A_ema_d_b_n[:, n - 1])

        i_max = np.argmax(V_ema)

        partials['X_ema', 'a_n'] = d_X_ema_d_a_n
        partials['X_ema', 'b_n'] = d_X_ema_d_b_n
        partials['X_final', 'a_n'] = d_X_ema_d_a_n[-1]
        partials['X_final', 'b_n'] = d_X_ema_d_b_n[-1]
        partials['V_ema', 'a_n'] = d_V_ema_d_a_n
        partials['V_ema', 'b_n'] = d_V_ema_d_b_n
        partials['V_max', 'a_n'] = d_V_ema_d_a_n[i_max]
        partials['V_max', 'b_n'] = d_V_ema_d_b_n[i_max]
        partials['V_final', 'a_n'] = d_V_ema_d_a_n[-1]
        partials['V_final', 'b_n'] = d_V_ema_d_b_n[-1]
        partials['A_ema', 'a_n'] = d_A_ema_d_a_n
        partials['A_ema', 'b_n'] = d_A_ema_d_b_n
        partials['A_rms', 'a_n'] = d_A_rms_d_a_n
        partials['A_rms', 'b_n'] = d_A_rms_d_b_n


class MotorTorque(ExplicitComponent):
    """ExplicitComponent that computes the motor torque."""

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(MotorTorque, self).__init__(**kwargs)
        self.derivative_method = derivative_method

    def setup(self):

        # Attributes to count number of compute and compute_partials performed
        self.num_compute = 0
        self.num_compute_partials = 0

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
        self.add_output('T_em', val=1.)

        if self.derivative_method == 'full_analytic':
            self.declare_partials('*', '*', method='exact')
        else:
            self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """Evaluates the equation
        T_em = J_mot * A_max * N_red/p + F_ema * p/N_red"""

        J_mot = inputs['J_mot']
        A_rms = inputs['A_rms']
        N_red = inputs['N_red']
        p = inputs['p']
        F_ema = inputs['F_ema']

        T_em = J_mot * A_rms * N_red / p + F_ema * p / N_red

        outputs['T_em'] = abs(T_em)
        self.num_compute += 1

    def compute_partials(self, inputs, partials):
        """ Jacobian for motor torque."""

        partials['T_em', 'J_mot'] = inputs['A_rms'] * \
                                    inputs['N_red'] / inputs['p']
        partials['T_em', 'A_rms'] = inputs['J_mot'] * \
                                    inputs['N_red'] / inputs['p']
        partials['T_em', 'N_red'] = inputs['J_mot'] * \
                                    inputs['A_rms'] / inputs['p'] \
                                    - inputs['F_ema'] * inputs['p'] / \
                                    inputs[
                                        'N_red'] ** 2.0
        partials['T_em', 'p'] = -inputs['J_mot'] * \
                                inputs['A_rms'] * \
                                inputs['N_red'] / inputs['p'] ** 2.0 \
                                + inputs['F_ema'] / \
                                inputs['N_red']
        partials['T_em', 'F_ema'] = inputs['p'] / inputs['N_red']
        self.num_compute_partials += 1


class MotorInertia(ExplicitComponent):
    """ExplicitComponent that computes the motor inertia."""

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(MotorInertia, self).__init__(**kwargs)
        self.derivative_method = derivative_method

    def setup(self):

        # Reference motor inertia
        self.add_input('J_mot_ref', val=1.)

        # Reference motor electromagnetic torque
        self.add_input('T_em_ref', val=1.)

        # Motor electromagnetic torque
        self.add_input('T_em', val=1.)

        # Motor inertia
        self.add_output('J_mot', val=1.)

        if self.derivative_method == 'full_analytic':
            self.declare_partials('*', '*', method='exact')
        else:
            self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """Evaluates the equation
        J_mot = J_mot_ref * (T_em/T_em_ref)**(5.0/3.5)"""

        J_mot_ref = inputs['J_mot_ref']
        T_em_ref = inputs['T_em_ref']
        T_em = inputs['T_em']

        # Using abs() since COBYLA goes out of bound for HYBRID and MDF
        outputs['J_mot'] = J_mot_ref * (abs(T_em) / T_em_ref) ** (5.0 / 3.5)

    def compute_partials(self, inputs, partials):
        """ Jacobian for motor inertia."""

        partials['J_mot', 'J_mot_ref'] = (abs(inputs['T_em']) / \
                                          inputs['T_em_ref']) ** (5.0 / 3.5)
        partials['J_mot', 'T_em_ref'] = -1.42857142857143 * \
                                        inputs['J_mot_ref'] * \
                                        (abs(inputs['T_em']) / \
                                         inputs['T_em_ref']) ** 1.42857142857143 / \
                                        inputs['T_em_ref']
        partials['J_mot', 'T_em'] = 1.42857142857143 * inputs['J_mot_ref'] \
                                    * (abs(inputs['T_em']) / \
                                       inputs['T_em_ref']) ** 1.42857142857143 / \
                                    inputs['T_em']


class MotorSpeed(ExplicitComponent):
    """ExplicitComponent that computes the motor maximum speed."""

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(MotorSpeed, self).__init__(**kwargs)
        self.derivative_method = derivative_method

    def setup(self):

        # Reference motor speed
        self.add_input('W_mot_ref', val=1.)

        # Reference motor electromagnetic torque
        self.add_input('T_em_ref', val=1.)

        # Motor electromagnetic torque
        self.add_input('T_em', val=1.)

        # Motor speed
        self.add_output('W_mot', val=1.)

        if self.derivative_method == 'full_analytic':
            self.declare_partials('*', '*', method='exact')
        else:
            self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """Evaluates the equation
        W_mot = W_mot_ref * (T_em/T_em_ref)**(-1.0/3.5)"""

        W_mot_ref = inputs['W_mot_ref']
        T_em_ref = inputs['T_em_ref']
        T_em = inputs['T_em']

        # Using abs() since COBYLA goes out of bound for HYBRID and MDF
        outputs['W_mot'] = W_mot_ref * (abs(T_em) / T_em_ref) ** (-1.0 / 3.5)

    def compute_partials(self, inputs, partials):
        """ Jacobian for motor speed."""

        partials['W_mot', 'W_mot_ref'] = (inputs['T_em'] / inputs['T_em_ref']) ** (-1.0 / 3.5)
        partials['W_mot', 'T_em_ref'] = 0.285714285714286 * inputs['W_mot_ref'] * \
                                        (inputs['T_em'] / \
                                         inputs['T_em_ref']) ** (-0.285714285714286) / \
                                        inputs['T_em_ref']
        partials['W_mot', 'T_em'] = -0.285714285714286 * inputs['W_mot_ref'] * \
                                    (inputs['T_em'] / inputs['T_em_ref']) ** (-0.285714285714286) / \
                                    inputs['T_em']


class MotorMass(ExplicitComponent):
    """ExplicitComponent that computes the motor mass."""

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(MotorMass, self).__init__(**kwargs)
        self.derivative_method = derivative_method

    def setup(self):

        # Reference motor mass
        self.add_input('M_mot_ref', val=1.)

        # Reference motor electromagnetic torque
        self.add_input('T_em_ref', val=1.)

        # Motor electromagnetic torque
        self.add_input('T_em', val=1.)

        # Motor mass with scaler
        self.add_output('M_mot', val=1., ref=100.)

        if self.derivative_method == 'full_analytic':
            self.declare_partials('*', '*', method='exact')
        else:
            self.declare_partials('*', '*', method='fd')

        self.obj_hist = {}

        self.num_iter = 0

        # First index is one because of init
        self.major_iterations = [1]

    def compute(self, inputs, outputs):
        """Evaluates the equation
        M_mot = M_mot_ref * (T_em/T_em_ref)**(3.0/3.5)"""

        M_mot_ref = inputs['M_mot_ref']
        T_em_ref = inputs['T_em_ref']
        T_em = inputs['T_em']

        # Using abs() since COBYLA goes out of bound for HYBRID
        M_mot = M_mot_ref * (abs(T_em) / T_em_ref) ** (3.0 / 3.5)

        outputs['M_mot'] = M_mot
        self.obj_hist[self.num_iter] = outputs['M_mot'][0]
        self.num_iter += 1

    def compute_partials(self, inputs, partials):
        """ Jacobian for motor inertia."""

        partials['M_mot', 'M_mot_ref'] = (inputs['T_em'] / \
                                          inputs['T_em_ref']) ** (3.0 / 3.5)
        partials['M_mot', 'T_em_ref'] = -0.857142857142857 * \
                                        inputs['M_mot_ref'] * \
                                        (inputs['T_em'] / \
                                         inputs['T_em_ref']) ** 0.857142857142857 / \
                                        inputs['T_em_ref']
        partials['M_mot', 'T_em'] = 0.857142857142857 * inputs['M_mot_ref'] * \
                                    (inputs['T_em'] / inputs['T_em_ref']) ** 0.857142857142857 / \
                                    inputs['T_em']

        self.major_iterations.append(self.num_iter)


class MotorSpeedConstraint(ExplicitComponent):
    """ExplicitComponent that computes the motor speed constraint."""

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(MotorSpeedConstraint, self).__init__(**kwargs)
        self.derivative_method = derivative_method

    def setup(self):

        # Motor speed
        self.add_input('W_mot', val=1.)

        self.add_input('V_max', val=1.)

        # Reducer reduction ratio
        self.add_input('N_red', val=1.)

        # Lever arm
        self.add_input('p', val=1.)
        self.add_output('W_mot_constr', val=1.)

        if self.derivative_method == 'full_analytic':
            self.declare_partials('*', '*', method='exact')
        else:
            self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """Evaluates the equation
        W_mot_constr = V_ema * N_red/p - W_mot"""
        V_max = inputs['V_max']
        N_red = inputs['N_red']
        p = inputs['p']
        W_mot = inputs['W_mot']

        outputs['W_mot_constr'] = V_max * N_red / p - W_mot

    def compute_partials(self, inputs, partials):
        partials['W_mot_constr', 'V_max'] = inputs['N_red'] / inputs['p']
        partials['W_mot_constr', 'N_red'] = inputs['V_max'] / inputs['p']
        partials['W_mot_constr', 'p'] = -inputs['V_max'] * inputs['N_red'] / (inputs['p'] ** 2.0)
        partials['W_mot_constr', 'W_mot'] = -1.


class MotorInertiaConsistencyConstraint(ExplicitComponent):
    """ExplicitComponent that computes the motor inertia consistency constraint for IDF and Hybrid."""

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(MotorInertiaConsistencyConstraint, self).__init__(**kwargs)
        self.derivative_method = derivative_method

    def setup(self):

        # Motor inertia consistency design variable
        self.add_input('J_mot_t', val=1.)

        # Motor inertia
        self.add_input('J_mot', val=1.)

        # Motor speed constraint
        self.add_output('J_mot_c_constr', val=1.)

        if self.derivative_method == 'full_analytic':
            self.declare_partials('*', '*', method='exact')
        else:
            self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """Evaluates the equation
        J_mot_c_constr = J_mot_t - J_mot"""

        J_mot_t = inputs['J_mot_t']
        J_mot = inputs['J_mot']

        outputs['J_mot_c_constr'] = J_mot_t - J_mot

    def compute_partials(self, inputs, partials):
        """ Jacobian for motor inertia consistency constraint."""

        partials['J_mot_c_constr', 'J_mot_t'] = 1.
        partials['J_mot_c_constr', 'J_mot'] = -1.
