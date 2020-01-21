from openmdao.api import ExplicitComponent


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
        self.add_input('A_max', val=1.)

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
        A_max = inputs['A_max']
        N_red = inputs['N_red']
        p = inputs['p']
        F_ema = inputs['F_ema']

        outputs['T_em'] = J_mot * A_max * N_red / p + F_ema * p / N_red
        self.num_compute += 1

    def compute_partials(self, inputs, partials):
        """ Jacobian for motor torque."""

        partials['T_em', 'J_mot'] = inputs['A_max'] * \
                                    inputs['N_red'] / inputs['p']
        partials['T_em', 'A_max'] = inputs['J_mot'] * \
                                    inputs['N_red'] / inputs['p']
        partials['T_em', 'N_red'] = inputs['J_mot'] * \
                                    inputs['A_max'] / inputs['p'] \
                                    - inputs['F_ema'] * inputs['p'] / \
                                    inputs[
                                        'N_red'] ** 2.0
        partials['T_em', 'p'] = -inputs['J_mot'] * \
                                inputs['A_max'] * \
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
                                         inputs['T_em_ref']) ** (-0.285714285714286) /\
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
                                         inputs['T_em_ref']) ** 0.857142857142857 /\
                                        inputs['T_em_ref']
        partials['M_mot', 'T_em'] = 0.857142857142857 * inputs['M_mot_ref'] *\
                                    (inputs['T_em'] / inputs['T_em_ref']) ** 0.857142857142857 /\
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

        # Max velocity at actuator level
        self.add_input('V_max', val=1.)

        # Reducer reduction ratio
        self.add_input('N_red', val=1.)

        # Lever arm
        self.add_input('p', val=1.)

        # Motor speed constraint
        self.add_output('W_mot_constr', val=1.)

        if self.derivative_method == 'full_analytic':
            self.declare_partials('*', '*', method='exact')
        else:
            self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """Evaluates the equation
        W_mot_constr = V_max * N_red/p - W_mot"""

        V_max = inputs['V_max']
        N_red = inputs['N_red']
        p = inputs['p']
        W_mot = inputs['W_mot']

        outputs['W_mot_constr'] = V_max * N_red / p - W_mot

    def compute_partials(self, inputs, partials):
        """ Jacobian for motor speed."""

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
