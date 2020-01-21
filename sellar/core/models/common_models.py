from openmdao.api import ExplicitComponent
import numpy as np


class SellarDis1(ExplicitComponent):
    """
    Component containing Discipline 1.
    """

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(SellarDis1, self).__init__(**kwargs)
        self.derivative_method = derivative_method

    def setup(self):
        # Attributes to count number of compute and compute_partials performed
        self.num_compute = 0
        self.num_compute_partials = 0

        # Global Design Variable
        self.add_input('z', val=np.zeros(2))

        # Local Design Variable
        self.add_input('x', val=0.)

        # Coupling parameter
        self.add_input('y2', val=0.)

        # Coupling output
        self.add_output('y1', val=0.)

        if self.derivative_method == 'full_analytic':
            self.declare_partials('*', '*', method='exact')
        else:
            self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y1 = z1**2 + z2 + x - 0.2*y2
        """

        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        x = inputs['x']
        y2 = inputs['y2']

        outputs['y1'] = z1 ** 2 + z2 + x - 0.2 * y2
        self.num_compute += 1

    def compute_partials(self, inputs, partials):
        """
        Jacobian for Sellar discipline 1.
        """
        partials['y1', 'y2'] = -0.2
        partials['y1', 'z'] = np.array([[2.0 * inputs['z'][0], 1.0]])
        partials['y1', 'x'] = 1.0
        self.num_compute_partials += 1


class SellarDis2(ExplicitComponent):
    """
    Component containing Discipline 2.
    """

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(SellarDis2, self).__init__(**kwargs)
        self.derivative_method = derivative_method

    def setup(self):
        # Attributes to count number of compute and compute_partials performed
        self.num_compute = 0
        self.num_compute_partials = 0

        # Global Design Variable
        self.add_input('z', val=np.zeros(2))

        # Coupling parameter
        self.add_input('y1', val=0.)

        # Coupling output
        self.add_output('y2', val=0.)

        if self.derivative_method == 'full_analytic':
            self.declare_partials('*', '*', method='exact')
        else:
            self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y2 = y1**(.5) + z1 + z2
        """

        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        y1 = inputs['y1']

        # Note: this may cause some issues. However, y1 is constrained to be
        # above 3.16, so lets just let it converge, and the optimizer will
        # throw it out
        if y1.real < 0.0:
            y1 *= -1

        outputs['y2'] = y1 ** .5 + z1 + z2

    def compute_partials(self, inputs, J):
        """
        Jacobian for Sellar discipline 2.
        """
        y1 = inputs['y1']
        if y1.real < 0.0:
            y1 *= -1
        if y1.real < 1e-8:
            y1 = 1e-8

        J['y2', 'y1'] = .5 * y1 ** -.5
        J['y2', 'z'] = np.array([[1.0, 1.0]])


class SellarConComp1(ExplicitComponent):
    """
    Component containing the constraint component 1 of the Sellar problem.
    """

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(SellarConComp1, self).__init__(**kwargs)
        self.derivative_method = derivative_method

    def setup(self):
        # Parameter
        self.add_input('y1', val=0.)

        # Constraint
        self.add_output('con1', val=0.)

        if self.derivative_method == 'full_analytic':
            self.declare_partials('*', '*', method='exact')
        else:
            self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        con1 = 3.16 - y1
        """

        y1 = inputs['y1']

        outputs['con1'] = 3.16 - y1

    def compute_partials(self, inputs, J):
        """
        Jacobian for component 1 of the Sellar problem.
        """
        J['con1', 'y1'] = -1.


class SellarConComp2(ExplicitComponent):
    """
    Component containing the constraint component 2 of the Sellar problem.
    """

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(SellarConComp2, self).__init__(**kwargs)
        self.derivative_method = derivative_method

    def setup(self):
        # Parameter
        self.add_input('y2', val=0.)

        # Constraint
        self.add_output('con2', val=0.)

        if self.derivative_method == 'full_analytic':
            self.declare_partials('*', '*', method='exact')
        else:
            self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        con2 = y2 - 24.0
        """

        y2 = inputs['y2']

        outputs['con2'] = y2 - 24.0

    def compute_partials(self, inputs, J):
        """
        Jacobian for component 2 of the Sellar problem.
        """
        J['con2', 'y2'] = 1.


class SellarConsistencyConstraintY2(ExplicitComponent):
    """ExplicitComponent that computes the y2 consistency constraint for IDF."""

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(SellarConsistencyConstraintY2, self).__init__(**kwargs)
        self.derivative_method = derivative_method

    def setup(self):

        # y2 consistency design variable
        self.add_input('y2_t', val=0.)

        # y2
        self.add_input('y2', val=0.)

        # y2 consistency constraint
        self.add_output('y2_c_constr', val=0.)

        if self.derivative_method == ('derivative_free' or 'full_analytic'):
            self.declare_partials('*', '*', method='exact')
        else:
            self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """Evaluates the equation
        y2_c_constr = y2_t - y2"""

        y2_t = inputs['y2_t']
        y2 = inputs['y2']

        outputs['y2_c_constr'] = y2_t - y2

    def compute_partials(self, inputs, partials):
        """ Jacobian for y2 consistency constraint."""

        partials['y2_c_constr', 'y2_t'] = 1.
        partials['y2_c_constr', 'y2'] = -1.


class SellarObjective(ExplicitComponent):
    """
    Component containing the objective of the Sellar problem.
    """

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(SellarObjective, self).__init__(**kwargs)
        self.derivative_method = derivative_method

    def setup(self):
        # Global Design Variable
        self.add_input('z', val=np.zeros(2))

        # Local Design Variable
        self.add_input('x', val=0.)

        # Coupling parameter
        self.add_input('y1', val=0.)
        self.add_input('y2', val=0.)

        # Output (objective)
        self.add_output('obj', val=0.)

        if self.derivative_method == 'full_analytic':
            self.declare_partials('*', '*', method='exact')
        else:
            self.declare_partials('*', '*', method='fd')

        self.obj_hist = {}

        self.num_iter = 0

        # First index is one because of init
        self.major_iterations = [1]

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        obj = x**2 + z2 + y1 + exp(-y2)
        """
        z2 = inputs['z'][1]
        x = inputs['x']
        y1 = inputs['y1']
        y2 = inputs['y2']

        outputs['obj'] = x ** 2. + z2 + y1 + np.exp(-y2)
        self.obj_hist[self.num_iter] = outputs['obj'][0]
        self.num_iter += 1

    def compute_partials(self, inputs, J):
        """
        Jacobian for objective of the Sellar problem.
        """
        J['obj', 'z'] = np.array([0., 1.])
        J['obj', 'x'] = 2. * inputs['x']
        J['obj', 'y1'] = 1.
        J['obj', 'y2'] = -np.exp(-inputs['y2'])

        self.major_iterations.append(self.num_iter)
