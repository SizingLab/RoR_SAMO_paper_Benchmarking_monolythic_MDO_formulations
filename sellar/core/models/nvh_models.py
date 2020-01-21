from openmdao.api import ExplicitComponent, Group, IndepVarComp, Problem
import numpy as np
from .common_models import SellarDis2, SellarConComp1, SellarConComp2, SellarObjective


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

        # Consistency variable
        self.add_input('k_os', val=0.)

        # Global Design Variable
        self.add_input('z', val=np.zeros(2))

        # Local Design Variable
        self.add_input('x', val=0.)

        # Coupling output
        self.add_output('y1', val=0.)

        if self.derivative_method == 'full_analytic':
            self.declare_partials('*', '*', method='exact')
        else:
            self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y1 = k_os * (z1**2 + z2 + x)
        """

        k_os = inputs['k_os']
        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        x = inputs['x']

        outputs['y1'] = k_os * (z1 ** 2 + z2 + x)
        self.num_compute += 1

    def compute_partials(self, inputs, partials):
        """
        Jacobian for Sellar discipline 1.
        """
        k_os = inputs['k_os']
        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        x = inputs['x']

        partials['y1', 'k_os'] = z1 ** 2 + z2 + x
        partials['y1', 'z'] = k_os * np.array([[2.0 * z1, 1.0]])
        partials['y1', 'x'] = k_os * 1.0
        self.num_compute_partials += 1


class SellarConsistencyConstraint(ExplicitComponent):
    """
    Component containing consistency constraint for NVH.
    """

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(SellarConsistencyConstraint, self).__init__(**kwargs)
        self.derivative_method = derivative_method

    def setup(self):
        # Global Design Variable
        self.add_input('z', val=np.zeros(2))

        # Local Design Variable
        self.add_input('x', val=0.)

        # Coupling parameter
        self.add_input('y2', val=0.)

        # Coupling parameter
        self.add_input('y1', val=0.)

        # Consistency constraint 
        self.add_output('c_constr', val=0.)

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
        y1 = inputs['y1']

        outputs['c_constr'] = z1 ** 2 + z2 + x - 0.2 * y2 - y1

    def compute_partials(self, inputs, partials):
        """
        Jacobian for Sellar discipline 1.
        """
        partials['c_constr', 'y2'] = -0.2
        partials['c_constr', 'z'] = np.array([[2.0 * inputs['z'][0], 1.0]])
        partials['c_constr', 'x'] = 1.0
        partials['c_constr', 'y1'] = -1.0


class Sellar(Group):
    """ Group containing the Sellar."""

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(Sellar, self).__init__(**kwargs)
        self.derivative_method = derivative_method

    def setup(self):
        # Design variables
        self.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        self.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])
        self.add_subsystem('pk_os', IndepVarComp('k_os', 0.5), promotes=['k_os'])

        # Subsystems
        self.add_subsystem('d1',
                           SellarDis1(derivative_method=self.derivative_method),
                           promotes=['x', 'z', 'y1', 'k_os'])
        self.add_subsystem('d2',
                           SellarDis2(derivative_method=self.derivative_method),
                           promotes=['z', 'y1', 'y2'])
        self.add_subsystem('con1_comp',
                           SellarConComp1(derivative_method=self.derivative_method),
                           promotes=['con1', 'y1'])
        self.add_subsystem('con2_comp',
                           SellarConComp2(derivative_method=self.derivative_method),
                           promotes=['con2', 'y2'])
        self.add_subsystem('con_c_comp',
                           SellarConsistencyConstraint(derivative_method=self.derivative_method),
                           promotes=['c_constr', 'y2', 'x', 'z', 'y1'])
        self.add_subsystem('obj_comp', SellarObjective(derivative_method=self.derivative_method),
                           promotes=['obj', 'x', 'z', 'y1', 'y2'])


class SellarBlackBoxComponent(ExplicitComponent):

    def __init__(self, derivative_method='full_analytic', problem=None, **kwargs):
        super(SellarBlackBoxComponent, self).__init__(**kwargs)
        self.derivative_method = derivative_method
        self.problem = problem

    def setup(self):
        # Inputs
        self.add_input('x', val=0.)
        self.add_input('z', val=np.zeros(2))
        self.add_input('k_os', val=0.)

        # Outputs
        self.add_output('y1', val=0.)
        self.add_output('y2', val=0.)
        self.add_output('con1', val=0.)
        self.add_output('con2', val=0.)
        self.add_output('c_constr', val=0.)
        self.add_output('obj', val=0.)

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
        p['x'] = inputs['x']
        p['z'] = inputs['z']
        p['k_os'] = inputs['k_os']

        # Running the analysis
        p.run_driver()

        # Updating outputs
        outputs['y1'] = p['y1']
        outputs['y2'] = p['y2']
        outputs['con1'] = p['con1']
        outputs['con2'] = p['con2']
        outputs['c_constr'] = p['c_constr']
        outputs['obj'] = p['obj']

        self.obj_hist[self.num_iter] = outputs['obj'][0]
        self.num_iter += 1

    def compute_partials(self, inputs, partials):

        self.major_iterations.append(self.num_iter)


class SellarBlackBox(Group):
    """ Group containing the Sellar."""

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(SellarBlackBox, self).__init__(**kwargs)
        self.derivative_method = derivative_method
        self.problem = Problem()
        self.problem.model = Sellar(derivative_method=self.derivative_method)

    def setup(self):
        # Design variables
        self.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        self.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])
        self.add_subsystem('pk_os', IndepVarComp('k_os', 0.5), promotes=['k_os'])

        # Subsystems
        self.add_subsystem('sellar_blackbox',
                           SellarBlackBoxComponent(derivative_method=self.derivative_method,
                                                   problem=self.problem),
                           promotes=['x', 'z', 'y1', 'y2', 'k_os', 'con1', 'con2', 'c_constr', 'obj'])
