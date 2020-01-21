from openmdao.api import ExplicitComponent, Group, IndepVarComp, Problem
import numpy as np
from .common_models import SellarDis1, SellarDis2, SellarConComp1, \
    SellarConComp2, SellarConsistencyConstraintY2, SellarObjective


class Sellar(Group):
    """ Group containing the Sellar."""

    def __init__(self, derivative_method='full_analytic', **kwargs):
        super(Sellar, self).__init__(**kwargs)
        self.derivative_method = derivative_method

    def setup(self):
        # Design variables
        self.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        self.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])
        self.add_subsystem('py2', IndepVarComp('y2', 0.0), promotes=['y2'])

        # Subsystems
        self.add_subsystem('d1',
                           SellarDis1(derivative_method=self.derivative_method),
                           promotes=['x', 'z', 'y1'])
        self.add_subsystem('d2',
                           SellarDis2(derivative_method=self.derivative_method),
                           promotes=['z', 'y1'])
        self.add_subsystem('con1_comp',
                           SellarConComp1(derivative_method=self.derivative_method),
                           promotes=['con1','y1'])
        self.add_subsystem('con2_comp',
                           SellarConComp2(derivative_method=self.derivative_method),
                           promotes=['con2'])
        self.add_subsystem('con_c_y2_comp', SellarConsistencyConstraintY2(derivative_method=self.derivative_method),
                           promotes=['y2_c_constr', 'y2'])
        self.add_subsystem('obj_comp',
                           SellarObjective(derivative_method=self.derivative_method),
                           promotes=['obj', 'x', 'z', 'y1'])

        # Connections
        self.connect('y2', 'd1.y2')     

        self.connect('d2.y2', 'con2_comp.y2')
        self.connect('d2.y2', 'con_c_y2_comp.y2_t')
        self.connect('d2.y2', 'obj_comp.y2')


class SellarBlackBoxComponent(ExplicitComponent):

    def __init__(self, derivative_method='full_analytic', problem=None, **kwargs):
        super(SellarBlackBoxComponent, self).__init__(**kwargs)
        self.derivative_method = derivative_method
        self.problem = problem

    def setup(self):
        # Inputs
        self.add_input('x', val=0.)
        self.add_input('z', val=np.zeros(2))
        self.add_input('y2', val=0.)

        # Outputs
        self.add_output('y1', val=0.)
        self.add_output('con1', val=0.)
        self.add_output('con2', val=0.)
        self.add_output('y2_c_constr', val=0.)
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
        p['y2'] = inputs['y2']

        # Running the analysis
        p.run_driver()

        # Updating outputs
        outputs['y1'] = p['y1']
        outputs['con1'] = p['con1']
        outputs['con2'] = p['con2']
        outputs['y2_c_constr'] = p['y2_c_constr']
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
        self.add_subsystem('py2', IndepVarComp('y2', 0.0), promotes=['y2'])

        # Subsystems
        self.add_subsystem('sellar_blackbox',
                           SellarBlackBoxComponent(derivative_method=self.derivative_method,
                                                   problem=self.problem),
                           promotes=['x', 'z', 'y1', 'y2', 'con1', 'con2', 'y2_c_constr', 'obj'])
