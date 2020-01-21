from openmdao.api import ExplicitComponent, Group, IndepVarComp, \
    NewtonSolver, DirectSolver, NonlinearBlockGS, Problem
import numpy as np
from .common_models import SellarDis1, SellarDis2, SellarConComp1, SellarConComp2, SellarObjective


class MDA(Group):
    """ Group that performs the MDA."""

    def __init__(self, optimizer='SLSQP', derivative_method='full_analytic', **kwargs):
        super(MDA, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.derivative_method = derivative_method

    def setup(self):
        self.add_subsystem('d1',
                           SellarDis1(derivative_method=self.derivative_method),
                           promotes=['x', 'z', 'y1', 'y2'])
        self.add_subsystem('d2',
                           SellarDis2(derivative_method=self.derivative_method),
                           promotes=['z', 'y1', 'y2'])

        if self.optimizer == 'SLSQP':
            # Non-linear Solver
            self.nonlinear_solver = NewtonSolver()
            self.nonlinear_solver.options['atol'] = 1.0e-8
            self.nonlinear_solver.options['maxiter'] = 500
            self.nonlinear_solver.options['iprint'] = 0

            # Linear Solver
            self.linear_solver = DirectSolver()
            self.linear_solver.options['iprint'] = 0

        elif self.optimizer == 'COBYLA':
            # Non-linear Solver
            self.nonlinear_solver = NonlinearBlockGS()
            self.nonlinear_solver.options['atol'] = 1.0e-6
            self.nonlinear_solver.options['maxiter'] = 500
            self.nonlinear_solver.options['iprint'] = 0
        else:
            raise ('Unknown optimizer' + self.optimizer)


class Sellar(Group):
    """ Group containing the Sellar problem."""

    def __init__(self, optimizer='SLSQP', derivative_method='full_analytic', **kwargs):
        super(Sellar, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.derivative_method = derivative_method

    def setup(self):
        # Design variables
        self.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        self.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        # Subsystems
        self.add_subsystem('MDA',
                           MDA(optimizer=self.optimizer, derivative_method=self.derivative_method),
                           promotes=['x', 'z', 'y1', 'y2'])
        self.add_subsystem('con1_comp',
                           SellarConComp1(derivative_method=self.derivative_method),
                           promotes=['con1', 'y1'])
        self.add_subsystem('con2_comp',
                           SellarConComp2(derivative_method=self.derivative_method),
                           promotes=['con2', 'y2'])
        self.add_subsystem('obj_comp',
                           SellarObjective(derivative_method=self.derivative_method),
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

        # Outputs
        self.add_output('y1', val=0.)
        self.add_output('y2', val=0.)
        self.add_output('con1', val=0.)
        self.add_output('con2', val=0.)
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

        # Running the analysis
        p.run_driver()

        # Updating outputs
        outputs['y1'] = p['y1']
        outputs['y2'] = p['y2']
        outputs['con1'] = p['con1']
        outputs['con2'] = p['con2']
        outputs['obj'] = p['obj']

        self.obj_hist[self.num_iter] = outputs['obj'][0]
        self.num_iter += 1

    def compute_partials(self, inputs, partials):

        self.major_iterations.append(self.num_iter)


class SellarBlackBox(Group):
    """ Group containing the Sellar."""

    def __init__(self, optimizer='SLSQP', derivative_method='full_analytic', **kwargs):
        super(SellarBlackBox, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.derivative_method = derivative_method
        self.problem = Problem()
        self.problem.model = Sellar(optimizer=self.optimizer, derivative_method=self.derivative_method)

    def setup(self):
        # Design variables
        self.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        self.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        # Subsystems
        self.add_subsystem('sellar_blackbox',
                           SellarBlackBoxComponent(derivative_method=self.derivative_method,
                                                   problem=self.problem),
                           promotes=['obj', 'con1', 'con2', 'x', 'z', 'y1', 'y2'])
