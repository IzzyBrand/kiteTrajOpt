import numpy as np
from pydrake.common.containers import namedview
from pydrake.systems.framework import BasicVector, LeafSystem
from simple_kite_dynamics import Kite

KiteState = namedview('KiteState', ['theta', 'phi', 'r', 'thetadot', 'phidot', 'rdot'])

class DrakeKite(LeafSystem):
    def __init__(self, kite=None):
        LeafSystem.__init__(self)
        self.mass = 1  # kg
        self.g = 9.81  # m/s^2
        self.kite = Kite() if kite is None else kite
        self.w = np.array([6, 0, 0])
        
        self.DeclareContinuousState(4)
        self.DeclareVectorInputPort("u", BasicVector(1))
        self.DeclareVectorOutputPort("y", BasicVector(6), self.CopyStateOut)

    def DoCalcTimeDerivatives(self, context, derivatives):
        x = KiteState(context.get_continuous_state_vector().CopyToVector())
        u = self.EvalVectorInput(context, 0).CopyToVector()
        xdot = self.kite.f(x, u, self.w)
        derivatives.get_mutable_vector().SetFromVector(xdot)
        
    def CopyStateOut(self, context, output):
        x = context.get_continuous_state_vector().CopyToVector()
        y = output.SetFromVector(x)

if __name__ == '__main__':
	DrakeKite()
