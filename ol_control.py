#!/usr/bin/python3
from matplotlib import pyplot as plt
import numpy as np

from pydrake.all import Variable, SymbolicVectorSystem
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogOutput
from pydrake.systems.drawing import plot_system_graphviz

from pydrake.common.containers import namedview
from pydrake.systems.framework import BasicVector, LeafSystem


class OlControl(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)

        period = 1/20.

        self.u = np.load('fig8_openloop_control.npy')
        # 10 seconds, 200 samples => 20 hz
        self.n_period = len(self.u)
        
        self.DeclareDiscreteState(1)
        self.DeclarePeriodicDiscreteUpdate(period)
        self.DeclareVectorOutputPort('u_d', BasicVector(1), self.Output)

    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        x = context.get_discrete_state_vector().GetAtIndex(0)
        xn = int(x + 1) % self.n_period
        discrete_state.get_mutable_vector().SetAtIndex(0, xn)

    def Output(self, context, output):
        x = context.get_discrete_state_vector().CopyToVector()
        y = output.SetFromVector([self.u[int(x)]])

if __name__ == '__main__':
    builder = DiagramBuilder()
    ts = builder.AddSystem(OlControl())

    logger = LogOutput(ts.get_output_port(0), builder)
    diagram = builder.Build()

    context = diagram.CreateDefaultContext()

    tc = diagram.GetMutableSubsystemContext(ts, context)

    tc.SetDiscreteState([0])

    simulator = Simulator(diagram, context)
    simulator.AdvanceTo(10)

    plt.plot(logger.data()[0])
    plt.show()
    
