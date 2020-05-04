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


class MPCDrake(LeafSystem):
    def __init__(self, mpc):
        LeafSystem.__init__(self)

        self.mpc = mpc
        self.U = None

        period = mpc.dt

        self.n_period = mpc.ref_T
        
        self.DeclareDiscreteState(1)
        self.DeclarePeriodicDiscreteUpdate(period)
        self.DeclareVectorInputPort('x_kite', BasicVector(6))
        self.DeclareVectorOutputPort('u_d', BasicVector(1), self.Output)

    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        current_index = context.get_discrete_state_vector().GetAtIndex(0) # current index

        x = self.EvalVectorInput(context, 0).CopyToVector()

        (_, _, _,self.U) = self.mpc.plan(current_index, x)
        
        next_index = int(current_index + 1) % self.n_period # next index
        discrete_state.get_mutable_vector().SetAtIndex(0, next_index)

    def Output(self, context, output):
        x = context.get_discrete_state_vector().CopyToVector()
        y = output.SetFromVector(self.U[0])

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
    
