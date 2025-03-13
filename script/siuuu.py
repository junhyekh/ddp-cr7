import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl
from crocoddyl.utils.biped import plotSolution

from ddp.utils import G1Loader, dataloader

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

robot = G1Loader(rpy='something').robot
rmodel = robot.model
lims = rmodel.effortLimit
data = dataloader().get_data()

rdata = rmodel.createData()
state = crocoddyl.StateMultibody(rmodel)
actuation = crocoddyl.ActuationModelFloatingBase(state)


rightFoot = "right_ankle_roll_link"
leftFoot = "left_ankle_roll_link"

rightFootId = rmodel.getFrameId(rightFoot)
leftFootId = rmodel.getFrameId(leftFoot)
q0 = robot.q0
x0 = np.concatenate([q0, np.zeros(rmodel.nv)])

DT = 1/30
n_traj = next(iter(data.values())).shape[0]

# Create two contact models used along the motion
contactModel1Foot = crocoddyl.ContactModelMultiple(state, actuation.nu)
contactModel2Feet = crocoddyl.ContactModelMultiple(state, actuation.nu)
supportContactModelLeft = crocoddyl.ContactModel6D(
    state,
    leftFootId,
    pinocchio.SE3.Identity(),
    pinocchio.LOCAL_WORLD_ALIGNED,
    actuation.nu,
    np.array([0, 40]),
)
supportContactModelRight = crocoddyl.ContactModel6D(
    state,
    rightFootId,
    pinocchio.SE3.Identity(),
    pinocchio.LOCAL_WORLD_ALIGNED,
    actuation.nu,
    np.array([0, 40]),
)
contactModel1Foot.addContact(rightFoot + "_contact", supportContactModelRight)
contactModel2Feet.addContact(leftFoot + "_contact", supportContactModelLeft)
contactModel2Feet.addContact(rightFoot + "_contact", supportContactModelRight)

# Cost for self-collision
maxfloat = sys.float_info.max
xlb = np.concatenate(
    [
        -maxfloat * np.ones(6),  # dimension of the SE(3) manifold
        rmodel.lowerPositionLimit[7:],
        -maxfloat * np.ones(state.nv),
    ]
)
xub = np.concatenate(
    [
        maxfloat * np.ones(6),  # dimension of the SE(3) manifold
        rmodel.upperPositionLimit[7:],
        maxfloat * np.ones(state.nv),
    ]
)
bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
xLimitResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
limitCost = crocoddyl.CostModelResidual(state, xLimitActivation, xLimitResidual)

# Cost for state and control
xResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
xActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [10] * state.nv) ** 2
)
uResidual = crocoddyl.ResidualModelControl(state, actuation.nu)
xTActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [100] * state.nv) ** 2
)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
xRegTermCost = crocoddyl.CostModelResidual(state, xTActivation, xResidual)

runningCostModel1 = crocoddyl.CostModelSum(state, actuation.nu)
runningCostModel2 = crocoddyl.CostModelSum(state, actuation.nu)
runningCostModel3 = crocoddyl.CostModelSum(state, actuation.nu)
terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)

runningCostModel1.addCost("stateReg", xRegCost, 1e-3)
runningCostModel1.addCost("ctrlReg", uRegCost, 1e-4)

runningCostModel2.addCost("stateReg", xRegCost, 1e-3)
runningCostModel2.addCost("ctrlReg", uRegCost, 1e-4)
runningCostModel2.addCost("limitCost", limitCost, 1e3)

runningCostModel3.addCost("ctrlReg", uRegCost, 1e-4)
runningCostModel3.addCost("limitCost", limitCost, 1e3)

terminalCostModel.addCost("stateReg", xRegTermCost, 1e-3)
terminalCostModel.addCost("limitCost", limitCost, 1e3)

dmodelRunning1 = crocoddyl.DifferentialActionModelContactFwdDynamics(
    state, actuation, contactModel2Feet, runningCostModel1
)

runningModel1 = crocoddyl.IntegratedActionModelEuler(dmodelRunning1, DT)

x0 = np.concatenate([q0, pinocchio.utils.zero(state.nv)])

problem = crocoddyl.ShootingProblem(
    x0, [runningModel1] * (n_traj-1), runningModel1
)

solver = crocoddyl.SolverIntro(problem)
solver.setCallbacks([crocoddyl.CallbackVerbose()])

xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
solver.th_stop = 1e-7
solver.solve(xs, us, 500, False, 1e-9)


display = None
if WITHDISPLAY:
    if display is None:
        try:
            import gepetto

            gepetto.corbaserver.Client()
            display = crocoddyl.GepettoDisplay(robot)
            display.robot.viewer.gui.addSphere(
                "world/point", 0.05, [1.0, 0.0, 0.0, 1.0]
            )  # radius = .1, RGBA=1001
            display.robot.viewer.gui.applyConfiguration(
                "world/point", [*target.tolist(), 0.0, 0.0, 0.0, 1.0]
            )  # xyz+quaternion
        except Exception:
            # print(target[None].shape,
            #  target[None].repeat((solver.problem.T + 1), 1).shape
            # )
            # print( np.concatenate(
            #             [np.repeat(np.array([0.0, 0.4, 0.0]), 2*T, axis=0),
            #             np.repeat(np.array([0.3, 0.15, 0.35]), 2*T, axis=0)
            #             ], axis=0
            #         ).shape)
            display = crocoddyl.MeshcatDisplay(robot,
            targets=data)
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        time.sleep(1.0)