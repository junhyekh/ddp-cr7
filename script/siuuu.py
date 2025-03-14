import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio
from copy import deepcopy

import crocoddyl
from crocoddyl.utils.biped import plotSolution

from ddp.utils import G1Loader, dataloader
from ddp.helper import *

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

#########################setup################################
# load robot and data
robot = G1Loader(rpy='something').robot
rmodel = robot.model
lims = rmodel.effortLimit
data_set = dataloader()
data = data_set.get_data()

rdata = rmodel.createData()
state = crocoddyl.StateMultibody(rmodel)
actuation = crocoddyl.ActuationModelFloatingBase(state)

##########################configuration######################
dt = 1/30
interpolate = 3
DT = dt / interpolate
offset = -0.1
n_traj = next(iter(data.values())).shape[0]
rightFoot = "right_ankle_roll_link"
leftFoot = "left_ankle_roll_link"
stepHeight = 0.2

rightFootId = rmodel.getFrameId(rightFoot)
leftFootId = rmodel.getFrameId(leftFoot)
q0 = robot.q0
x0 = np.concatenate([q0, np.zeros(rmodel.nv)])
rmodel.defaultState = np.concatenate([q0, np.zeros(rmodel.nv)])

############################# prepare data###########################
contact_data = data_set.parse_contact(leftFoot, rightFoot)
if offset != 0:
    for k, v in data.items():
        vv = v.copy()
        vv[..., 2] -= offset
        data[k] = vv
if interpolate >1:
    interpolated_data =deepcopy(data)
    for k, v in data.items():
        interpolated_data[k] = np.repeat(v, interpolate,
                            axis=0)
else:
    interpolated_data = data

target = {}
for k in data.keys():
    target[k] = []

problems = []

# # Create two contact models used along the motion
contactModelnull = crocoddyl.ContactModelMultiple(state, actuation.nu)
contactModel1FootLeft = crocoddyl.ContactModelMultiple(state, actuation.nu)
contactModel1FootRight = crocoddyl.ContactModelMultiple(state, actuation.nu)
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
contactModel1FootLeft.addContact(leftFoot + "_contact", supportContactModelLeft)
contactModel1FootRight.addContact(rightFoot + "_contact", supportContactModelRight)
contactModel2Feet.addContact(leftFoot + "_contact", supportContactModelLeft)
contactModel2Feet.addContact(rightFoot + "_contact", supportContactModelRight)

################################# COST ########################################
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

########################## Step to init ###############################

frameTrackingActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([1, 1, 1] + [0.0] * 3) ** 2
    )
footTrackingActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([1, 1, 0.1] + [0.1] * 2 + [0.0]) ** 2
    )

problems = []
for ii in range(n_traj):
    for _ in range(interpolate):
        # check contact mode
        if contact_data[leftFoot][ii] > 0:
            if contact_data[rightFoot][ii]>0:
                contact_model = contactModel2Feet
            else:
                contact_model = contactModel1FootLeft
        elif contact_data[rightFoot][ii]>0:
            contact_model = contactModel1FootRight
        else:
            contact_model = contactModelnull
        initcostModel = crocoddyl.CostModelSum(state, actuation.nu)
        for i, k in enumerate(data.keys()):
            act = footTrackingActivation if 'ankle' in k else frameTrackingActivation
            t = data[k][ii]
            # print(k)
            framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                            state, rmodel.getFrameId(k), pinocchio.SE3(np.eye(3), t), actuation.nu)
            frameTrack = crocoddyl.CostModelResidual(
                            state,act, framePlacementResidual
                        )
            initcostModel.addCost(
                            k + "_Track", frameTrack, 1e2
                        )
        initcostModel.addCost("stateReg", xRegCost, 1e-3)
        initcostModel.addCost("ctrlReg", uRegCost, 1e-4)
        initcostModel.addCost("limitCost", limitCost, 1e3)
        dmodelRunning = crocoddyl.DifferentialActionModelContactFwdDynamics(
            state, actuation, contact_model, initcostModel
        )
        runningModel = crocoddyl.IntegratedActionModelEuler(dmodelRunning, DT)
        problems.append(runningModel)


# runningCostModel1.addCost("stateReg", xRegCost, 1e-3)
# runningCostModel1.addCost("ctrlReg", uRegCost, 1e-4)


# dmodelRunning1 = crocoddyl.DifferentialActionModelContactFwdDynamics(
#     state, actuation, contactModel2Feet, runningCostModel1
# )

# runningModel1 = crocoddyl.IntegratedActionModelEuler(dmodelRunning1, DT)

x0 = np.concatenate([q0, pin.utils.zero(state.nv)])

problem = crocoddyl.ShootingProblem(
    x0, problems[:-1], problems[-1]
)

solver = crocoddyl.SolverIntro(problem)
solver.setCallbacks([crocoddyl.CallbackVerbose()])

xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
solver.th_stop = 1e-7
solver.solve(xs, us, 500, False, 1e-9)

for k, v in data.items():
    target[k].extend([v[10]]*(solver.problem.T + 1))


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
            for k, v in target.items():
                target[k] = np.stack(v, 0)
            dd = {}
            # for k, v in data.items():
            #     if 'elbow' in k 
            # print(target[None].shape,
            #  target[None].repeat((solver.problem.T + 1), 1).shape
            # )
            # print( np.concatenate(
            #             [np.repeat(np.array([0.0, 0.4, 0.0]), 2*T, axis=0),
            #             np.repeat(np.array([0.3, 0.15, 0.35]), 2*T, axis=0)
            #             ], axis=0
            #         ).shape)
            display = crocoddyl.MeshcatDisplay(robot,
            targets=interpolated_data)
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        time.sleep(1.0)