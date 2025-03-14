import crocoddyl
import pinocchio as pin
import numpy as np

def createSwingFootModel(
        rmodel,
        state,
        actuation,
        timeStep, supportFootIds,
        comTask=None,
        swingFootTask=None
    ):
        """Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """
        # Creating a 6D multi-contact model, and then including the supporting
        # foot
        
        nu = actuation.nu
       
        contactModel = crocoddyl.ContactModelMultiple(state, nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel6D(
                state,
                i,
                pin.SE3.Identity(),
                pin.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 40.0]),
            )
            contactModel.addContact(
                rmodel.frames[i].name + "_contact", supportContactModel
            )
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(state, nu)
        if isinstance(comTask, np.ndarray):
            comResidual = crocoddyl.ResidualModelCoMPosition(state, comTask, nu)
            comTrack = crocoddyl.CostModelResidual(state, comResidual)
            costModel.addCost("comTrack", comTrack, 1e1)
        for i in supportFootIds:
            cone = crocoddyl.WrenchCone(np.eye(3), 0.7, np.array([0.1, 0.05]))
            wrenchResidual = crocoddyl.ResidualModelContactWrenchCone(
                state, i, cone, nu, True
            )
            wrenchActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            wrenchCone = crocoddyl.CostModelResidual(
                state, wrenchActivation, wrenchResidual
            )
            costModel.addCost(
                rmodel.frames[i].name + "_wrenchCone", wrenchCone, 1e1
            )
        if swingFootTask is not None:
            for i in swingFootTask:
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                    state, i[0], i[1], nu
                )
                footTrack = crocoddyl.CostModelResidual(
                    state, framePlacementResidual
                )
                costModel.addCost(
                    rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e6
                )
        stateWeights = np.array(
            [0] * 3 + [50.0] * 3 + [0.01] * (state.nv - 6) + [10] * state.nv
        )
        stateResidual = crocoddyl.ResidualModelState(
            state, rmodel.defaultState, nu
        )
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(
            state, stateActivation, stateResidual
        )
        ctrlResidual = crocoddyl.ResidualModelControl(state, nu)

        ctrlReg = crocoddyl.CostModelResidual(state, ctrlResidual)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)
        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                state, actuation, contactModel, costModel, 0.0, True
            )
        control = crocoddyl.ControlParametrizationModelPolyZero(nu)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, control, timeStep)
        return model

def createFootstepModels(
        rmodel,
        state,
        actuation,
        feetPos0,
        feetPos1,
        stepHeight,
        timeStep,
        numKnots,
        supportFootIds,
        swingFootIds,
    ):
        """Action models for a footstep phase.

        :param comPos0, initial CoM position
        :param feetPos0: initial position of the swinging feet
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: time step
        :param numKnots: number of knots for the footstep phase
        :param supportFootIds: Ids of the supporting feet
        :param swingFootIds: Ids of the swinging foot
        :return footstep action models
        """
        numLegs = len(supportFootIds) + len(swingFootIds)
        comPercentage = float(len(swingFootIds)) / numLegs
        # Action models for the foot swing
        footSwingModel = []
        for k in range(numKnots):
            swingFootTask = []
            for idx, (i, p) in enumerate(zip(swingFootIds, feetPos0)):
                # Defining a foot swing task given the step length. The swing task
                # is decomposed on two phases: swing-up and swing-down. We decide
                # deliveratively to allocated the same number of nodes (i.e. phKnots)
                # in each phase. With this, we define a proper z-component for the
                # swing-leg motion.
                phKnots = numKnots / 2
                dp = np.zeros_like(p)
                dp[:2] = (feetPos1[idx] - p)[:2] * (k + 1) / numKnots
                if k <= phKnots:
                    # dp = np.array(
                    #     [stepLength * (k + 1) / numKnots, 0.0, stepHeight * k / phKnots]
                    # )
                    dp[2] = stepHeight * k / phKnots
                else:
                    dp[2] = stepHeight * (1 - float(k - phKnots) / phKnots)
                    # dp = np.array(
                    #     [
                    #         stepLength * (k + 1) / numKnots,
                    #         0.0,
                    #         stepHeight * (1 - float(k - phKnots) / phKnots),
                    #     ]
                    # )
                tref = p + dp
                swingFootTask += [[i, pin.SE3(np.eye(3), tref)]]
            # comTask = (
            #     np.array([stepLength * (k + 1) / numKnots, 0.0, 0.0]) * comPercentage
            #     + comPos0
            # )
            comTask=None
            footSwingModel += [
                createSwingFootModel(
                    rmodel,
                    state,
                    actuation,
                    timeStep,
                    supportFootIds,
                    comTask=comTask,
                    swingFootTask=swingFootTask,
                )
            ]
        # Action model for the foot switch
        footSwitchModel = createImpulseModel(rmodel,
                                             state,
                                             swingFootIds,
                                             swingFootTask)
        # Updating the current foot position for next step
        # comPos0 += [stepLength * comPercentage, 0.0, 0.0]
        # for p in feetPos0:
        #     p += [stepLength, 0.0, 0.0]
        return [*footSwingModel, footSwitchModel]

def createImpulseModel(rmodel,
        state, supportFootIds, swingFootTask, JMinvJt_damping=1e-12, r_coeff=0.0
    ):
        """Action model for impulse models.

        An impulse model consists of describing the impulse dynamics against a set of
        contacts.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return impulse action model
        """
        # Creating a 6D multi-contact model, and then including the supporting foot
        impulseModel = crocoddyl.ImpulseModelMultiple(state)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ImpulseModel6D(
                state, i, pin.LOCAL_WORLD_ALIGNED
            )
            impulseModel.addImpulse(
                rmodel.frames[i].name + "_impulse", supportContactModel
            )
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(state, 0)
        if swingFootTask is not None:
            for i in swingFootTask:
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                    state, i[0], i[1], 0
                )
                footTrack = crocoddyl.CostModelResidual(
                    state, framePlacementResidual
                )
                costModel.addCost(
                    rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e3
                )
        stateWeights = np.array(
            [1.0] * 6 + [0.1] * (rmodel.nv - 6) + [10] * rmodel.nv
        )
        stateResidual = crocoddyl.ResidualModelState(
            state, rmodel.defaultState, 0
        )
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(
            state, stateActivation, stateResidual
        )
        costModel.addCost("stateReg", stateReg, 1e1)
        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        model = crocoddyl.ActionModelImpulseFwdDynamics(
            state, impulseModel, costModel
        )
        model.JMinvJt_damping = JMinvJt_damping
        model.r_coeff = r_coeff
        return model