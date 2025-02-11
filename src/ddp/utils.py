from pathlib import Path
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

"""
    G1 loading based on https://github.com/Gepetto/example-robot-data
"""

class G1Loader:
    path: str = '../data/robots/g1'
    urdf_filename: str = "g1_29dof_rev_1_0.urdf"
    has_rotor_parameters = False
    free_flyer = True
    model_path = None

    def __init__(self):
        urdf_path = Path(self.path) / self.urdf_filename
        print(urdf_path)
        if not urdf_path.exists():
            raise ValueError(f"No such {self.urdf_filename} in {Path(self.path).resolve()}")
        
        self.model_path = str(urdf_path.resolve().parent)
        builder = RobotWrapper.BuildFromURDF
        self.robot = builder(
                urdf_path,
                self.model_path,
                pin.JointModelFreeFlyer() if self.free_flyer else None,
            )
        
        self.srdf_path = None
        self.robot.q0 = pin.neutral(self.robot.model)
        self.robot.urdf = urdf_path

        if self.free_flyer:
            self.addFreeFlyerJointLimits()

    def addFreeFlyerJointLimits(self):
        ub = self.robot.model.upperPositionLimit
        ub[:7] = 1
        self.robot.model.upperPositionLimit = ub
        lb = self.robot.model.lowerPositionLimit
        lb[:7] = -1
        self.robot.model.lowerPositionLimit = lb