from pathlib import Path
from typing import Optional
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import joblib
"""
    G1 loading based on https://github.com/Gepetto/example-robot-data
"""

class G1Loader:
    path: Optional[str] = None
    urdf_filename: str = "g1_29dof_rev_1_0.urdf"
    has_rotor_parameters = False
    free_flyer = True
    model_path = None

    def __init__(self):
        if self.path is None:
            self.path = Path(__file__).parents[2] / 'data/robots/g1'
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

def get_data(data_path: str):
    link_names = {
        "left_hip_pitch_joint": 1,
        "left_hip_roll_joint": 1,
        "left_hip_yaw_joint": 1,
        "left_knee_joint": 4,
        "left_ankle_pitch_joint": 7,
        "left_ankle_roll_joint": 7,
        "right_hip_pitch_joint": 2,
        "right_hip_roll_joint": 2,
        "right_hip_yaw_joint": 2,
        "right_knee_joint": 5,
        "right_ankle_pitch_joint": 8,
        "right_ankle_roll_joint": 8,
        "waist_yaw_joint": 0,
        "waist_roll_joint": 0,
        "waist_pitch_joint": 0,
        "left_shoulder_pitch_joint": 16,
        "left_shoulder_roll_joint": 16,
        "left_shoulder_yaw_joint": 16,
        "left_elbow_joint": 18,
        "right_shoulder_pitch_joint": 17,
        "right_shoulder_roll_joint": 17,
        "right_shoulder_yaw_joint": 17,
        "right_elbow_joint": 19,
    }
    smpl_joint_names = [
        "PELVIS", 
        "L_HIP", 
        "R_HIP", 
        "SPINE1", 
        "L_KNEE", 
        "R_KNEE", 
        "SPINE2", 
        "L_ANKLE", 
        "R_ANKLE", 
        "SPINE3", 
        "L_FOOT", 
        "R_FOOT", 
        "NECK", 
        "L_COLLAR", 
        "R_COLLAR", 
        "HEAD", 
        "L_SHOULDER", 
        "R_SHOULDER", 
        "L_ELBOW", 
        "R_ELBOW", 
        "L_WRIST", 
        "R_WRIST", 
        "L_HAND", 
        "R_HAND", 
    ]
    with open(data_path, "rb") as f:
        data = joblib.load(f)
    smpl_joints = data['smpl_joints']
    print("matched keys")
    for link_name, smpl_idx in link_names.items():
        print(f"{link_name}: {smpl_joint_names[smpl_idx]}")
    ret = {link_name: smpl_joints[:, link_id, :] for link_name, link_id in link_names.items()}
    return ret
        

if __name__ == "__main__":
    data_path = "/root/ddp-cr7/data/traj/ddp_cr7_data.pkl"
    data = get_data(data_path)
    print(data.keys())